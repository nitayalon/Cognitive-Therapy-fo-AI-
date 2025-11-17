"""
Training system for LSTM networks on mixed-motive games.

This module implements the main training loop that:
1. Runs T games per partner with the same opponent
2. Manages multiple training episodes with different opponents  
3. Handles convergence criteria and early stopping
4. Tracks training metrics and saves checkpoints
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm
import os
from collections import defaultdict
import random

from .games import MixedMotiveGame, Action, GameFactory
from .opponent import Opponent, OpponentFactory
from .network import GameLSTM, NetworkManager
from .loss import CompositeLoss, AdaptiveLoss
from .tom_rl_loss import ToMRLLoss, AdaptiveToMRLLoss, LossAnalyzer
from .utils import MetricsTracker, set_random_seeds
from .config import TrainingConfig, NetworkConfig
from .training_monitor import TrainingMonitor, BatchedTrainingMonitor
from .testing_monitor import TestingMonitor


class GameSession:
    """
    Represents a single training session between the LSTM network and an opponent.
    
    A session consists of T games played consecutively with the same opponent.
    """
    
    def __init__(
        self,
        game: MixedMotiveGame,
        opponent: Opponent,
        network: GameLSTM,
        num_games: int,
        training_mode: bool = True
    ):
        """
        Initialize a game session.
        
        Args:
            game: The mixed-motive game to play
            opponent: The opponent to play against
            network: The LSTM network
            num_games: Number of games to play in this session (T parameter)
            training_mode: Whether this session is for training (True) or evaluation (False)
        """
        self.game = game
        self.opponent = opponent
        self.network = network
        self.num_games = num_games
        self.training_mode = training_mode
        
        # Session data
        self.session_data = []
        self.cumulative_reward = 0.0
        self.logger = logging.getLogger(__name__)
    
    def play_session(self, device: torch.device) -> Dict[str, Any]:
        """
        Play a complete session of T games.
        
        Args:
            device: Device to run computations on
            
        Returns:
            Dictionary containing session results and training data
        """
        self.game.reset()
        self.opponent.reset()
        self.session_data = []
        self.cumulative_reward = 0.0
        
        # Initialize hidden state for LSTM
        hidden = self.network.init_hidden(1, device)
        
        for game_num in range(self.num_games):
            game_data = self._play_single_game(game_num, hidden, device)
            self.session_data.append(game_data)
            hidden = game_data['new_hidden']
        
        # Compile session results
        session_results = self._compile_session_results()
        return session_results
    
    def _play_single_game(
        self, 
        game_num: int, 
        hidden: Tuple[torch.Tensor, torch.Tensor],
        device: torch.device
    ) -> Dict[str, Any]:
        """
        Play a single game within the session.
        
        Args:
            game_num: Game number within the session
            hidden: Current LSTM hidden state
            device: Device for computations
            
        Returns:
            Dictionary with game data for training
        """
        # Get current state representation
        state_vector = self.game.get_state_vector()
        state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Network forward pass - enable gradients during training, disable during evaluation
        if self.training_mode:
            # Enable gradients for training
            policy_logits, opponent_policy_logits, value_estimate, new_hidden = self.network.forward(
                state_tensor, hidden
            )
            
            # Convert policy logits to action logits for backward compatibility
            # opponent_policy_logits: (batch_size, 2) with raw logits for [defect, cooperate]
            # Convert to action logits: (batch_size, 2) for [Cooperate, Defect] (swapped order)
            opponent_action_logits = torch.cat([
                opponent_policy_logits[:, 1:2],  # Cooperation logit (action 0)  
                opponent_policy_logits[:, 0:1]   # Defection logit (action 1)
            ], dim=1)
        else:
            # Disable gradients for evaluation
            with torch.no_grad():
                policy_logits, opponent_policy_logits, value_estimate, new_hidden = self.network.forward(
                    state_tensor, hidden
                )
                
                # Convert policy logits to action logits for backward compatibility
                # opponent_policy_logits: (batch_size, 2) with raw logits for [defect, cooperate]
                # Convert to action logits: (batch_size, 2) for [Cooperate, Defect] (swapped order)
                opponent_action_logits = torch.cat([
                    opponent_policy_logits[:, 1:2],  # Cooperation logit (action 0)
                    opponent_policy_logits[:, 0:1]   # Defection logit (action 1)
                ], dim=1)
        
        # Sample action from policy
        action_probs = torch.softmax(policy_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action_idx = action_dist.sample()
        action_log_prob = action_dist.log_prob(action_idx)
        
        # Convert to Action enum
        player_action = Action.COOPERATE if action_idx.item() == 0 else Action.DEFECT
        
        # Opponent chooses action
        opponent_action = self.opponent.play_action(self.game.history, game_num)
        
        # Play the round
        player_reward, opponent_reward = self.game.play_round(player_action, opponent_action)
        self.opponent.update_payoff(opponent_reward)
        self.cumulative_reward += player_reward
        
        # Store game data for training
        game_data = {
            'state_vector': state_vector,
            'player_action': player_action,
            'opponent_action': opponent_action,
            'player_reward': player_reward,
            'opponent_reward': opponent_reward,
            'action_log_prob': action_log_prob,
            'policy_logits': policy_logits,
            'opponent_action_logits': opponent_action_logits,
            'opponent_policy_logits': opponent_policy_logits,
            'value_estimate': value_estimate,
            'opponent_type_true': self.opponent.get_type_parameter(),
            'hidden': hidden,
            'new_hidden': new_hidden,
            'game_num': game_num
        }
        
        return game_data
    
    def _compile_session_results(self) -> Dict[str, Any]:
        """Compile results from the entire session."""
        if not self.session_data:
            return {}
        
        # Extract training data
        states = np.stack([data['state_vector'] for data in self.session_data])
        actions = torch.tensor([data['player_action'].value for data in self.session_data])
        opponent_actions = torch.tensor([data['opponent_action'].value for data in self.session_data])
        rewards = torch.tensor([data['player_reward'] for data in self.session_data], dtype=torch.float32)
        
        # Stack network outputs
        policy_logits = torch.cat([data['policy_logits'] for data in self.session_data], dim=0)
        opponent_action_logits = torch.cat([data['opponent_action_logits'] for data in self.session_data], dim=0)
        opponent_policy_logits = torch.cat([data['opponent_policy_logits'] for data in self.session_data], dim=0)
        value_estimates = torch.cat([data['value_estimate'] for data in self.session_data], dim=0)
        
        # True opponent type (same for all games in session)
        opponent_type_true = self.session_data[0]['opponent_type_true']
        if opponent_type_true is not None:
            opponent_type_tensor = torch.full((len(self.session_data), 1), opponent_type_true, dtype=torch.float32)
            # CRITICAL: Create true opponent policy tensor matching network output format
            # Network opponent_policy_head outputs: [defect_logit, cooperate_logit] (index 0=defect, 1=cooperate)
            # True policy should match this format: [p_defect, p_cooperate]
            p_d = opponent_type_true  # defection probability from opponent type
            p_c = 1.0 - p_d  # cooperation probability 
            true_opponent_policy = torch.tensor([[p_d, p_c]], dtype=torch.float32).expand(len(self.session_data), -1)
            
            # Data consistency logging
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Opponent type: {opponent_type_true:.3f} -> True policy: [defect={p_d:.3f}, cooperate={p_c:.3f}]")
        else:
            opponent_type_tensor = torch.zeros((len(self.session_data), 1), dtype=torch.float32)
            # Default uniform policy if opponent type unknown
            true_opponent_policy = torch.tensor([[0.5, 0.5]], dtype=torch.float32).expand(len(self.session_data), -1)
        
        # Session statistics
        cooperation_rate = sum(1 for data in self.session_data if data['player_action'] == Action.COOPERATE) / len(self.session_data)
        opponent_cooperation_rate = sum(1 for data in self.session_data if data['opponent_action'] == Action.COOPERATE) / len(self.session_data)
        
        return {
            'training_data': {
                'states': states,
                'actions': actions,
                'opponent_actions': opponent_actions,
                'rewards': rewards,
                'policy_logits': policy_logits,
                'opponent_action_logits': opponent_action_logits,
                'opponent_policy_logits': opponent_policy_logits,
                'value_estimate': value_estimates,
                'opponent_type_true': opponent_type_tensor,
                'true_opponent_policy': true_opponent_policy
            },
            'session_stats': {
                'cumulative_reward': self.cumulative_reward,
                'average_reward': self.cumulative_reward / len(self.session_data),
                'cooperation_rate': cooperation_rate,
                'opponent_cooperation_rate': opponent_cooperation_rate,
                'num_games': len(self.session_data),
                'opponent_type': opponent_type_true,
                'opponent_name': self.opponent.get_strategy_name()
            }
        }


class GameTrainer:
    """
    Main trainer class for LSTM networks on mixed-motive games.
    
    Manages the complete training process including multiple sessions,
    different opponents, convergence checking, and result tracking.
    """
    
    def __init__(
        self,
        network: GameLSTM,
        training_config: TrainingConfig,
        device: Optional[torch.device] = None,
        use_adaptive_loss: bool = False
    ):
        """
        Initialize the game trainer.
        
        Args:
            network: The LSTM network to train
            training_config: Training configuration parameters
            device: Device for computations
            use_adaptive_loss: Whether to use adaptive loss weighting
        """
        self.network = network
        self.config = training_config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        
        # Initialize loss function - use ToM-RL losses (current framework)
        if use_adaptive_loss:
            self.loss_fn = AdaptiveToMRLLoss(
                alpha=self.config.action_prediction_loss_weight,
                gamma=0.99,
                use_gae=True
            )
        else:
            self.loss_fn = ToMRLLoss(
                alpha=self.config.action_prediction_loss_weight,
                gamma=0.99,
                use_gae=True
            )
        
        # Tracking and analysis
        self.loss_analyzer = LossAnalyzer()
        self.metrics_tracker = MetricsTracker()
        self.network_manager = NetworkManager(self.network, self.device)
        
        # Network serial ID for linking training and testing data
        self.network_serial_id = self._generate_network_serial_id()
        
        # Training monitor for detailed logging (initialized when training starts)
        self.training_monitor: Optional[BatchedTrainingMonitor] = None
        self.detailed_logging_enabled = False
        
        # Testing monitor for evaluation phases
        self.testing_monitor: Optional[TestingMonitor] = None
        self.testing_logging_enabled = False
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Performance tracking across epochs
        self.epoch_rewards = []  # Store cumulative rewards per epoch
        self.epoch_cooperation_rates = []  # Store network cooperation rates per epoch
        
        self.logger = logging.getLogger(__name__)
    
    def save_experiment_metadata(
        self,
        save_dir: str,
        opponents: List[Opponent],
        game_configs: Optional[List[Dict[str, Any]]] = None,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save experiment metadata to a text file for documentation.
        
        Args:
            save_dir: Directory to save the metadata file
            opponents: List of opponents used in the experiment
            game_configs: Game configurations used (if any)
            additional_params: Additional parameters to document
            
        Returns:
            Path to the created metadata file
        """
        import os
        from datetime import datetime
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Create metadata filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_file = os.path.join(save_dir, f"experiment_metadata_{self.network_serial_id}_{timestamp}.txt")
        
        # Extract opponent information
        opponent_probs = []
        opponent_types = set()
        for opp in opponents:
            prob = opp.get_type_parameter()
            if prob is not None:
                opponent_probs.append(prob)
            opponent_types.add(opp.get_strategy_name())
        
        # Calculate opponent statistics
        if opponent_probs:
            min_prob = min(opponent_probs)
            max_prob = max(opponent_probs)
            num_opponents = len(opponents)
            prob_range = max_prob - min_prob
            avg_spacing = prob_range / (num_opponents - 1) if num_opponents > 1 else 0.0
        else:
            min_prob = max_prob = prob_range = avg_spacing = 0.0
            num_opponents = len(opponents)
        
        # Write metadata file
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("EXPERIMENT METADATA\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Experiment Timestamp: {timestamp}\n")
            f.write(f"Network Serial ID: {self.network_serial_id}\n\n")
            
            f.write("OPPONENT CONFIGURATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Opponents: {num_opponents}\n")
            f.write(f"Opponent Types: {', '.join(opponent_types)}\n")
            
            if opponent_probs:
                f.write(f"Defection Probability Range: [{min_prob:.3f}, {max_prob:.3f}]\n")
                f.write(f"Probability Range Span: {prob_range:.3f}\n")
                f.write(f"Average Spacing: {avg_spacing:.3f}\n\n")
                
                f.write("Individual Opponent Defection Probabilities:\n")
                sorted_probs = sorted(opponent_probs)
                for i, prob in enumerate(sorted_probs):
                    f.write(f"  {i+1:2d}. {prob:.3f}\n")
                f.write("\n")
            
            f.write("NETWORK CONFIGURATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Hidden Size: {self.network.hidden_size}\n")
            f.write(f"Number of Layers: {self.network.num_layers}\n")
            f.write(f"Input Size: {self.network.input_size}\n")
            f.write(f"Dropout: {getattr(self.network, 'dropout', 'N/A')}\n\n")
            
            f.write("TRAINING CONFIGURATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Learning Rate: {self.config.learning_rate}\n")
            f.write(f"Max Epochs: {self.config.max_epochs}\n")
            f.write(f"Batch Size: {self.config.batch_size}\n")
            f.write(f"Games per Partner: {self.config.num_games_per_partner}\n")
            f.write(f"Patience: {self.config.patience}\n")
            f.write(f"Convergence Threshold: {self.config.convergence_threshold}\n")
            f.write(f"Loss Type: {type(self.loss_fn).__name__}\n")
            f.write(f"Loss Alpha: {getattr(self.loss_fn, 'alpha', 'N/A')}\n\n")
            
            if game_configs:
                f.write("GAME CONFIGURATION\n")
                f.write("-"*40 + "\n")
                for i, config in enumerate(game_configs):
                    f.write(f"Game {i+1}: {config.get('name', 'Unknown')}\n")
                    f.write(f"  Weight: {config.get('weight', 1.0)}\n")
                    if 'kwargs' in config:
                        f.write(f"  Parameters: {config['kwargs']}\n")
                f.write("\n")
            
            if additional_params:
                f.write("ADDITIONAL PARAMETERS\n")
                f.write("-"*40 + "\n")
                for key, value in additional_params.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            f.write("DEVICE INFORMATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"CUDA Device Count: {torch.cuda.device_count()}\n")
                f.write(f"Current CUDA Device: {torch.cuda.current_device()}\n")
            f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("END OF METADATA\n")
            f.write("="*80 + "\n")
        
        self.logger.info(f"Experiment metadata saved to: {metadata_file}")
        return metadata_file
    
    def _generate_network_serial_id(self) -> str:
        """Generate a unique serial ID for the network to link training and testing data."""
        import uuid
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"NET_{timestamp}_{unique_id}"
    
    def get_network_serial_id(self) -> str:
        """Get the network's unique serial ID."""
        return self.network_serial_id
    
    def enable_detailed_logging(self, output_dir: str, save_frequency: int = 50):
        """
        Enable detailed training monitoring and logging.
        
        Args:
            output_dir: Directory to save detailed training logs
            save_frequency: How often to save logs to disk (iterations)
        """
        self.detailed_logging_enabled = True
        self.training_monitor = BatchedTrainingMonitor(output_dir, save_frequency)
        # Pass network serial ID to training monitor
        self.training_monitor.network_serial_id = self.network_serial_id
        self.logger.info(f"Detailed training logging enabled for network {self.network_serial_id}. Logs will be saved to {output_dir}")
    
    def disable_detailed_logging(self):
        """Disable detailed training monitoring."""
        if self.training_monitor is not None:
            self.training_monitor.finalize()
        self.detailed_logging_enabled = False
        self.training_monitor = None
        self.logger.info("Detailed training logging disabled")
    
    def enable_testing_monitoring(self, output_dir: str, save_frequency: int = 25):
        """
        Enable detailed testing monitoring and logging.
        
        Args:
            output_dir: Directory to save detailed testing logs
            save_frequency: How often to save logs to disk (iterations)
        """
        self.testing_logging_enabled = True
        self.testing_monitor = TestingMonitor(output_dir, self.network_serial_id, save_frequency)
        self.logger.info(f"Detailed testing logging enabled for network {self.network_serial_id}. Logs will be saved to {output_dir}")
    
    def disable_testing_monitoring(self):
        """Disable detailed testing monitoring."""
        if self.testing_monitor is not None:
            self.testing_monitor.finalize()
        self.testing_logging_enabled = False
        self.testing_monitor = None
        self.logger.info("Detailed testing logging disabled")
    
    def train_on_game(
        self,
        game_name: str,
        opponents: List[Opponent],
        game_kwargs: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the network on a specific game with multiple opponents.
        
        Args:
            game_name: Name of the game to train on
            opponents: List of opponents to train against
            game_kwargs: Optional arguments for game initialization
            save_dir: Directory to save checkpoints and results
            
        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Starting training on {game_name} with {len(opponents)} opponents")
        
        # Enable detailed logging if save_dir is provided
        if save_dir and not self.detailed_logging_enabled:
            detailed_log_dir = os.path.join(save_dir, 'detailed_training_logs')
            self.enable_detailed_logging(detailed_log_dir)
        
        # Save experiment metadata if save_dir is provided
        if save_dir:
            game_config = [{'name': game_name, 'kwargs': game_kwargs or {}, 'weight': 1.0}]
            self.save_experiment_metadata(save_dir, opponents, game_config)
        
        # Create game instance
        game_kwargs = game_kwargs or {}
        game = GameFactory.create_game(game_name, **game_kwargs)
        
        # Training results
        training_results = {
            'game_name': game_name,
            'opponents': [opp.get_strategy_name() for opp in opponents],
            'epoch_results': [],
            'final_metrics': {}
        }
        
        # Training loop
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            epoch_results = self._train_epoch(game, opponents)
            training_results['epoch_results'].append(epoch_results)
            
            # Track metrics
            self._update_metrics(epoch_results, epoch)
            
            # Check convergence
            if self._check_convergence():
                self.logger.info(f"Training converged at epoch {epoch}")
                break
            
            # Save checkpoint
            if save_dir and epoch % 50 == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
                self.network_manager.save_checkpoint(
                    checkpoint_path, 
                    self.optimizer.state_dict(), 
                    epoch, 
                    epoch_results['total_loss']
                )
        
        # Final results
        training_results['final_metrics'] = self._compile_final_metrics()
        
        # Save final model
        if save_dir:
            final_model_path = os.path.join(save_dir, "final_model.pt")
            self.network_manager.save_checkpoint(
                final_model_path,
                self.optimizer.state_dict(),
                self.current_epoch,
                self.best_loss
            )
        
        # Finalize detailed logging
        if self.detailed_logging_enabled:
            self.disable_detailed_logging()
        
        return training_results

    def train_on_multiple_games(
        self,
        game_configs: List[Dict[str, Any]],
        opponents: List[Opponent],
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the network on multiple games simultaneously.
        
        Each training epoch samples from all games, creating mixed batches that
        expose the network to different payoff structures and strategic contexts
        within the same training step.
        
        Args:
            game_configs: List of game configurations, each containing:
                - 'name': Game name (str)
                - 'kwargs': Game initialization parameters (dict, optional)
                - 'weight': Sampling weight for this game (float, optional, default=1.0)
            opponents: List of opponents to train against across all games
            save_dir: Directory to save checkpoints and results
            
        Returns:
            Dictionary with multi-game training results
        """
        self.logger.info(f"Starting simultaneous training on {len(game_configs)} games")
        
        # Enable detailed logging if save_dir is provided
        if save_dir and not self.detailed_logging_enabled:
            detailed_log_dir = os.path.join(save_dir, 'detailed_training_logs')
            self.enable_detailed_logging(detailed_log_dir, save_frequency=25)  # More frequent saves for multi-game
        
        # Save experiment metadata if save_dir is provided
        if save_dir:
            self.save_experiment_metadata(save_dir, opponents, game_configs)
        
        # Initialize games
        games = {}
        game_weights = {}
        total_weight = 0
        
        for config in game_configs:
            game_name = config['name']
            game_kwargs = config.get('kwargs', {})
            weight = config.get('weight', 1.0)
            
            games[game_name] = GameFactory.create_game(game_name, **game_kwargs)
            game_weights[game_name] = weight
            total_weight += weight
            
            self.logger.info(f"  - {game_name}: weight={weight}")
        
        # Normalize weights
        for game_name in game_weights:
            game_weights[game_name] /= total_weight
        
        # Training results
        training_results = {
            'games': list(games.keys()),
            'game_weights': game_weights,
            'opponents': [opp.get_strategy_name() for opp in opponents],
            'epoch_results': [],
            'final_metrics': {}
        }
        
        # Training loop
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            epoch_results = self._train_multi_game_epoch(games, game_weights, opponents)
            training_results['epoch_results'].append(epoch_results)
            
            # Track metrics
            self._update_multi_game_metrics(epoch_results, epoch)
            
            # Check convergence (using total loss)
            if self._check_convergence():
                self.logger.info(f"Multi-game training converged at epoch {epoch}")
                break
            
            # Save checkpoint
            if save_dir and epoch % 50 == 0:
                checkpoint_path = os.path.join(save_dir, f"multi_game_checkpoint_epoch_{epoch}.pt")
                self.network_manager.save_checkpoint(
                    checkpoint_path, 
                    self.optimizer.state_dict(), 
                    epoch, 
                    epoch_results['total_loss']
                )
            
            # Interim summary report every 100 epochs
            if epoch > 0 and epoch % 100 == 0:
                self._log_interim_summary(training_results, epoch)
        
        # Final results
        training_results['final_metrics'] = self._compile_multi_game_final_metrics(games)
        
        # Save final model
        if save_dir:
            final_model_path = os.path.join(save_dir, "multi_game_final_model.pt")
            self.network_manager.save_checkpoint(
                final_model_path,
                self.optimizer.state_dict(),
                self.current_epoch,
                self.best_loss
            )
        
        # Finalize detailed logging
        if self.detailed_logging_enabled:
            self.disable_detailed_logging()
        
        return training_results

    def _update_multi_game_metrics(self, epoch_results: Dict[str, Any], epoch: int):
        """Update metrics for multi-game training."""
        # Track overall loss using ToM-RL LossAnalyzer format
        total_loss = epoch_results['total_loss']
        
        # Extract loss components from mixed_batch_loss if available
        if 'mixed_batch_loss' in epoch_results and isinstance(epoch_results['mixed_batch_loss'], dict):
            loss_source = epoch_results['mixed_batch_loss']
        else:
            loss_source = epoch_results
            
        # Build loss dictionary for ToM-RL LossAnalyzer
        loss_dict = {
            'total_loss': torch.tensor(total_loss),
            'rl_loss': torch.tensor(loss_source.get('rl_loss', 0.0)),
            'opponent_policy_loss': torch.tensor(loss_source.get('opponent_policy_loss', 0.0)),
            'alpha': loss_source.get('alpha', 1.0)
        }
        
        # Add normalized losses if available
        if 'rl_loss_normalized' in loss_source:
            loss_dict['rl_loss_normalized'] = torch.tensor(loss_source['rl_loss_normalized'])
            loss_dict['opponent_policy_loss_normalized'] = torch.tensor(loss_source['opponent_policy_loss_normalized'])
        
        self.loss_analyzer.record_loss(loss_dict, epoch)
        
        # Update best loss
        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

    def _compile_multi_game_final_metrics(self, games: Dict[str, MixedMotiveGame]) -> Dict[str, Any]:
        """Compile final metrics for multi-game training."""
        # Use ToM-RL LossAnalyzer methods or create fallback metrics
        if hasattr(self.loss_analyzer, 'get_convergence_info'):
            convergence_info = self.loss_analyzer.get_convergence_info()
        else:
            convergence_info = {
                'converged': self.patience_counter >= self.config.patience,
                'final_loss': self.best_loss,
                'epochs_trained': self.current_epoch + 1
            }
        
        if hasattr(self.loss_analyzer, 'get_loss_statistics'):
            loss_stats = self.loss_analyzer.get_loss_statistics()
        else:
            loss_stats = {
                'loss_balance_ratio': self.loss_analyzer.get_loss_balance_ratio() if hasattr(self.loss_analyzer, 'get_loss_balance_ratio') else 1.0,
                'tom_contribution': self.loss_analyzer.get_tom_contribution() if hasattr(self.loss_analyzer, 'get_tom_contribution') else 0.0,
                'total_loss_history': self.loss_analyzer.loss_history.get('total', []),
                'rl_loss_history': self.loss_analyzer.loss_history.get('rl', []),
                'opponent_policy_history': self.loss_analyzer.loss_history.get('opponent_policy', [])
            }
        
        # Add game-specific information
        game_info = {}
        for game_name, game in games.items():
            game_info[game_name] = {
                'payoff_matrix': game.get_payoff_matrix().tolist(),
                'state_size': game.get_state_size()
            }
        
        # Calculate reward and cooperation statistics
        reward_stats = {
            'cumulative_rewards_per_epoch': self.epoch_rewards,
            'average_reward_per_epoch': [r / max(1, len(games) * self.config.__dict__.get('num_games_per_partner', 1)) for r in self.epoch_rewards],
            'total_cumulative_reward': sum(self.epoch_rewards),
            'mean_reward_across_epochs': np.mean(self.epoch_rewards) if self.epoch_rewards else 0.0,
            'final_epoch_reward': self.epoch_rewards[-1] if self.epoch_rewards else 0.0
        }
        
        cooperation_stats = {
            'cooperation_rates_per_epoch': self.epoch_cooperation_rates,
            'mean_cooperation_rate': np.mean(self.epoch_cooperation_rates) if self.epoch_cooperation_rates else 0.0,
            'final_cooperation_rate': self.epoch_cooperation_rates[-1] if self.epoch_cooperation_rates else 0.0,
            'cooperation_trend': 'increasing' if len(self.epoch_cooperation_rates) > 1 and self.epoch_cooperation_rates[-1] > self.epoch_cooperation_rates[0] else 'decreasing' if len(self.epoch_cooperation_rates) > 1 else 'stable'
        }
        
        return {
            'total_epochs': self.current_epoch + 1,
            'best_loss': self.best_loss,
            'convergence_info': convergence_info,
            'loss_statistics': loss_stats,
            'games_info': game_info,
            'reward_statistics': reward_stats,
            'cooperation_statistics': cooperation_stats,
            'final_weights': {
                'alpha': self.loss_fn.alpha,
                'gamma': self.loss_fn.gamma,
                'use_gae': self.loss_fn.use_gae
            }
        }
    
    def _train_epoch(self, game: MixedMotiveGame, opponents: List[Opponent]) -> Dict[str, Any]:
        """
        Train for one epoch across all opponents.
        
        Args:
            game: Game to play
            opponents: List of opponents to train against
            
        Returns:
            Dictionary with epoch results
        """
        self.network.train()
        epoch_losses = []
        epoch_session_stats = []
        
        # Shuffle opponents for this epoch
        random.shuffle(opponents)
        
        for opponent in opponents:
            # Play session with this opponent (training mode)
            session = GameSession(game, opponent, self.network, self.config.num_games_per_partner, training_mode=True)
            session_results = session.play_session(self.device)
            
            if not session_results:
                continue
            
            # Extract training data
            training_data = session_results['training_data']
            session_stats = session_results['session_stats']
            
            # Calculate loss using ToM-RL loss function
            try:
                loss_dict = self.loss_fn(
                    policy_logits=training_data['policy_logits'].to(self.device),
                    opponent_policy_logits=training_data['opponent_policy_logits'].to(self.device),
                    value_estimates=training_data['value_estimate'].to(self.device),
                    actions_taken=training_data['actions'].to(self.device),
                    rewards=training_data['rewards'].to(self.device),
                    opponent_actions=training_data['opponent_actions'].to(self.device),
                    true_opponent_policy=training_data['true_opponent_policy'].to(self.device)
                )
            except Exception as e:
                self.logger.error(f"Loss calculation failed: {e}")
                self.logger.error(f"Training data shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in training_data.items()]}")
                raise
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # Gradient clipping and record gradient norm
            gradient_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Detailed logging if enabled
            if self.detailed_logging_enabled and self.training_monitor is not None:
                # Update gradient norm for logging
                self.training_monitor.update_gradient_norm(gradient_norm.item())
                
                # Log the complete training batch
                self.training_monitor.log_training_batch(
                    loss_dict=loss_dict,
                    training_data=training_data,
                    session_results=session_results,
                    epoch=self.current_epoch,
                    game_name=game.get_name(),
                    opponent_name=opponent.get_strategy_name()
                )
            
            # Record results (handle multi-element tensors properly)
            epoch_loss_record = {}
            for k, v in loss_dict.items():
                if torch.is_tensor(v):
                    if v.numel() == 1:  # Scalar tensor
                        epoch_loss_record[k] = v.item()
                    else:  # Multi-element tensor (like advantages)
                        if k == 'advantages':
                            epoch_loss_record[k + '_mean'] = v.mean().item()
                            epoch_loss_record[k + '_std'] = v.std().item()
                        elif k == 'normalization_stats' and isinstance(v, dict):
                            # Handle nested dictionary - flatten it
                            for stat_key, stat_val in v.items():
                                if torch.is_tensor(stat_val):
                                    epoch_loss_record[f'{k}_{stat_key}'] = stat_val.item()
                                else:
                                    epoch_loss_record[f'{k}_{stat_key}'] = stat_val
                        else:
                            epoch_loss_record[k] = v.mean().item()  # Default: take mean
                else:
                    epoch_loss_record[k] = v
            
            epoch_losses.append(epoch_loss_record)
            epoch_session_stats.append(session_stats)
            
            # Update adaptive loss if applicable
            if hasattr(self.loss_fn, 'update_weights'):
                self.loss_fn.update_weights(loss_dict)
        
        # Compile epoch results
        epoch_results = self._compile_epoch_results(epoch_losses, epoch_session_stats)
        return epoch_results

    def _train_multi_game_epoch(
        self, 
        games: Dict[str, MixedMotiveGame], 
        game_weights: Dict[str, float],
        opponents: List[Opponent]
    ) -> Dict[str, Any]:
        """
        Train for one epoch with simultaneous multi-game learning.
        
        Instead of training on one game at a time, this method creates mixed batches
        where different training samples come from different games, allowing the
        network to learn representations that work across multiple strategic contexts.
        
        Args:
            games: Dictionary of game_name -> game_instance
            game_weights: Dictionary of game weights (maintained for compatibility, no longer used for sampling)
            opponents: List of opponents to train against
            
        Returns:
            Dictionary with epoch results including per-game breakdowns
        """
        self.network.train()
        
        # Collect training data from all games
        all_training_data = []
        game_session_stats = {game_name: [] for game_name in games.keys()}
        
        # Shuffle opponents for this epoch
        random.shuffle(opponents)
        
        # Generate training sessions across all games (deterministic coverage)
        for opponent in opponents:
            # Train on all games deterministically (no sampling)
            game_names = list(games.keys())
            # Note: game_weights no longer used for probabilistic sampling
            
            # Each opponent plays sessions in ALL games (deterministic coverage)
            for game_name in game_names:
                # Always train on every game - no probabilistic sampling
                game = games[game_name]
                
                # Play session with this opponent on this game (training mode)
                session = GameSession(game, opponent, self.network, self.config.num_games_per_partner, training_mode=True)
                session_results = session.play_session(self.device)
                
                if session_results:
                    # Annotate training data with game information
                    training_data = session_results['training_data']
                    training_data['game_name'] = game_name
                    training_data['opponent_name'] = opponent.get_strategy_name()
                    
                    all_training_data.append(training_data)
                    game_session_stats[game_name].append(session_results['session_stats'])
        
        if not all_training_data:
            self.logger.warning("No training data generated in this epoch - this should not occur with deterministic game coverage")
            return {'total_loss': float('inf'), 'game_losses': {}}
        
        # Log training volume for monitoring
        expected_sessions = len(opponents) * len(games)
        actual_sessions = len(all_training_data)
        self.logger.debug(f"Deterministic training: {actual_sessions}/{expected_sessions} sessions generated "
                         f"({len(games)} games × {len(opponents)} opponents)")
        
        # Create mixed batches from all games
        mixed_batch = self._create_mixed_batch(all_training_data)
        
        # Calculate loss on mixed batch using ToM-RL loss function
        try:
            loss_dict = self.loss_fn(
                policy_logits=mixed_batch['policy_logits'].to(self.device),
                opponent_policy_logits=mixed_batch['opponent_policy_logits'].to(self.device),
                value_estimates=mixed_batch['value_estimate'].to(self.device),
                actions_taken=mixed_batch['actions'].to(self.device),
                rewards=mixed_batch['rewards'].to(self.device),
                opponent_actions=mixed_batch['opponent_actions'].to(self.device),
                true_opponent_policy=mixed_batch['true_opponent_policy'].to(self.device)
            )
        except Exception as e:
            self.logger.error(f"Mixed batch loss calculation failed: {e}")
            self.logger.error(f"Mixed batch shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in mixed_batch.items()]}")
            raise
        
        # Backward pass on mixed batch
        self.optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        
        # Gradient clipping and record gradient norm
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Detailed logging if enabled (log each training data point in the batch)
        if self.detailed_logging_enabled and self.training_monitor is not None:
            # Update gradient norm for logging
            self.training_monitor.update_gradient_norm(gradient_norm.item())
            
            # Log each training data point from all games
            for training_data in all_training_data:
                # Create dummy session results for the monitor
                dummy_session_results = {
                    'session_stats': {
                        'opponent_type': 0.5,  # Default value
                        'opponent_name': training_data.get('opponent_name', 'unknown')
                    }
                }
                
                # Log this training batch
                self.training_monitor.log_training_batch(
                    loss_dict=loss_dict,
                    training_data=training_data,
                    session_results=dummy_session_results,
                    epoch=self.current_epoch,
                    game_name=training_data.get('game_name', 'unknown'),
                    opponent_name=training_data.get('opponent_name', 'unknown')
                )
        
        # Calculate per-game loss breakdown for analysis
        per_game_losses = self._calculate_per_game_losses(all_training_data, games)
        
        # Compile epoch results (handle multi-element tensors properly)
        mixed_batch_loss_record = {}
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                if v.numel() == 1:  # Scalar tensor
                    mixed_batch_loss_record[k] = v.item()
                else:  # Multi-element tensor (like advantages)
                    if k == 'advantages':
                        mixed_batch_loss_record[k + '_mean'] = v.mean().item()
                        mixed_batch_loss_record[k + '_std'] = v.std().item()
                    elif k == 'normalization_stats' and isinstance(v, dict):
                        # Handle nested dictionary - flatten it
                        for stat_key, stat_val in v.items():
                            if torch.is_tensor(stat_val):
                                mixed_batch_loss_record[f'{k}_{stat_key}'] = stat_val.item()
                            else:
                                mixed_batch_loss_record[f'{k}_{stat_key}'] = stat_val
                    else:
                        mixed_batch_loss_record[k] = v.mean().item()  # Default: take mean
            else:
                mixed_batch_loss_record[k] = v
        
        # Calculate epoch-level reward and cooperation metrics
        epoch_cumulative_reward = 0.0
        epoch_cooperation_rates = []
        
        for game_stats_list in game_session_stats.values():
            for stats in game_stats_list:
                epoch_cumulative_reward += stats.get('cumulative_reward', 0.0)
                epoch_cooperation_rates.append(stats.get('cooperation_rate', 0.0))
        
        # Calculate average policy probabilities for interim reporting
        with torch.no_grad():
            # Agent policy probabilities (from mixed batch)
            agent_policy_probs = torch.softmax(mixed_batch['policy_logits'], dim=-1)
            avg_agent_coop_prob = agent_policy_probs[:, 0].mean().item()  # Cooperation probability (action 0)
            avg_agent_defect_prob = agent_policy_probs[:, 1].mean().item()  # Defection probability (action 1)
            
            # Predicted opponent policy probabilities (from mixed batch)
            pred_opp_policy_probs = torch.softmax(mixed_batch['opponent_policy_logits'], dim=-1)
            avg_pred_opp_defect_prob = pred_opp_policy_probs[:, 0].mean().item()  # Defection probability (output 0)
            avg_pred_opp_coop_prob = pred_opp_policy_probs[:, 1].mean().item()  # Cooperation probability (output 1)
        
        # Track epoch-level metrics
        avg_cooperation_rate = np.mean(epoch_cooperation_rates) if epoch_cooperation_rates else 0.0
        self.epoch_rewards.append(epoch_cumulative_reward)
        self.epoch_cooperation_rates.append(avg_cooperation_rate)
        
        epoch_results = {
            'total_loss': loss_dict['total_loss'].item(),
            'mixed_batch_loss': mixed_batch_loss_record,
            'per_game_losses': per_game_losses,
            'game_session_stats': game_session_stats,
            'num_sessions_per_game': {game: len(stats) for game, stats in game_session_stats.items()},
            'total_sessions': len(all_training_data),
            'epoch_cumulative_reward': epoch_cumulative_reward,
            'epoch_average_cooperation_rate': avg_cooperation_rate,
            'agent_policy_probs': {
                'cooperate': avg_agent_coop_prob,
                'defect': avg_agent_defect_prob
            },
            'predicted_opponent_policy_probs': {
                'cooperate': avg_pred_opp_coop_prob,
                'defect': avg_pred_opp_defect_prob
            }
        }
        
        # Update adaptive loss if applicable
        if hasattr(self.loss_fn, 'update_weights'):
            self.loss_fn.update_weights(loss_dict)
        
        return epoch_results

    def _create_mixed_batch(self, training_data_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Create a mixed batch from training data from multiple games.
        
        Concatenates tensors from different games/sessions into unified batches
        for simultaneous training across strategic contexts.
        
        Args:
            training_data_list: List of training data dictionaries from different sessions
            
        Returns:
            Dictionary with concatenated tensors for batch training
        """
        if not training_data_list:
            raise ValueError("Cannot create batch from empty training data")
        
        # Initialize batch with first item structure
        batch = {}
        tensor_keys = ['policy_logits', 'opponent_action_logits', 'opponent_policy_logits', 'value_estimate', 
                       'actions', 'rewards', 'opponent_actions', 'opponent_type_true', 'true_opponent_policy']
        
        for key in tensor_keys:
            tensors_to_concat = []
            
            for data in training_data_list:
                if key in data and data[key] is not None:
                    tensor = data[key]
                    tensors_to_concat.append(tensor)
            
            if tensors_to_concat:
                # Concatenate along the appropriate dimension based on tensor type
                if key in ['actions', 'rewards', 'opponent_actions']:
                    # These should remain 1D after concatenation for loss functions
                    # Each session contributes (T,) -> concatenate to (total_T,)
                    batch[key] = torch.cat(tensors_to_concat, dim=0)
                elif key in ['policy_logits', 'opponent_action_logits', 'opponent_policy_logits', 'value_estimate', 
                           'opponent_type_true', 'true_opponent_policy']:
                    # These should be 2D after concatenation 
                    # Each session contributes (T, features) -> concatenate to (total_T, features)
                    batch[key] = torch.cat(tensors_to_concat, dim=0)
                else:
                    # Default concatenation
                    batch[key] = torch.cat(tensors_to_concat, dim=0)
            else:
                self.logger.warning(f"No data found for key: {key}")
        
        return batch

    def _calculate_per_game_losses(
        self, 
        training_data_list: List[Dict[str, Any]], 
        games: Dict[str, MixedMotiveGame]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate loss breakdown per game for analysis purposes.
        
        Args:
            training_data_list: List of training data with game annotations
            games: Dictionary of available games
            
        Returns:
            Dictionary mapping game_name -> loss_components
        """
        per_game_losses = {}
        
        # Group training data by game
        game_data = {game_name: [] for game_name in games.keys()}
        for data in training_data_list:
            game_name = data.get('game_name')
            if game_name in game_data:
                game_data[game_name].append(data)
        
        # Calculate loss for each game separately
        for game_name, data_list in game_data.items():
            if not data_list:
                per_game_losses[game_name] = {'total_loss': 0.0}
                continue
            
            try:
                game_batch = self._create_mixed_batch(data_list)
                
                with torch.no_grad():
                    loss_dict = self.loss_fn(
                        policy_logits=game_batch['policy_logits'].to(self.device),
                        opponent_policy_logits=game_batch['opponent_policy_logits'].to(self.device),
                        value_estimates=game_batch['value_estimate'].to(self.device),
                        actions_taken=game_batch['actions'].to(self.device),
                        rewards=game_batch['rewards'].to(self.device),
                        opponent_actions=game_batch['opponent_actions'].to(self.device),
                        true_opponent_policy=game_batch['true_opponent_policy'].to(self.device)
                    )
                
                # Handle multi-element tensors properly
                per_game_loss_record = {}
                for k, v in loss_dict.items():
                    if torch.is_tensor(v):
                        if v.numel() == 1:  # Scalar tensor
                            per_game_loss_record[k] = v.item()
                        else:  # Multi-element tensor (like advantages)
                            if k == 'advantages':
                                per_game_loss_record[k + '_mean'] = v.mean().item()
                                per_game_loss_record[k + '_std'] = v.std().item()
                            elif k == 'normalization_stats' and isinstance(v, dict):
                                # Handle nested dictionary - flatten it
                                for stat_key, stat_val in v.items():
                                    if torch.is_tensor(stat_val):
                                        per_game_loss_record[f'{k}_{stat_key}'] = stat_val.item()
                                    else:
                                        per_game_loss_record[f'{k}_{stat_key}'] = stat_val
                            else:
                                per_game_loss_record[k] = v.mean().item()  # Default: take mean
                    else:
                        per_game_loss_record[k] = v
                
                per_game_losses[game_name] = per_game_loss_record
                
            except Exception as e:
                self.logger.warning(f"Could not calculate loss for {game_name}: {e}")
                per_game_losses[game_name] = {'total_loss': float('inf')}
        
        return per_game_losses
    
    def _compile_epoch_results(
        self, 
        epoch_losses: List[Dict[str, float]], 
        session_stats: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compile results from all sessions in an epoch."""
        if not epoch_losses:
            return {}
        
        # Average losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            if isinstance(epoch_losses[0][key], (int, float)):
                avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
        
        # Average session statistics
        avg_stats = {}
        epoch_cumulative_reward = 0.0
        epoch_cooperation_rates = []
        
        if session_stats:
            for key in session_stats[0].keys():
                if isinstance(session_stats[0][key], (int, float)):
                    avg_stats[key] = np.mean([stats[key] for stats in session_stats])
            
            # Extract reward and cooperation metrics for epoch tracking
            epoch_cumulative_reward = sum([stats.get('cumulative_reward', 0.0) for stats in session_stats])
            epoch_cooperation_rates = [stats.get('cooperation_rate', 0.0) for stats in session_stats]
        
        # Track epoch-level metrics
        self.epoch_rewards.append(epoch_cumulative_reward)
        self.epoch_cooperation_rates.append(np.mean(epoch_cooperation_rates) if epoch_cooperation_rates else 0.0)
        
        return {
            **avg_losses,
            'session_stats': avg_stats,
            'num_sessions': len(epoch_losses),
            'epoch_cumulative_reward': epoch_cumulative_reward,
            'epoch_average_cooperation_rate': np.mean(epoch_cooperation_rates) if epoch_cooperation_rates else 0.0
        }
    
    def _update_metrics(self, epoch_results: Dict[str, Any], epoch: int):
        """Update metrics tracking."""
        if not epoch_results:
            return
        
        # Record losses (ToM-RL structure) - use float values directly
        # Handle both direct values and nested structures from mixed_batch_loss
        if 'mixed_batch_loss' in epoch_results and isinstance(epoch_results['mixed_batch_loss'], dict):
            loss_source = epoch_results['mixed_batch_loss']
        else:
            loss_source = epoch_results
            
        loss_data = {
            'total_loss': epoch_results.get('total_loss', 0.0),
            'rl_loss': loss_source.get('rl_loss', 0.0),
            'opponent_policy_loss': loss_source.get('opponent_policy_loss', 0.0),
            'rl_loss_normalized': loss_source.get('rl_loss_normalized', 0.0),
            'opponent_policy_loss_normalized': loss_source.get('opponent_policy_loss_normalized', 0.0)
        }
        
        # Convert to tensors for loss_analyzer
        loss_tensors = {k: torch.tensor(v) if isinstance(v, (int, float)) else v 
                       for k, v in loss_data.items()}
        self.loss_analyzer.record_loss(loss_tensors, epoch)
        
        # Add metrics to tracker (only numeric values)
        for key, value in epoch_results.items():
            if isinstance(value, (int, float)):
                self.metrics_tracker.add_metric(key, value, epoch)
            elif key.endswith('_mean') or key.endswith('_std'):
                # Include advantage statistics
                self.metrics_tracker.add_metric(key, value, epoch)
    
    def _check_convergence(self) -> bool:
        """Check if training has converged."""
        # DISABLED: Early termination criteria for full epoch training
        # Still track loss improvement for metrics, but don't terminate early
        current_loss = self.loss_analyzer.loss_history['total']
        if current_loss:
            latest_loss = current_loss[-1]
            if latest_loss < self.best_loss:
                self.best_loss = latest_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        # DISABLED: All early termination conditions
        # Training will run for the full number of epochs (config.max_epochs)
        
        # Original convergence checks (commented out):
        # - Patience-based early stopping
        # - Loss variance convergence 
        # - LossAnalyzer convergence detection
        
        return False  # Never terminate early - always train full epochs
    
    def _compile_final_metrics(self) -> Dict[str, Any]:
        """Compile final training metrics."""
        # Use ToM-RL LossAnalyzer methods or create fallback metrics
        if hasattr(self.loss_analyzer, 'get_convergence_info'):
            convergence_info = self.loss_analyzer.get_convergence_info()
        else:
            # Fallback convergence info
            convergence_info = {
                'converged': self.patience_counter >= self.config.patience,
                'final_loss': self.best_loss,
                'epochs_trained': self.current_epoch + 1
            }
        
        if hasattr(self.loss_analyzer, 'get_loss_statistics'):
            loss_stats = self.loss_analyzer.get_loss_statistics()
        else:
            # Fallback loss statistics using ToM-RL methods
            loss_stats = {
                'loss_balance_ratio': self.loss_analyzer.get_loss_balance_ratio() if hasattr(self.loss_analyzer, 'get_loss_balance_ratio') else 1.0,
                'tom_contribution': self.loss_analyzer.get_tom_contribution() if hasattr(self.loss_analyzer, 'get_tom_contribution') else 0.0,
                'total_loss_history': self.loss_analyzer.loss_history.get('total', []),
                'rl_loss_history': self.loss_analyzer.loss_history.get('rl', []),
                'opponent_policy_history': self.loss_analyzer.loss_history.get('opponent_policy', [])
            }
        
        # Calculate reward and cooperation statistics
        reward_stats = {
            'cumulative_rewards_per_epoch': self.epoch_rewards,
            'average_reward_per_epoch': [r / max(1, self.config.__dict__.get('num_games_per_partner', 1)) for r in self.epoch_rewards],
            'total_cumulative_reward': sum(self.epoch_rewards),
            'mean_reward_across_epochs': np.mean(self.epoch_rewards) if self.epoch_rewards else 0.0,
            'final_epoch_reward': self.epoch_rewards[-1] if self.epoch_rewards else 0.0
        }
        
        cooperation_stats = {
            'cooperation_rates_per_epoch': self.epoch_cooperation_rates,
            'mean_cooperation_rate': np.mean(self.epoch_cooperation_rates) if self.epoch_cooperation_rates else 0.0,
            'final_cooperation_rate': self.epoch_cooperation_rates[-1] if self.epoch_cooperation_rates else 0.0,
            'cooperation_trend': 'increasing' if len(self.epoch_cooperation_rates) > 1 and self.epoch_cooperation_rates[-1] > self.epoch_cooperation_rates[0] else 'decreasing' if len(self.epoch_cooperation_rates) > 1 else 'stable'
        }
        
        return {
            'total_epochs': self.current_epoch + 1,
            'best_loss': self.best_loss,
            'convergence_info': convergence_info,
            'loss_statistics': loss_stats,
            'reward_statistics': reward_stats,
            'cooperation_statistics': cooperation_stats,
            'final_weights': {
                'alpha': self.loss_fn.alpha,
                'gamma': self.loss_fn.gamma,
                'use_gae': self.loss_fn.use_gae
            }
        }
    
    def evaluate(
        self,
        game: MixedMotiveGame,
        opponents: List[Opponent],
        num_sessions: int = 10,
        enable_detailed_testing: bool = True,
        testing_log_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the trained network.
        
        Args:
            game: Game to evaluate on
            opponents: Opponents to evaluate against
            num_sessions: Number of evaluation sessions per opponent
            enable_detailed_testing: Whether to enable detailed testing logs
            testing_log_dir: Directory for testing logs (auto-generated if None and detailed testing enabled)
            
        Returns:
            Evaluation results
        """
        self.network.eval()
        evaluation_results = defaultdict(list)
        
        # Enable testing monitoring if requested
        if enable_detailed_testing and not self.testing_logging_enabled:
            if testing_log_dir is None:
                testing_log_dir = f"testing_logs_{self.network_serial_id}"
            self.enable_testing_monitoring(testing_log_dir)
        
        # Start a new test session if monitoring is enabled
        if self.testing_monitor is not None:
            self.testing_monitor.start_test_session(f"evaluation_{game.get_name()}")
        
        with torch.no_grad():
            for opponent in opponents:
                opponent_results = []
                
                for session_num in range(num_sessions):
                    # Play session for evaluation
                    session_results = self._evaluate_session_with_monitoring(
                        game, opponent, session_num, num_sessions
                    )
                    
                    if session_results:
                        opponent_results.append(session_results['session_stats'])
                
                # Average results for this opponent
                if opponent_results:
                    avg_results = {}
                    for key in opponent_results[0].keys():
                        if isinstance(opponent_results[0][key], (int, float)):
                            avg_results[key] = np.mean([r[key] for r in opponent_results])
                    
                    evaluation_results[opponent.get_strategy_name()] = avg_results
        
        # Finalize test session if monitoring is enabled
        if self.testing_monitor is not None:
            self.testing_monitor.finalize_session()
        
        return dict(evaluation_results)
    
    def _evaluate_session_with_monitoring(
        self,
        game: MixedMotiveGame,
        opponent: Opponent,
        session_num: int,
        total_sessions: int
    ) -> Dict[str, Any]:
        """
        Evaluate a single session with detailed monitoring.
        
        Args:
            game: Game to play
            opponent: Opponent to play against
            session_num: Current session number
            total_sessions: Total number of sessions
            
        Returns:
            Session results
        """
        # Reset game and opponent
        game.reset()
        opponent.reset()
        
        # Initialize hidden state
        hidden = self.network.init_hidden(1, self.device)
        
        session_stats = {
            'cumulative_reward': 0.0,
            'cooperation_count': 0,
            'total_games': self.config.num_games_per_partner
        }
        
        # Play each game in the session
        for game_step in range(self.config.num_games_per_partner):
            # Get current state
            state_vector = game.get_state_vector()
            state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Get network outputs
            policy_logits, opponent_policy_logits, value_estimate, new_hidden = self.network.forward(state_tensor, hidden)
            
            # Sample action from policy
            action_probs = torch.softmax(policy_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample()
            
            # Convert to Action enum
            from .games import Action
            agent_action = Action.COOPERATE if action_idx.item() == 0 else Action.DEFECT
            
            # Opponent chooses action
            opponent_action = opponent.play_action(game.history, game_step)
            
            # Play the round
            agent_reward, opponent_reward = game.play_round(agent_action, opponent_action)
            opponent.update_payoff(opponent_reward)
            
            # Update session stats
            session_stats['cumulative_reward'] += agent_reward
            if agent_action == Action.COOPERATE:
                session_stats['cooperation_count'] += 1
            
            # Log to testing monitor if enabled
            if self.testing_monitor is not None:
                # Create network outputs dictionary
                network_outputs = {
                    'policy_logits': policy_logits.squeeze(0),
                    'opponent_policy_logits': opponent_policy_logits.squeeze(0),
                    'value_estimate': value_estimate.squeeze(0)
                }
                
                # Get true opponent policy
                opponent_type = opponent.get_type_parameter()
                true_opponent_policy = None
                if opponent_type is not None:
                    # [defect_prob, cooperate_prob]
                    true_opponent_policy = torch.tensor([opponent_type, 1.0 - opponent_type])
                
                # Log the test step
                self.testing_monitor.log_test_step(
                    network_outputs=network_outputs,
                    agent_action=action_idx.item(),
                    opponent_action=opponent_action.value,
                    agent_reward=agent_reward,
                    opponent_reward=opponent_reward,
                    game_step=game_step,
                    total_games_in_session=self.config.num_games_per_partner,
                    game_name=game.get_name(),
                    opponent_name=opponent.get_strategy_name(),
                    opponent_type=opponent_type or 0.5,
                    true_opponent_policy=true_opponent_policy
                )
            
            # Update hidden state for next step
            hidden = new_hidden
        
        # Compile session results
        session_results = {
            'session_stats': {
                'cumulative_reward': session_stats['cumulative_reward'],
                'average_reward': session_stats['cumulative_reward'] / session_stats['total_games'],
                'cooperation_rate': session_stats['cooperation_count'] / session_stats['total_games'],
                'opponent_cooperation_rate': sum(1 for round_data in game.history if round_data['opponent_action'] == Action.COOPERATE) / len(game.history) if game.history else 0.0,
                'num_games': session_stats['total_games'],
                'opponent_type': opponent.get_type_parameter(),
                'opponent_name': opponent.get_strategy_name()
            }
        }
        
        return session_results

    def evaluate_on_multiple_games(
        self,
        games: Dict[str, MixedMotiveGame],
        opponents: List[Opponent],
        num_sessions: int = 10,
        enable_detailed_testing: bool = True,
        testing_log_dir: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate the trained network on multiple games.
        
        Args:
            games: Dictionary of game_name -> game_instance
            opponents: Opponents to evaluate against
            num_sessions: Number of evaluation sessions per opponent per game
            enable_detailed_testing: Whether to enable detailed testing logs
            testing_log_dir: Directory for testing logs (auto-generated if None and detailed testing enabled)
            
        Returns:
            Nested dictionary: game_name -> opponent_name -> evaluation_results
        """
        self.network.eval()
        multi_game_results = {}
        
        # Enable testing monitoring if requested
        if enable_detailed_testing and not self.testing_logging_enabled:
            if testing_log_dir is None:
                testing_log_dir = f"multi_game_testing_logs_{self.network_serial_id}"
            self.enable_testing_monitoring(testing_log_dir)
        
        # Start a new test session if monitoring is enabled
        if self.testing_monitor is not None:
            self.testing_monitor.start_test_session("multi_game_evaluation")
        
        with torch.no_grad():
            for game_name, game in games.items():
                self.logger.info(f"Evaluating on {game_name}")
                game_results = defaultdict(list)
                
                for opponent in opponents:
                    opponent_results = []
                    
                    for session_num in range(num_sessions):
                        # Play session with monitoring
                        session_results = self._evaluate_session_with_monitoring(
                            game, opponent, session_num, num_sessions
                        )
                        
                        if session_results:
                            opponent_results.append(session_results['session_stats'])
                    
                    # Average results for this opponent on this game
                    if opponent_results:
                        avg_results = {}
                        for key in opponent_results[0].keys():
                            if isinstance(opponent_results[0][key], (int, float)):
                                avg_results[key] = np.mean([r[key] for r in opponent_results])
                        
                        game_results[opponent.get_strategy_name()] = avg_results
                
                multi_game_results[game_name] = dict(game_results)
        
        # Finalize test session if monitoring is enabled
        if self.testing_monitor is not None:
            self.testing_monitor.finalize_session()
        
        return multi_game_results

    @staticmethod
    def create_game_config(
        name: str, 
        weight: float = 1.0, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a game configuration for multi-game training.
        
        Args:
            name: Game name (e.g., 'prisoners-dilemma', 'hawk-dove')
            weight: Sampling weight for this game (higher = more frequent)
            **kwargs: Additional game parameters
            
        Returns:
            Game configuration dictionary
        """
        return {
            'name': name,
            'weight': weight,
            'kwargs': kwargs
        }

    @staticmethod
    def create_balanced_game_configs(game_names: List[str]) -> List[Dict[str, Any]]:
        """
        Create balanced game configurations with equal weights.
        
        Args:
            game_names: List of game names
            
        Returns:
            List of game configurations with equal weights
        """
        return [
            GameTrainer.create_game_config(name, weight=1.0) 
            for name in game_names
        ]

    def get_multi_game_training_summary(self, training_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of multi-game training results.
        
        Args:
            training_results: Results from train_on_multiple_games()
            
        Returns:
            Formatted summary string
        """
        summary_lines = []
        summary_lines.append("=== MULTI-GAME TRAINING SUMMARY ===")
        summary_lines.append(f"Games Trained: {', '.join(training_results['games'])}")
        summary_lines.append(f"Game Weights: {training_results['game_weights']}")
        summary_lines.append(f"Total Epochs: {training_results['final_metrics']['total_epochs']}")
        summary_lines.append(f"Best Loss: {training_results['final_metrics']['best_loss']:.6f}")
        
        # Reward statistics
        if 'reward_statistics' in training_results['final_metrics']:
            reward_stats = training_results['final_metrics']['reward_statistics']
            summary_lines.append("\n=== REWARD PERFORMANCE ===")
            summary_lines.append(f"Total Cumulative Reward: {reward_stats['total_cumulative_reward']:.2f}")
            summary_lines.append(f"Mean Reward per Epoch: {reward_stats['mean_reward_across_epochs']:.2f}")
            summary_lines.append(f"Final Epoch Reward: {reward_stats['final_epoch_reward']:.2f}")
            
            # Show trend over last few epochs if available
            if len(reward_stats['cumulative_rewards_per_epoch']) >= 5:
                recent_rewards = reward_stats['cumulative_rewards_per_epoch'][-5:]
                summary_lines.append(f"Last 5 Epochs Rewards: {[f'{r:.2f}' for r in recent_rewards]}")
        
        # Cooperation statistics
        if 'cooperation_statistics' in training_results['final_metrics']:
            coop_stats = training_results['final_metrics']['cooperation_statistics']
            summary_lines.append("\n=== COOPERATION BEHAVIOR ===")
            summary_lines.append(f"Mean Cooperation Rate: {coop_stats['mean_cooperation_rate']:.3f}")
            summary_lines.append(f"Final Cooperation Rate: {coop_stats['final_cooperation_rate']:.3f}")
            summary_lines.append(f"Cooperation Trend: {coop_stats['cooperation_trend']}")
            
            # Show trend over last few epochs if available
            if len(coop_stats['cooperation_rates_per_epoch']) >= 5:
                recent_coop = coop_stats['cooperation_rates_per_epoch'][-5:]
                summary_lines.append(f"Last 5 Epochs Cooperation: {[f'{c:.3f}' for c in recent_coop]}")
        
        # Per-game performance if available
        if training_results['epoch_results']:
            last_epoch = training_results['epoch_results'][-1]
            if 'per_game_losses' in last_epoch:
                summary_lines.append("\n=== FINAL PER-GAME LOSSES ===")
                for game_name, losses in last_epoch['per_game_losses'].items():
                    summary_lines.append(f"  {game_name}: {losses.get('total_loss', 'N/A'):.6f}")
        
        return "\n".join(summary_lines)

    def get_single_game_training_summary(self, training_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of single-game training results.
        
        Args:
            training_results: Results from train_on_game()
            
        Returns:
            Formatted summary string
        """
        summary_lines = []
        summary_lines.append("=== SINGLE-GAME TRAINING SUMMARY ===")
        summary_lines.append(f"Game: {training_results['game_name']}")
        summary_lines.append(f"Opponents: {', '.join(training_results['opponents'])}")
        summary_lines.append(f"Total Epochs: {training_results['final_metrics']['total_epochs']}")
        summary_lines.append(f"Best Loss: {training_results['final_metrics']['best_loss']:.6f}")
        
        # Reward statistics
        if 'reward_statistics' in training_results['final_metrics']:
            reward_stats = training_results['final_metrics']['reward_statistics']
            summary_lines.append("\n=== REWARD PERFORMANCE ===")
            summary_lines.append(f"Total Cumulative Reward: {reward_stats['total_cumulative_reward']:.2f}")
            summary_lines.append(f"Mean Reward per Epoch: {reward_stats['mean_reward_across_epochs']:.2f}")
            summary_lines.append(f"Final Epoch Reward: {reward_stats['final_epoch_reward']:.2f}")
            
            # Show trend over last few epochs if available
            if len(reward_stats['cumulative_rewards_per_epoch']) >= 5:
                recent_rewards = reward_stats['cumulative_rewards_per_epoch'][-5:]
                summary_lines.append(f"Last 5 Epochs Rewards: {[f'{r:.2f}' for r in recent_rewards]}")
        
        # Cooperation statistics
        if 'cooperation_statistics' in training_results['final_metrics']:
            coop_stats = training_results['final_metrics']['cooperation_statistics']
            summary_lines.append("\n=== COOPERATION BEHAVIOR ===")
            summary_lines.append(f"Mean Cooperation Rate: {coop_stats['mean_cooperation_rate']:.3f}")
            summary_lines.append(f"Final Cooperation Rate: {coop_stats['final_cooperation_rate']:.3f}")
            summary_lines.append(f"Cooperation Trend: {coop_stats['cooperation_trend']}")
            
            # Show trend over last few epochs if available
            if len(coop_stats['cooperation_rates_per_epoch']) >= 5:
                recent_coop = coop_stats['cooperation_rates_per_epoch'][-5:]
                summary_lines.append(f"Last 5 Epochs Cooperation: {[f'{c:.3f}' for c in recent_coop]}")
        
        return "\n".join(summary_lines)

    def _log_interim_summary(self, training_results: Dict[str, Any], current_epoch: int):
        """
        Log interim summary report every 100 epochs during multi-game training.
        
        Args:
            training_results: Current training results dictionary
            current_epoch: Current epoch number
        """
        # Get last 10 epochs of data for averaging (or fewer if not available)
        num_epochs_to_average = min(10, len(training_results['epoch_results']))
        if num_epochs_to_average == 0:
            return
        
        recent_epochs = training_results['epoch_results'][-num_epochs_to_average:]
        
        # Calculate average agent policy probabilities over last 10 epochs
        agent_coop_probs = []
        agent_defect_probs = []
        
        # Calculate average predicted opponent policy probabilities over last 10 epochs  
        pred_opp_coop_probs = []
        pred_opp_defect_probs = []
        
        # Calculate average rewards over last 10 epochs
        avg_rewards = []
        
        for epoch_result in recent_epochs:
            # Extract agent policy probabilities
            if 'agent_policy_probs' in epoch_result:
                agent_coop_probs.append(epoch_result['agent_policy_probs']['cooperate'])
                agent_defect_probs.append(epoch_result['agent_policy_probs']['defect'])
            
            # Extract predicted opponent policy probabilities
            if 'predicted_opponent_policy_probs' in epoch_result:
                pred_opp_coop_probs.append(epoch_result['predicted_opponent_policy_probs']['cooperate'])
                pred_opp_defect_probs.append(epoch_result['predicted_opponent_policy_probs']['defect'])
            
            # Extract average reward
            if 'epoch_cumulative_reward' in epoch_result:
                avg_rewards.append(epoch_result['epoch_cumulative_reward'])
        
        # Calculate averages
        avg_agent_coop = np.mean(agent_coop_probs) if agent_coop_probs else 0.0
        avg_agent_defect = np.mean(agent_defect_probs) if agent_defect_probs else 0.0
        
        avg_pred_opp_coop = np.mean(pred_opp_coop_probs) if pred_opp_coop_probs else 0.0
        avg_pred_opp_defect = np.mean(pred_opp_defect_probs) if pred_opp_defect_probs else 0.0
        
        avg_reward = np.mean(avg_rewards) if avg_rewards else 0.0
        
        # Enhanced diagnostics - extract loss component information
        loss_ratios = []
        alpha_contributions = []
        for epoch_result in recent_epochs:
            if 'mixed_batch_loss' in epoch_result:
                loss_info = epoch_result['mixed_batch_loss']
                if 'loss_ratio' in loss_info:
                    loss_ratios.append(loss_info['loss_ratio'])
                if 'alpha_contribution' in loss_info:
                    alpha_contributions.append(loss_info['alpha_contribution'])
        
        avg_loss_ratio = np.mean(loss_ratios) if loss_ratios else 0.0
        avg_alpha_contribution = np.mean(alpha_contributions) if alpha_contributions else 0.0
        
        # Log comprehensive interim summary
        self.logger.info(f"=== INTERIM SUMMARY - EPOCH {current_epoch} ===")
        self.logger.info(f"Average over last {num_epochs_to_average} epochs:")
        self.logger.info(f"Agent Policy - Cooperate: {avg_agent_coop:.4f}, Defect: {avg_agent_defect:.4f}")
        self.logger.info(f"Predicted Opponent Policy - Cooperate: {avg_pred_opp_coop:.4f}, Defect: {avg_pred_opp_defect:.4f}")
        self.logger.info(f"Average Reward: {avg_reward:.4f}")
        self.logger.info(f"Loss Analysis - OpPolicy/RL Ratio: {avg_loss_ratio:.4f}, Alpha Contribution: {avg_alpha_contribution:.4f}")
        
        # Policy prediction accuracy check
        if avg_pred_opp_coop > 0 and avg_pred_opp_defect > 0:
            policy_entropy = -(avg_pred_opp_coop * np.log(avg_pred_opp_coop + 1e-8) + 
                             avg_pred_opp_defect * np.log(avg_pred_opp_defect + 1e-8))
            self.logger.info(f"Opponent Policy Prediction Entropy: {policy_entropy:.4f} (lower = more confident)")
        
        self.logger.info("=" * 60)