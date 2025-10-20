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
            'opponent_type_pred': torch.full_like(value_estimate, self.opponent.get_type_parameter() if self.opponent.get_type_parameter() is not None else 0.0),
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
        opponent_type_preds = torch.cat([data['opponent_type_pred'] for data in self.session_data], dim=0)
        
        # True opponent type (same for all games in session)
        opponent_type_true = self.session_data[0]['opponent_type_true']
        if opponent_type_true is not None:
            opponent_type_tensor = torch.full((len(self.session_data), 1), opponent_type_true, dtype=torch.float32)
            # Create true opponent policy tensor: [p_defect, p_cooperate] = [1-p_d, p_d]
            p_d = opponent_type_true  # defection probability from opponent type
            true_opponent_policy = torch.tensor([[1-p_d, p_d]], dtype=torch.float32).expand(len(self.session_data), -1)
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
                'opponent_type_pred': opponent_type_preds,
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
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        self.logger = logging.getLogger(__name__)
    
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
        
        return {
            'total_epochs': self.current_epoch + 1,
            'best_loss': self.best_loss,
            'convergence_info': convergence_info,
            'loss_statistics': loss_stats,
            'games_info': game_info,
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
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
        
        epoch_results = {
            'total_loss': loss_dict['total_loss'].item(),
            'mixed_batch_loss': mixed_batch_loss_record,
            'per_game_losses': per_game_losses,
            'game_session_stats': game_session_stats,
            'num_sessions_per_game': {game: len(stats) for game, stats in game_session_stats.items()},
            'total_sessions': len(all_training_data)
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
                      'opponent_type_pred', 'actions', 'rewards', 'opponent_actions', 'opponent_type_true', 'true_opponent_policy']
        
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
                           'opponent_type_pred', 'opponent_type_true', 'true_opponent_policy']:
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
        if session_stats:
            for key in session_stats[0].keys():
                if isinstance(session_stats[0][key], (int, float)):
                    avg_stats[key] = np.mean([stats[key] for stats in session_stats])
        
        return {
            **avg_losses,
            'session_stats': avg_stats,
            'num_sessions': len(epoch_losses)
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
        # Early stopping based on loss improvement
        current_loss = self.loss_analyzer.loss_history['total']
        if current_loss:
            latest_loss = current_loss[-1]
            if latest_loss < self.best_loss:
                self.best_loss = latest_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        # Check patience
        if self.patience_counter >= self.config.patience:
            return True
        
        # Check loss convergence using ToM-RL LossAnalyzer
        if hasattr(self.loss_analyzer, 'check_convergence'):
            if self.loss_analyzer.check_convergence('total', threshold=self.config.convergence_threshold):
                return True
        else:
            # Fallback convergence check if method doesn't exist
            if len(current_loss) >= 10:
                recent_losses = current_loss[-10:]
                loss_variance = torch.tensor(recent_losses).var().item()
                if loss_variance < self.config.convergence_threshold:
                    return True
        
        return False
    
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
        
        return {
            'total_epochs': self.current_epoch + 1,
            'best_loss': self.best_loss,
            'convergence_info': convergence_info,
            'loss_statistics': loss_stats,
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
        num_sessions: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate the trained network.
        
        Args:
            game: Game to evaluate on
            opponents: Opponents to evaluate against
            num_sessions: Number of evaluation sessions per opponent
            
        Returns:
            Evaluation results
        """
        self.network.eval()
        evaluation_results = defaultdict(list)
        
        with torch.no_grad():
            for opponent in opponents:
                opponent_results = []
                
                for session_num in range(num_sessions):
                    session = GameSession(game, opponent, self.network, self.config.num_games_per_partner, training_mode=False)
                    session_results = session.play_session(self.device)
                    
                    if session_results:
                        opponent_results.append(session_results['session_stats'])
                
                # Average results for this opponent
                if opponent_results:
                    avg_results = {}
                    for key in opponent_results[0].keys():
                        if isinstance(opponent_results[0][key], (int, float)):
                            avg_results[key] = np.mean([r[key] for r in opponent_results])
                    
                    evaluation_results[opponent.get_strategy_name()] = avg_results
        
        return dict(evaluation_results)

    def evaluate_on_multiple_games(
        self,
        games: Dict[str, MixedMotiveGame],
        opponents: List[Opponent],
        num_sessions: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate the trained network on multiple games.
        
        Args:
            games: Dictionary of game_name -> game_instance
            opponents: Opponents to evaluate against
            num_sessions: Number of evaluation sessions per opponent per game
            
        Returns:
            Nested dictionary: game_name -> opponent_name -> evaluation_results
        """
        self.network.eval()
        multi_game_results = {}
        
        with torch.no_grad():
            for game_name, game in games.items():
                self.logger.info(f"Evaluating on {game_name}")
                game_results = defaultdict(list)
                
                for opponent in opponents:
                    opponent_results = []
                    
                    for session_num in range(num_sessions):
                        session = GameSession(game, opponent, self.network, self.config.num_games_per_partner, training_mode=False)
                        session_results = session.play_session(self.device)
                        
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
        
        # Per-game performance if available
        if training_results['epoch_results']:
            last_epoch = training_results['epoch_results'][-1]
            if 'per_game_losses' in last_epoch:
                summary_lines.append("\nFinal Per-Game Losses:")
                for game_name, losses in last_epoch['per_game_losses'].items():
                    summary_lines.append(f"  {game_name}: {losses.get('total_loss', 'N/A'):.6f}")
        
        return "\n".join(summary_lines)