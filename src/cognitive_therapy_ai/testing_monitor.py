"""
Testing Monitor for detailed evaluation phase tracking.

This module provides comprehensive logging of testing/evaluation phases including:
- Network predicted values from opponent policy prediction
- Policy head outputs and sampled actions
- Actions of both agents and resulting rewards
- Network serial numbers for linking training and testing data
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import os
import uuid
from collections import defaultdict


class TestingMonitor:
    """
    Monitors and logs detailed testing/evaluation information.
    
    Creates comprehensive tables tracking every evaluation step with:
    - Network predictions and outputs
    - Actions and rewards for both agents
    - Network serial numbers for data linking
    """
    
    def __init__(self, output_dir: str, network_serial_id: Optional[str] = None, save_frequency: int = 50):
        """
        Initialize the testing monitor.
        
        Args:
            output_dir: Directory to save testing logs and tables
            network_serial_id: Unique identifier for the network (auto-generated if None)
            save_frequency: How often to save the table to disk (iterations)
        """
        self.output_dir = output_dir
        self.save_frequency = save_frequency
        self.network_serial_id = network_serial_id or self._generate_network_serial_id()
        self.logger = logging.getLogger(__name__)
        
        # Testing data storage
        self.testing_log = []
        self.test_iteration_count = 0
        self.current_test_session = 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize CSV file with headers
        self.csv_path = os.path.join(output_dir, 'detailed_testing_log.csv')
        
        # Define column headers for testing
        self.columns = [
            # Network identification
            'network_serial_id', 'test_session', 'test_iteration', 'timestamp',
            
            # Game context
            'game_name', 'opponent_name', 'opponent_type', 'game_step_in_session',
            
            # Network predictions and outputs
            'predicted_opponent_policy_defect', 'predicted_opponent_policy_cooperate',
            'predicted_opponent_value', 'predicted_opponent_cooperation_likelihood',
            
            # Policy head outputs (raw logits)
            'agent_policy_logit_cooperate', 'agent_policy_logit_defect',
            'agent_value_estimate',
            
            # Processed probabilities
            'agent_policy_prob_cooperate', 'agent_policy_prob_defect',
            
            # Sampled actions and outcomes
            'agent_sampled_action', 'opponent_actual_action',
            'agent_reward', 'opponent_reward', 'total_reward',
            
            # True opponent information (for comparison)
            'true_opponent_defect_prob', 'true_opponent_cooperate_prob',
            
            # Performance metrics
            'prediction_accuracy', 'action_prediction_error',
            'value_prediction_error', 'cumulative_agent_reward',
            
            # Additional context
            'session_game_number', 'total_games_in_session'
        ]
        
        # Initialize CSV with headers if file doesn't exist
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=self.columns).to_csv(self.csv_path, index=False)
        
        # Track session-level statistics
        self.session_stats = defaultdict(list)
        
        self.logger.info(f"Testing monitor initialized for network {self.network_serial_id}")
        self.logger.info(f"Test logs will be saved to {output_dir}")
    
    def _generate_network_serial_id(self) -> str:
        """Generate a unique serial ID for the network."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"NET_{timestamp}_{unique_id}"
    
    def start_test_session(self, session_name: str = None) -> int:
        """
        Start a new testing session.
        
        Args:
            session_name: Optional name for the session
            
        Returns:
            Session ID number
        """
        self.current_test_session += 1
        session_name = session_name or f"test_session_{self.current_test_session}"
        
        self.logger.info(f"Started test session {self.current_test_session}: {session_name}")
        return self.current_test_session
    
    def log_test_step(
        self,
        network_outputs: Dict[str, torch.Tensor],
        agent_action: int,
        opponent_action: int,
        agent_reward: float,
        opponent_reward: float,
        game_step: int,
        total_games_in_session: int,
        game_name: str = "unknown",
        opponent_name: str = "unknown",
        opponent_type: float = 0.5,
        true_opponent_policy: Optional[torch.Tensor] = None
    ):
        """
        Log a single test step with all relevant information.
        
        Args:
            network_outputs: Dictionary containing network predictions
            agent_action: Action taken by the agent (0=cooperate, 1=defect)
            opponent_action: Action taken by the opponent (0=cooperate, 1=defect)
            agent_reward: Reward received by the agent
            opponent_reward: Reward received by the opponent
            game_step: Current game step within the session
            total_games_in_session: Total number of games in this session
            game_name: Name of the current game
            opponent_name: Name/type of current opponent
            opponent_type: Opponent's true defection probability
            true_opponent_policy: True opponent policy tensor [defect_prob, cooperate_prob]
        """
        self.test_iteration_count += 1
        
        # Extract network outputs safely
        policy_logits = network_outputs.get('policy_logits', torch.zeros(2))
        opponent_policy_logits = network_outputs.get('opponent_policy_logits', torch.zeros(2))
        value_estimate = network_outputs.get('value_estimate', torch.zeros(1))
        
        # Convert to probabilities
        policy_probs = torch.softmax(policy_logits, dim=-1) if policy_logits.numel() > 1 else torch.tensor([0.5, 0.5])
        opponent_policy_probs = torch.softmax(opponent_policy_logits, dim=-1) if opponent_policy_logits.numel() > 1 else torch.tensor([0.5, 0.5])
        
        # Calculate predicted opponent cooperation likelihood
        predicted_cooperation_likelihood = opponent_policy_probs[1].item()  # cooperate probability
        
        # Calculate prediction accuracy
        actual_cooperation = 1 if opponent_action == 0 else 0  # 0=cooperate, 1=defect
        prediction_accuracy = abs(predicted_cooperation_likelihood - actual_cooperation)
        
        # Calculate action prediction error (how well we predicted the specific action)
        predicted_action_prob = opponent_policy_probs[opponent_action].item()
        action_prediction_error = 1.0 - predicted_action_prob
        
        # Value prediction error (we'll need to calculate this based on actual outcome)
        # For now, use the difference between predicted value and actual reward
        value_prediction_error = abs(value_estimate.item() - agent_reward)
        
        # True opponent policy information
        if true_opponent_policy is not None:
            true_defect_prob = true_opponent_policy[0].item()
            true_cooperate_prob = true_opponent_policy[1].item()
        else:
            true_defect_prob = opponent_type
            true_cooperate_prob = 1.0 - opponent_type
        
        # Calculate cumulative reward for this session
        session_rewards = [entry['agent_reward'] for entry in self.testing_log 
                          if entry['test_session'] == self.current_test_session]
        cumulative_reward = sum(session_rewards) + agent_reward
        
        # Create log entry
        log_entry = {
            # Network identification
            'network_serial_id': self.network_serial_id,
            'test_session': self.current_test_session,
            'test_iteration': self.test_iteration_count,
            'timestamp': datetime.now().isoformat(),
            
            # Game context
            'game_name': game_name,
            'opponent_name': opponent_name,
            'opponent_type': opponent_type,
            'game_step_in_session': game_step,
            
            # Network predictions and outputs
            'predicted_opponent_policy_defect': self._safe_extract(opponent_policy_probs[0]),
            'predicted_opponent_policy_cooperate': self._safe_extract(opponent_policy_probs[1]),
            'predicted_opponent_value': self._safe_extract(value_estimate),  # Value from opponent's perspective
            'predicted_opponent_cooperation_likelihood': predicted_cooperation_likelihood,
            
            # Policy head outputs (raw logits)
            'agent_policy_logit_cooperate': self._safe_extract(policy_logits[0] if len(policy_logits) > 0 else 0.0),
            'agent_policy_logit_defect': self._safe_extract(policy_logits[1] if len(policy_logits) > 1 else 0.0),
            'agent_value_estimate': self._safe_extract(value_estimate),
            
            # Processed probabilities
            'agent_policy_prob_cooperate': self._safe_extract(policy_probs[0]),
            'agent_policy_prob_defect': self._safe_extract(policy_probs[1]),
            
            # Sampled actions and outcomes
            'agent_sampled_action': agent_action,
            'opponent_actual_action': opponent_action,
            'agent_reward': agent_reward,
            'opponent_reward': opponent_reward,
            'total_reward': agent_reward + opponent_reward,
            
            # True opponent information
            'true_opponent_defect_prob': true_defect_prob,
            'true_opponent_cooperate_prob': true_cooperate_prob,
            
            # Performance metrics
            'prediction_accuracy': prediction_accuracy,
            'action_prediction_error': action_prediction_error,
            'value_prediction_error': value_prediction_error,
            'cumulative_agent_reward': cumulative_reward,
            
            # Additional context
            'session_game_number': game_step,
            'total_games_in_session': total_games_in_session
        }
        
        # Add to log
        self.testing_log.append(log_entry)
        
        # Update session statistics
        self.session_stats[self.current_test_session].append({
            'agent_reward': agent_reward,
            'prediction_accuracy': prediction_accuracy,
            'agent_action': agent_action,
            'opponent_action': opponent_action
        })
        
        # Periodic save to disk
        if self.test_iteration_count % self.save_frequency == 0:
            self.save_to_disk()
            self.print_recent_test_summary()
    
    def _safe_extract(self, value) -> float:
        """Safely extract a float value from various tensor/scalar types."""
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.item()
            else:
                return value.mean().item()
        elif isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return float(value.mean())
        else:
            return 0.0
    
    def _json_serialize_default(self, obj):
        """Custom JSON serialization for NumPy and PyTorch types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif torch.is_tensor(obj):
            return obj.detach().cpu().numpy().tolist()
        elif hasattr(obj, 'item'):  # Handle single-element tensors
            try:
                return obj.item()
            except (ValueError, TypeError):
                pass
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    def save_to_disk(self):
        """Save the current testing log to disk."""
        if not self.testing_log:
            return
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.testing_log)
            
            # Save as CSV (append if file exists)
            if os.path.exists(self.csv_path):
                df.tail(self.save_frequency).to_csv(self.csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(self.csv_path, index=False)
            
            self.logger.debug(f"Saved {len(df)} testing log entries to disk")
            
        except Exception as e:
            self.logger.error(f"Failed to save testing log: {str(e)}")
    
    def print_recent_test_summary(self, num_recent: int = 10):
        """Print a summary of recent test steps."""
        if len(self.testing_log) < num_recent:
            recent_entries = self.testing_log
        else:
            recent_entries = self.testing_log[-num_recent:]
        
        if not recent_entries:
            return
        
        print(f"\n--- RECENT TEST SUMMARY (Last {len(recent_entries)} steps) ---")
        print(f"Network: {self.network_serial_id}, Session: {self.current_test_session}")
        
        # Calculate averages
        avg_agent_reward = np.mean([entry['agent_reward'] for entry in recent_entries])
        avg_prediction_accuracy = np.mean([entry['prediction_accuracy'] for entry in recent_entries])
        avg_cooperation_likelihood = np.mean([entry['predicted_opponent_cooperation_likelihood'] for entry in recent_entries])
        
        # Count actions
        agent_cooperations = sum(1 for entry in recent_entries if entry['agent_sampled_action'] == 0)
        agent_cooperation_rate = agent_cooperations / len(recent_entries)
        
        print(f"Performance - Reward: {avg_agent_reward:.4f}, Prediction Acc: {avg_prediction_accuracy:.4f}")
        print(f"Agent Cooperation: {agent_cooperation_rate:.2%}, Predicted Opp Coop: {avg_cooperation_likelihood:.2%}")
        print(f"Files: {self.csv_path}")
        print("-" * 65)
    
    def finalize_session(self):
        """Finalize the current test session and save summary statistics."""
        if not self.session_stats[self.current_test_session]:
            return
        
        session_data = self.session_stats[self.current_test_session]
        
        # Calculate session-level statistics
        total_reward = sum(entry['agent_reward'] for entry in session_data)
        avg_reward = total_reward / len(session_data)
        avg_prediction_accuracy = np.mean([entry['prediction_accuracy'] for entry in session_data])
        
        cooperation_count = sum(1 for entry in session_data if entry['agent_action'] == 0)
        cooperation_rate = cooperation_count / len(session_data)
        
        # Save session summary
        session_summary = {
            'network_serial_id': self.network_serial_id,
            'test_session': self.current_test_session,
            'total_steps': len(session_data),
            'total_reward': total_reward,
            'average_reward': avg_reward,
            'average_prediction_accuracy': avg_prediction_accuracy,
            'cooperation_rate': cooperation_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save session summary to separate file
        summary_file = os.path.join(self.output_dir, f'session_{self.current_test_session}_summary.json')
        import json
        with open(summary_file, 'w') as f:
            json.dump(session_summary, f, indent=2, default=self._json_serialize_default)
        
        self.logger.info(f"Finalized test session {self.current_test_session}: "
                        f"{len(session_data)} steps, avg_reward={avg_reward:.4f}")
    
    def finalize(self):
        """Final save and cleanup when testing ends."""
        # Finalize current session if active
        if self.session_stats[self.current_test_session]:
            self.finalize_session()
        
        # Save all data
        self.save_to_disk()
        
        if self.testing_log:
            # Create final summary
            summary_file = os.path.join(self.output_dir, 'testing_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("TESTING MONITOR SUMMARY\n")
                f.write("="*50 + "\n\n")
                f.write(f"Network Serial ID: {self.network_serial_id}\n")
                f.write(f"Total test iterations logged: {len(self.testing_log)}\n")
                f.write(f"Total test sessions: {self.current_test_session}\n")
                f.write(f"Testing started: {self.testing_log[0]['timestamp']}\n")
                f.write(f"Testing ended: {self.testing_log[-1]['timestamp']}\n\n")
                
                # Calculate final statistics
                final_avg_reward = np.mean([entry['agent_reward'] for entry in self.testing_log[-100:]])
                final_avg_accuracy = np.mean([entry['prediction_accuracy'] for entry in self.testing_log[-100:]])
                
                f.write(f"Final average reward (last 100 steps): {final_avg_reward:.4f}\n")
                f.write(f"Final prediction accuracy (last 100 steps): {final_avg_accuracy:.4f}\n")
            
            self.logger.info(f"Testing monitor finalized. {len(self.testing_log)} test iterations logged.")
            self.logger.info(f"Detailed test logs saved to: {self.csv_path}")
            print(f"\n🎯 TESTING MONITOR FINALIZED")
            print(f"📊 Network ID: {self.network_serial_id}")
            print(f"📈 Total test steps logged: {len(self.testing_log)}")
            print(f"📁 Detailed test logs: {self.csv_path}")
    
    def get_network_serial_id(self) -> str:
        """Get the network's serial ID for linking with training data."""
        return self.network_serial_id
    
    def link_to_training_data(self, training_log_path: str) -> str:
        """
        Create a linkage file between training and testing data.
        
        Args:
            training_log_path: Path to the training log file
            
        Returns:
            Path to the linkage file created
        """
        linkage_file = os.path.join(self.output_dir, 'training_test_linkage.json')
        
        linkage_data = {
            'network_serial_id': self.network_serial_id,
            'training_data_path': training_log_path,
            'testing_data_path': self.csv_path,
            'linkage_created': datetime.now().isoformat(),
            'description': 'Links training and testing data for the same network instance'
        }
        
        import json
        with open(linkage_file, 'w') as f:
            json.dump(linkage_data, f, indent=2, default=self._json_serialize_default)
        
        self.logger.info(f"Created training-test linkage file: {linkage_file}")
        return linkage_file