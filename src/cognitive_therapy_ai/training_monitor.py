"""
Training Monitor for detailed step-by-step tracking.

This module provides comprehensive logging of training iterations including:
- Loss components (RL, opponent prediction, total)
- Network outputs (policy logits, opponent policy logits, value estimates)
- Sampled actions and opponent actions
- Rewards for both parties
- Real-time table updates during training
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import os
from collections import defaultdict


class TrainingMonitor:
    """
    Monitors and logs detailed training information in real-time.
    
    Creates comprehensive tables tracking every training iteration with:
    - Loss components
    - Network head outputs
    - Actions and rewards
    - Opponent information
    """
    
    def __init__(self, output_dir: str, save_frequency: int = 100):
        """
        Initialize the training monitor.
        
        Args:
            output_dir: Directory to save training logs and tables
            save_frequency: How often to save the table to disk (iterations)
        """
        self.output_dir = output_dir
        self.save_frequency = save_frequency
        self.logger = logging.getLogger(__name__)
        
        # Training data storage
        self.training_log = []
        self.iteration_count = 0
        self.epoch_count = 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize CSV file with headers
        self.csv_path = os.path.join(output_dir, 'detailed_training_log.csv')
        self.excel_path = os.path.join(output_dir, 'detailed_training_log.xlsx')
        
        # Define column headers
        self.columns = [
            # Meta information
            'iteration', 'epoch', 'game_step', 'timestamp',
            
            # Game context
            'game_name', 'opponent_name', 'opponent_type',
            
            # Loss components
            'total_loss', 'rl_loss', 'rl_loss_normalized', 
            'opponent_policy_loss', 'opponent_policy_loss_normalized',
            'loss_ratio', 'alpha_contribution', 'alpha',
            
            # Network outputs (raw logits)
            'policy_logit_cooperate', 'policy_logit_defect',
            'opponent_policy_logit_defect', 'opponent_policy_logit_cooperate', 
            'value_estimate',
            
            # Processed probabilities
            'policy_prob_cooperate', 'policy_prob_defect',
            'opponent_policy_prob_defect', 'opponent_policy_prob_cooperate',
            
            # Actions and outcomes
            'agent_action', 'opponent_action',
            'agent_reward', 'opponent_reward',
            
            # True opponent information
            'true_opponent_defect_prob', 'true_opponent_cooperate_prob',
            
            # Additional metrics
            'advantage', 'temperature', 'gradient_norm'
        ]
        
        # Initialize CSV with headers if file doesn't exist
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=self.columns).to_csv(self.csv_path, index=False)
        
        self.logger.info(f"Training monitor initialized. Logs will be saved to {output_dir}")
    
    def log_training_step(
        self,
        loss_dict: Dict[str, Any],
        training_data: Dict[str, torch.Tensor],
        game_step_data: Dict[str, Any],
        epoch: int,
        game_step: int,
        game_name: str = "unknown",
        opponent_name: str = "unknown"
    ):
        """
        Log a single training step with all relevant information.
        
        Args:
            loss_dict: Dictionary containing loss components
            training_data: Training data tensors
            game_step_data: Individual game step data
            epoch: Current training epoch
            game_step: Step within the current session
            game_name: Name of the current game
            opponent_name: Name/type of current opponent
        """
        self.iteration_count += 1
        self.epoch_count = epoch
        
        # Extract data for this specific step
        step_idx = min(game_step, training_data['actions'].shape[0] - 1)
        
        # Extract network outputs for this step
        policy_logits = training_data['policy_logits'][step_idx] if training_data['policy_logits'].dim() > 1 else training_data['policy_logits']
        opponent_policy_logits = training_data['opponent_policy_logits'][step_idx] if training_data['opponent_policy_logits'].dim() > 1 else training_data['opponent_policy_logits']
        value_estimate = training_data['value_estimate'][step_idx] if training_data['value_estimate'].dim() > 1 else training_data['value_estimate']
        
        # Convert logits to probabilities
        policy_probs = torch.softmax(policy_logits, dim=-1) if policy_logits.dim() > 0 else torch.softmax(policy_logits.unsqueeze(0), dim=-1)
        opponent_policy_probs = torch.softmax(opponent_policy_logits, dim=-1) if opponent_policy_logits.dim() > 0 else torch.softmax(opponent_policy_logits.unsqueeze(0), dim=-1)
        
        # Extract actions and rewards
        agent_action = training_data['actions'][step_idx].item() if step_idx < len(training_data['actions']) else -1
        opponent_action = training_data['opponent_actions'][step_idx].item() if step_idx < len(training_data['opponent_actions']) else -1
        agent_reward = training_data['rewards'][step_idx].item() if step_idx < len(training_data['rewards']) else 0.0
        
        # Extract opponent type information
        true_opponent_policy = training_data.get('true_opponent_policy', None)
        if true_opponent_policy is not None and step_idx < len(true_opponent_policy):
            true_defect_prob = true_opponent_policy[step_idx][0].item()  # [defect, cooperate]
            true_cooperate_prob = true_opponent_policy[step_idx][1].item()
        else:
            true_defect_prob = 0.5
            true_cooperate_prob = 0.5
        
        # Extract additional data from game_step_data if available
        opponent_reward = game_step_data.get('opponent_reward', 0.0)
        opponent_type = game_step_data.get('opponent_type_true', 0.5)
        
        # Calculate gradient norm
        gradient_norm = 0.0
        if hasattr(self, '_last_gradient_norm'):
            gradient_norm = self._last_gradient_norm
        
        # Create log entry
        log_entry = {
            # Meta information
            'iteration': self.iteration_count,
            'epoch': epoch,
            'game_step': game_step,
            'timestamp': datetime.now().isoformat(),
            
            # Game context
            'game_name': game_name,
            'opponent_name': opponent_name,
            'opponent_type': opponent_type,
            
            # Loss components (handle tensors safely)
            'total_loss': self._safe_extract(loss_dict.get('total_loss', 0.0)),
            'rl_loss': self._safe_extract(loss_dict.get('rl_loss', 0.0)),
            'rl_loss_normalized': self._safe_extract(loss_dict.get('rl_loss_normalized', 0.0)),
            'opponent_policy_loss': self._safe_extract(loss_dict.get('opponent_policy_loss', 0.0)),
            'opponent_policy_loss_normalized': self._safe_extract(loss_dict.get('opponent_policy_loss_normalized', 0.0)),
            'loss_ratio': self._safe_extract(loss_dict.get('loss_ratio', 1.0)),
            'alpha_contribution': self._safe_extract(loss_dict.get('alpha_contribution', 1.0)),
            'alpha': self._safe_extract(loss_dict.get('alpha', 1.0)),
            
            # Network outputs (raw logits)
            'policy_logit_cooperate': self._safe_extract(policy_logits[0] if len(policy_logits) > 0 else 0.0),
            'policy_logit_defect': self._safe_extract(policy_logits[1] if len(policy_logits) > 1 else 0.0),
            'opponent_policy_logit_defect': self._safe_extract(opponent_policy_logits[0] if len(opponent_policy_logits) > 0 else 0.0),
            'opponent_policy_logit_cooperate': self._safe_extract(opponent_policy_logits[1] if len(opponent_policy_logits) > 1 else 0.0),
            'value_estimate': self._safe_extract(value_estimate),
            
            # Processed probabilities
            'policy_prob_cooperate': self._safe_extract(policy_probs[0] if len(policy_probs) > 0 else 0.5),
            'policy_prob_defect': self._safe_extract(policy_probs[1] if len(policy_probs) > 1 else 0.5),
            'opponent_policy_prob_defect': self._safe_extract(opponent_policy_probs[0] if len(opponent_policy_probs) > 0 else 0.5),
            'opponent_policy_prob_cooperate': self._safe_extract(opponent_policy_probs[1] if len(opponent_policy_probs) > 1 else 0.5),
            
            # Actions and outcomes
            'agent_action': agent_action,  # 0=cooperate, 1=defect
            'opponent_action': opponent_action,  # 0=cooperate, 1=defect
            'agent_reward': agent_reward,
            'opponent_reward': opponent_reward,
            
            # True opponent information
            'true_opponent_defect_prob': true_defect_prob,
            'true_opponent_cooperate_prob': true_cooperate_prob,
            
            # Additional metrics
            'advantage': self._safe_extract(loss_dict.get('advantages_mean', 0.0)),
            'temperature': self._safe_extract(loss_dict.get('temperature', 1.0)),
            'gradient_norm': gradient_norm
        }
        
        # Add to log
        self.training_log.append(log_entry)
        
        # Periodic save to disk
        if self.iteration_count % self.save_frequency == 0:
            self.save_to_disk()
            self.print_recent_summary()
    
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
    
    def update_gradient_norm(self, gradient_norm: float):
        """Update the stored gradient norm for the next log entry."""
        self._last_gradient_norm = gradient_norm
    
    def save_to_disk(self):
        """Save the current training log to disk."""
        if not self.training_log:
            return
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.training_log)
            
            # Save as CSV (append if file exists)
            if os.path.exists(self.csv_path):
                df.tail(self.save_frequency).to_csv(self.csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(self.csv_path, index=False)
            
            # Save as Excel (full file each time for better formatting)
            df.to_excel(self.excel_path, index=False, sheet_name='Training_Log')
            
            self.logger.debug(f"Saved {len(df)} training log entries to disk")
            
        except Exception as e:
            self.logger.error(f"Failed to save training log: {str(e)}")
    
    def print_recent_summary(self, num_recent: int = 10):
        """Print a summary of recent training steps."""
        if len(self.training_log) < num_recent:
            recent_entries = self.training_log
        else:
            recent_entries = self.training_log[-num_recent:]
        
        if not recent_entries:
            return
        
        print(f"\n--- RECENT TRAINING SUMMARY (Last {len(recent_entries)} steps) ---")
        print(f"Iteration: {recent_entries[-1]['iteration']}, Epoch: {recent_entries[-1]['epoch']}")
        
        # Calculate averages
        avg_total_loss = np.mean([entry['total_loss'] for entry in recent_entries])
        avg_rl_loss = np.mean([entry['rl_loss_normalized'] for entry in recent_entries])
        avg_opp_loss = np.mean([entry['opponent_policy_loss_normalized'] for entry in recent_entries])
        avg_agent_reward = np.mean([entry['agent_reward'] for entry in recent_entries])
        
        # Count actions
        cooperate_count = sum(1 for entry in recent_entries if entry['agent_action'] == 0)
        cooperation_rate = cooperate_count / len(recent_entries)
        
        print(f"Losses - Total: {avg_total_loss:.6f}, RL: {avg_rl_loss:.6f}, Opp: {avg_opp_loss:.6f}")
        print(f"Agent - Reward: {avg_agent_reward:.4f}, Cooperation: {cooperation_rate:.2%}")
        print(f"Files: {self.csv_path}")
        print("-" * 60)
    
    def finalize(self):
        """Final save and cleanup when training ends."""
        self.save_to_disk()
        
        if self.training_log:
            # Create final summary
            summary_file = os.path.join(self.output_dir, 'training_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("TRAINING MONITOR SUMMARY\n")
                f.write("="*50 + "\n\n")
                f.write(f"Total iterations logged: {len(self.training_log)}\n")
                f.write(f"Final epoch: {self.epoch_count}\n")
                f.write(f"Training started: {self.training_log[0]['timestamp']}\n")
                f.write(f"Training ended: {self.training_log[-1]['timestamp']}\n\n")
                
                # Calculate final statistics
                final_loss = np.mean([entry['total_loss'] for entry in self.training_log[-100:]])
                final_cooperation = np.mean([1 for entry in self.training_log[-100:] if entry['agent_action'] == 0]) / min(100, len(self.training_log))
                
                f.write(f"Final average loss (last 100 steps): {final_loss:.6f}\n")
                f.write(f"Final cooperation rate (last 100 steps): {final_cooperation:.2%}\n")
            
            self.logger.info(f"Training monitor finalized. {len(self.training_log)} iterations logged.")
            self.logger.info(f"Detailed logs saved to: {self.csv_path}")
            print(f"\n🎯 TRAINING MONITOR FINALIZED")
            print(f"📊 Total steps logged: {len(self.training_log)}")
            print(f"📁 Detailed logs: {self.csv_path}")
            print(f"📈 Excel format: {self.excel_path}")


class BatchedTrainingMonitor(TrainingMonitor):
    """
    Extended training monitor that can handle batch processing efficiently.
    
    Processes multiple game steps at once for better performance during training.
    """
    
    def log_training_batch(
        self,
        loss_dict: Dict[str, Any],
        training_data: Dict[str, torch.Tensor],
        session_results: Dict[str, Any],
        epoch: int,
        game_name: str = "unknown",
        opponent_name: str = "unknown"
    ):
        """
        Log a complete batch of training steps from a session.
        
        Args:
            loss_dict: Dictionary containing loss components
            training_data: Training data tensors for the entire session
            session_results: Results from the complete session
            epoch: Current training epoch
            game_name: Name of the current game
            opponent_name: Name/type of current opponent
        """
        num_steps = training_data['actions'].shape[0]
        session_stats = session_results.get('session_stats', {})
        
        # Extract session-level information
        opponent_type = session_stats.get('opponent_type', 0.5)
        
        # Process each step in the batch
        for step_idx in range(num_steps):
            # Create a pseudo game_step_data for this step
            game_step_data = {
                'opponent_reward': 0.0,  # We don't have individual step rewards
                'opponent_type_true': opponent_type
            }
            
            # Log this step
            self.log_training_step(
                loss_dict=loss_dict,
                training_data=training_data,
                game_step_data=game_step_data,
                epoch=epoch,
                game_step=step_idx,
                game_name=game_name,
                opponent_name=opponent_name
            )