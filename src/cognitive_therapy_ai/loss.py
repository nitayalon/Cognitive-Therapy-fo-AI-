"""
Loss functions for multi-task LSTM training in mixed-motive games.

This module implements the composite loss function that combines:
1. Policy gradient loss (for maximizing cumulative reward)
2. Cross-entropy loss (for opponent action prediction)  
3. MSE loss (for opponent type prediction)

For the new ToM-RL framework, see tom_rl_loss.py module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging

from .games import Action

# Import new ToM-RL losses for compatibility
from .tom_rl_loss import ToMRLLoss, AdaptiveToMRLLoss


class CompositeLoss(nn.Module):
    """
    Composite loss function for multi-task learning in mixed-motive games.
    
    Combines three loss components:
    1. Policy gradient loss: REINFORCE-style loss for action selection
    2. Opponent action prediction loss: Cross-entropy loss
    3. Opponent type prediction loss: MSE loss for predicting defection probability
    """
    
    def __init__(
        self,
        reward_weight: float = 1.0,
        action_prediction_weight: float = 1.0,
        type_prediction_weight: float = 1.0,
        use_baseline: bool = True
    ):
        """
        Initialize composite loss function.
        
        Args:
            reward_weight: Weight for policy gradient loss
            action_prediction_weight: Weight for opponent action prediction loss
            type_prediction_weight: Weight for opponent type prediction loss
            use_baseline: Whether to use baseline subtraction in policy gradient
        """
        super(CompositeLoss, self).__init__()
        
        self.reward_weight = reward_weight
        self.action_prediction_weight = action_prediction_weight
        self.type_prediction_weight = type_prediction_weight
        self.use_baseline = use_baseline
        
        # Loss functions
        self.action_prediction_criterion = nn.CrossEntropyLoss()
        self.type_prediction_criterion = nn.MSELoss()
        
        self.logger = logging.getLogger(__name__)
    
    def forward(
        self,
        policy_logits: torch.Tensor,
        opponent_action_logits: torch.Tensor,
        opponent_type_preds: torch.Tensor,
        actions_taken: torch.Tensor,
        rewards: torch.Tensor,
        opponent_actions: torch.Tensor,
        opponent_type_true: torch.Tensor,
        baseline_values: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate composite loss for multi-task learning in mixed-motive games.
        
        TENSOR SHAPE REQUIREMENTS (following PyTorch conventions):
        
        Args:
            policy_logits: Logits for action selection 
                Shape: (batch_size, num_actions), dtype: float32
                Used in: Policy gradient loss (REINFORCE)
                
            opponent_action_logits: Logits for opponent action prediction
                Shape: (batch_size, num_actions), dtype: float32  
                Used in: CrossEntropyLoss as input
                
            opponent_type_preds: Predictions for opponent type (defection probability)
                Shape: (batch_size, 1), dtype: float32
                Used in: MSELoss as input
                
            actions_taken: Actions actually taken by the agent
                Shape: (batch_size,), dtype: int64 (LongTensor)
                Used in: Policy gradient loss for indexing log probabilities
                
            rewards: Rewards received for actions
                Shape: (batch_size,), dtype: float32
                Used in: Policy gradient loss as advantages
                
            opponent_actions: True opponent actions (ground truth)
                Shape: (batch_size,), dtype: int64 (LongTensor) 
                Used in: CrossEntropyLoss as target (class indices)
                
            opponent_type_true: True opponent type parameter
                Shape: (batch_size, 1), dtype: float32
                Used in: MSELoss as target
                
            baseline_values: Optional baseline values for variance reduction
                Shape: (batch_size,), dtype: float32, optional
                Used in: Policy gradient loss for advantage calculation
            
        Returns:
            Dictionary containing individual losses and total loss
            
        Note:
            batch_size represents the total number of game steps across all sessions
            in multi-game training. Each session contributes T steps, so 
            batch_size = sum(T_i) for all sessions i in the mixed batch.
        """
        batch_size = policy_logits.size(0)
        
        # Validate tensor shapes for loss function requirements
        try:
            self.logger.debug(f"Input tensor shapes:")
            self.logger.debug(f"  policy_logits: {policy_logits.shape}")
            self.logger.debug(f"  opponent_action_logits: {opponent_action_logits.shape}")
            self.logger.debug(f"  opponent_type_preds: {opponent_type_preds.shape}")
            self.logger.debug(f"  actions_taken: {actions_taken.shape}")
            self.logger.debug(f"  rewards: {rewards.shape}")
            self.logger.debug(f"  opponent_actions: {opponent_actions.shape}")
            self.logger.debug(f"  opponent_type_true: {opponent_type_true.shape}")
            
            # Get consistent batch size from 2D tensors
            batch_size = policy_logits.size(0)
            
            # Validate tensor shapes match PyTorch loss function requirements
            # CrossEntropyLoss: input=(N, C), target=(N,) as LongTensor
            # MSELoss: input=(N, *), target=(N, *) same shape
            # Policy gradient: element-wise operations need same batch size
            
            # Ensure opponent_actions is 1D LongTensor for CrossEntropyLoss
            if opponent_actions.dim() != 1 or opponent_actions.size(0) != batch_size:
                self.logger.error(f"opponent_actions shape mismatch: expected ({batch_size},), got {opponent_actions.shape}")
                raise ValueError(f"opponent_actions must be 1D with batch_size {batch_size}, got shape {opponent_actions.shape}")
            opponent_actions = opponent_actions.long()
            
            # Ensure actions_taken is 1D LongTensor for policy loss
            if actions_taken.dim() != 1 or actions_taken.size(0) != batch_size:
                self.logger.error(f"actions_taken shape mismatch: expected ({batch_size},), got {actions_taken.shape}")
                raise ValueError(f"actions_taken must be 1D with batch_size {batch_size}, got shape {actions_taken.shape}")
            actions_taken = actions_taken.long()
            
            # Ensure rewards is 1D FloatTensor for policy loss
            if rewards.dim() != 1 or rewards.size(0) != batch_size:
                self.logger.error(f"rewards shape mismatch: expected ({batch_size},), got {rewards.shape}")
                raise ValueError(f"rewards must be 1D with batch_size {batch_size}, got shape {rewards.shape}")
            
            # Validate 2D tensors have correct batch size
            expected_2d_shapes = {
                'policy_logits': (batch_size, policy_logits.size(1)),
                'opponent_action_logits': (batch_size, opponent_action_logits.size(1)),
                'opponent_type_preds': (batch_size, opponent_type_preds.size(1)),
                'opponent_type_true': (batch_size, opponent_type_true.size(1))
            }
            
            for name, tensor in [
                ('policy_logits', policy_logits),
                ('opponent_action_logits', opponent_action_logits), 
                ('opponent_type_preds', opponent_type_preds),
                ('opponent_type_true', opponent_type_true)
            ]:
                if tensor.shape != expected_2d_shapes[name]:
                    self.logger.error(f"{name} shape mismatch: expected {expected_2d_shapes[name]}, got {tensor.shape}")
                    raise ValueError(f"{name} shape mismatch: expected {expected_2d_shapes[name]}, got {tensor.shape}")
            
            # Handle baseline_values if provided
            if baseline_values is not None:
                if baseline_values.dim() != 1 or baseline_values.size(0) != batch_size:
                    self.logger.error(f"baseline_values shape mismatch: expected ({batch_size},), got {baseline_values.shape}")
                    raise ValueError(f"baseline_values must be 1D with batch_size {batch_size}, got shape {baseline_values.shape}")
            
            self.logger.debug(f"✓ All tensor shapes validated for PyTorch loss functions")
            self.logger.debug(f"  Batch size: {batch_size}")
            self.logger.debug(f"  CrossEntropyLoss - input: {opponent_action_logits.shape}, target: {opponent_actions.shape}")
            self.logger.debug(f"  MSELoss - input: {opponent_type_preds.shape}, target: {opponent_type_true.shape}")
            self.logger.debug(f"  Policy gradient - actions: {actions_taken.shape}, rewards: {rewards.shape}")
            
        except Exception as e:
            self.logger.error(f"Error in tensor shape validation: {e}")
            self.logger.error(f"Tensor shapes: policy_logits={policy_logits.shape}, opponent_action_logits={opponent_action_logits.shape}")
            self.logger.error(f"actions_taken={actions_taken.shape}, opponent_actions={opponent_actions.shape}")
            self.logger.error(f"rewards={rewards.shape}, opponent_type_preds={opponent_type_preds.shape}")
            self.logger.error(f"opponent_type_true={opponent_type_true.shape}")
            raise RuntimeError(f"Failed to validate tensor shapes for loss calculation: {e}")
        
        # 1. Policy Gradient Loss (REINFORCE)
        policy_loss = self._calculate_policy_loss(
            policy_logits, actions_taken, rewards, baseline_values
        )
        
        # 2. Opponent Action Prediction Loss (Cross-entropy)
        try:
            # Validate CrossEntropyLoss requirements before calculation
            num_classes = opponent_action_logits.size(1)
            max_target = opponent_actions.max().item()
            min_target = opponent_actions.min().item()
            
            self.logger.debug(f"CrossEntropyLoss validation:")
            self.logger.debug(f"  Input shape: {opponent_action_logits.shape} (batch_size={batch_size}, num_classes={num_classes})")
            self.logger.debug(f"  Target shape: {opponent_actions.shape}")
            self.logger.debug(f"  Target range: [{min_target}, {max_target}]")
            self.logger.debug(f"  Valid target range: [0, {num_classes-1}]")
            
            # Check if target indices are within valid range
            if max_target >= num_classes:
                self.logger.error(f"Target indices out of bounds: max target {max_target} >= num_classes {num_classes}")
                self.logger.error(f"opponent_actions values: {opponent_actions.unique()}")
                raise ValueError(f"Target indices must be in range [0, {num_classes-1}], but found max target {max_target}")
            
            if min_target < 0:
                self.logger.error(f"Target indices negative: min target {min_target} < 0")
                raise ValueError(f"Target indices must be non-negative, but found min target {min_target}")
            
            # Check for binary action game (expected for mixed-motive games)
            if num_classes != 2:
                self.logger.warning(f"Expected 2 action classes (Cooperate/Defect), got {num_classes}")
                # For binary games, we expect exactly 2 classes
                if num_classes == 1:
                    self.logger.error("Cannot compute CrossEntropyLoss with only 1 class")
                    # Expand to 2 classes if we have degenerate case
                    opponent_action_logits = torch.cat([
                        opponent_action_logits, 
                        torch.zeros_like(opponent_action_logits)
                    ], dim=1)
                    self.logger.debug(f"Expanded opponent_action_logits to shape: {opponent_action_logits.shape}")
            
            action_pred_loss = self.action_prediction_criterion(
                opponent_action_logits, opponent_actions
            )
            
        except Exception as e:
            self.logger.error(f"Error in CrossEntropyLoss calculation: {e}")
            self.logger.error(f"opponent_action_logits: shape={opponent_action_logits.shape}, dtype={opponent_action_logits.dtype}")
            self.logger.error(f"opponent_actions: shape={opponent_actions.shape}, dtype={opponent_actions.dtype}")
            self.logger.error(f"opponent_actions values: {opponent_actions}")
            self.logger.error(f"opponent_action_logits sample: {opponent_action_logits[:5] if len(opponent_action_logits) > 0 else 'empty'}")
            raise RuntimeError(f"Failed to calculate opponent action prediction loss: {e}")
        
        # 3. Opponent Type Prediction Loss (MSE)
        type_pred_loss = self.type_prediction_criterion(
            opponent_type_preds, opponent_type_true
        )
        
        # Combine losses with weights
        total_loss = (
            self.reward_weight * policy_loss +
            self.action_prediction_weight * action_pred_loss +
            self.type_prediction_weight * type_pred_loss
        )
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'action_prediction_loss': action_pred_loss,
            'type_prediction_loss': type_pred_loss,
            'reward_weight': self.reward_weight,
            'action_prediction_weight': self.action_prediction_weight,
            'type_prediction_weight': self.type_prediction_weight
        }
    
    def _calculate_policy_loss(
        self,
        policy_logits: torch.Tensor,
        actions_taken: torch.Tensor,
        rewards: torch.Tensor,
        baseline_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate policy gradient loss using REINFORCE.
        
        Args:
            policy_logits: Action logits from policy network
            actions_taken: Actions that were taken
            rewards: Rewards received for those actions
            baseline_values: Optional baseline values for variance reduction
            
        Returns:
            Policy gradient loss
        """
        # Convert logits to log probabilities
        log_probs = F.log_softmax(policy_logits, dim=-1)
        
        # Validate input shapes and handle dimension mismatches
        try:
            batch_size = policy_logits.size(0)
            num_actions = policy_logits.size(1)
            
            # Ensure actions_taken is in the right format
            # Convert to long type and ensure proper shape
            actions_indices = actions_taken.long()
            
            # Handle different action tensor shapes
            if actions_indices.dim() == 0:
                # Scalar - convert to batch of size 1
                actions_indices = actions_indices.unsqueeze(0).unsqueeze(1)
            elif actions_indices.dim() == 1:
                # 1D tensor (batch_size,) - add dimension for gather
                actions_indices = actions_indices.unsqueeze(1)
            elif actions_indices.dim() == 2:
                # 2D tensor - should be (batch_size, 1) already
                if actions_indices.size(1) != 1:
                    # If more than 1 column, take first column
                    actions_indices = actions_indices[:, 0:1]
            else:
                # Higher dimensions - reshape to (batch_size, 1)
                actions_indices = actions_indices.view(batch_size, 1)
            
            # Validate indices are within bounds
            if actions_indices.max() >= num_actions or actions_indices.min() < 0:
                raise ValueError(f"Action indices out of bounds. Max: {actions_indices.max()}, Min: {actions_indices.min()}, Num actions: {num_actions}")
                
        except Exception as e:
            self.logger.error(f"Error processing actions tensor: {e}")
            self.logger.error(f"policy_logits shape: {policy_logits.shape}")
            self.logger.error(f"actions_taken shape: {actions_taken.shape}")
            self.logger.error(f"actions_taken dtype: {actions_taken.dtype}")
            self.logger.error(f"actions_taken values: {actions_taken}")
            raise RuntimeError(f"Failed to process action indices for loss calculation: {e}")
        
        # Gather log probabilities for actions taken
        action_log_probs = log_probs.gather(1, actions_indices).squeeze(1)
        
        # Ensure rewards has correct shape
        if rewards.dim() > 1:
            rewards = rewards.squeeze()
        
        # Calculate advantages (rewards minus baseline if provided)
        if self.use_baseline and baseline_values is not None:
            if baseline_values.dim() > 1:
                baseline_values = baseline_values.squeeze()
            advantages = rewards - baseline_values
        else:
            advantages = rewards
        
        # Debug logging and shape validation
        try:
            self.logger.debug(f"action_log_probs shape: {action_log_probs.shape}")
            self.logger.debug(f"advantages shape: {advantages.shape}")
            
            # Ensure both tensors have compatible shapes for element-wise multiplication
            if action_log_probs.dim() != advantages.dim():
                if action_log_probs.dim() == 1 and advantages.dim() == 0:
                    # advantages is scalar, expand to match batch size
                    advantages = advantages.expand_as(action_log_probs)
                elif action_log_probs.dim() == 0 and advantages.dim() == 1:
                    # action_log_probs is scalar, expand to match batch size
                    action_log_probs = action_log_probs.expand_as(advantages)
                elif action_log_probs.dim() > advantages.dim():
                    # Expand advantages to match action_log_probs
                    advantages = advantages.view(-1, *([1] * (action_log_probs.dim() - advantages.dim())))
                    advantages = advantages.expand_as(action_log_probs)
                elif advantages.dim() > action_log_probs.dim():
                    # Expand action_log_probs to match advantages
                    action_log_probs = action_log_probs.view(-1, *([1] * (advantages.dim() - action_log_probs.dim())))
                    action_log_probs = action_log_probs.expand_as(advantages)
            
            # Final shape check
            if action_log_probs.shape != advantages.shape:
                self.logger.warning(f"Shape mismatch after alignment: action_log_probs {action_log_probs.shape}, advantages {advantages.shape}")
                # Try broadcasting by reshaping to compatible shapes
                if action_log_probs.numel() == advantages.numel():
                    # Same number of elements, reshape to match
                    min_shape = min(action_log_probs.shape, advantages.shape, key=lambda x: len(x))
                    action_log_probs = action_log_probs.view(-1)[:advantages.numel()].view(advantages.shape)
                else:
                    # Different number of elements, use broadcasting rules
                    # Reshape to be broadcastable
                    batch_size = max(action_log_probs.size(0) if action_log_probs.dim() > 0 else 1, 
                                   advantages.size(0) if advantages.dim() > 0 else 1)
                    action_log_probs = action_log_probs.view(batch_size, -1).mean(dim=1)
                    advantages = advantages.view(batch_size, -1).mean(dim=1)
        
        except Exception as e:
            self.logger.error(f"Error in tensor shape alignment: {e}")
            self.logger.error(f"action_log_probs: shape={action_log_probs.shape}, values={action_log_probs}")
            self.logger.error(f"advantages: shape={advantages.shape}, values={advantages}")
            raise RuntimeError(f"Failed to align tensor shapes for policy loss: {e}")
        
        # Policy gradient loss: -E[log π(a|s) * advantage]
        policy_loss = -(action_log_probs * advantages).mean()
        
        return policy_loss


class AdaptiveLoss(CompositeLoss):
    """
    Adaptive loss function that adjusts weights based on training progress.
    
    This version automatically balances the three loss components during training
    to prevent any single task from dominating.
    """
    
    def __init__(
        self,
        initial_reward_weight: float = 1.0,
        initial_action_weight: float = 1.0,
        initial_type_weight: float = 1.0,
        adaptation_rate: float = 0.01,
        min_weight: float = 0.1,
        max_weight: float = 10.0
    ):
        """
        Initialize adaptive loss function.
        
        Args:
            initial_reward_weight: Initial weight for policy gradient loss
            initial_action_weight: Initial weight for action prediction loss
            initial_type_weight: Initial weight for type prediction loss
            adaptation_rate: Rate at which weights adapt
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
        """
        super().__init__(initial_reward_weight, initial_action_weight, initial_type_weight)
        
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Track loss histories for adaptation
        self.loss_history = {
            'policy': [],
            'action_prediction': [],
            'type_prediction': []
        }
        
        self.adaptation_steps = 0
    
    def update_weights(self, loss_dict: Dict[str, torch.Tensor]):
        """
        Update loss weights based on recent performance.
        
        Args:
            loss_dict: Dictionary containing individual loss values
        """
        # Add current losses to history
        self.loss_history['policy'].append(loss_dict['policy_loss'].item())
        self.loss_history['action_prediction'].append(loss_dict['action_prediction_loss'].item())
        self.loss_history['type_prediction'].append(loss_dict['type_prediction_loss'].item())
        
        # Keep only recent history
        history_length = 100
        for key in self.loss_history:
            if len(self.loss_history[key]) > history_length:
                self.loss_history[key] = self.loss_history[key][-history_length:]
        
        self.adaptation_steps += 1
        
        # Update weights every 10 steps
        if self.adaptation_steps % 10 == 0 and len(self.loss_history['policy']) >= 10:
            self._adapt_weights()
    
    def _adapt_weights(self):
        """Adapt weights based on loss magnitudes."""
        # Calculate recent average losses
        recent_losses = {}
        for key in self.loss_history:
            recent_losses[key] = np.mean(self.loss_history[key][-10:])
        
        # Calculate relative loss magnitudes
        total_loss = sum(recent_losses.values())
        if total_loss > 0:
            # Inverse relationship: higher loss gets lower weight
            policy_ratio = recent_losses['policy'] / total_loss
            action_ratio = recent_losses['action_prediction'] / total_loss
            type_ratio = recent_losses['type_prediction'] / total_loss
            
            # Adjust weights (move towards balance)
            target_reward_weight = 1.0 / (1.0 + policy_ratio)
            target_action_weight = 1.0 / (1.0 + action_ratio)
            target_type_weight = 1.0 / (1.0 + type_ratio)
            
            # Update weights with momentum
            self.reward_weight += self.adaptation_rate * (target_reward_weight - self.reward_weight)
            self.action_prediction_weight += self.adaptation_rate * (target_action_weight - self.action_prediction_weight)
            self.type_prediction_weight += self.adaptation_rate * (target_type_weight - self.type_prediction_weight)
            
            # Clamp weights to valid range
            self.reward_weight = np.clip(self.reward_weight, self.min_weight, self.max_weight)
            self.action_prediction_weight = np.clip(self.action_prediction_weight, self.min_weight, self.max_weight)
            self.type_prediction_weight = np.clip(self.type_prediction_weight, self.min_weight, self.max_weight)


class LossAnalyzer:
    """
    Utility class for analyzing loss components during training.
    """
    
    def __init__(self):
        self.loss_history = {
            'total': [],
            'policy': [],
            'action_prediction': [],
            'type_prediction': [],
            'epoch': []
        }
    
    def record_loss(self, loss_dict: Dict[str, torch.Tensor], epoch: int):
        """Record loss values for analysis."""
        self.loss_history['total'].append(loss_dict['total_loss'].item())
        self.loss_history['policy'].append(loss_dict['policy_loss'].item())
        self.loss_history['action_prediction'].append(loss_dict['action_prediction_loss'].item())
        self.loss_history['type_prediction'].append(loss_dict['type_prediction_loss'].item())
        self.loss_history['epoch'].append(epoch)
    
    def get_loss_statistics(self, window_size: int = 50) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for recent loss values.
        
        Args:
            window_size: Number of recent values to analyze
            
        Returns:
            Dictionary with statistics for each loss component
        """
        stats = {}
        
        for loss_type, values in self.loss_history.items():
            if loss_type == 'epoch':
                continue
            
            recent_values = values[-window_size:] if len(values) > window_size else values
            
            if recent_values:
                stats[loss_type] = {
                    'mean': np.mean(recent_values),
                    'std': np.std(recent_values),
                    'min': np.min(recent_values),
                    'max': np.max(recent_values),
                    'recent': recent_values[-1] if recent_values else 0
                }
        
        return stats
    
    def check_convergence(
        self, 
        loss_type: str = 'total', 
        window_size: int = 50, 
        threshold: float = 1e-6
    ) -> bool:
        """
        Check if loss has converged.
        
        Args:
            loss_type: Type of loss to check
            window_size: Window size for checking convergence
            threshold: Convergence threshold
            
        Returns:
            True if converged, False otherwise
        """
        if loss_type not in self.loss_history:
            return False
        
        values = self.loss_history[loss_type]
        if len(values) < window_size:
            return False
        
        recent_values = values[-window_size:]
        return np.std(recent_values) < threshold
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Get convergence information for all loss types."""
        convergence_info = {}
        
        for loss_type in ['total', 'policy', 'action_prediction', 'type_prediction']:
            convergence_info[loss_type] = {
                'converged': self.check_convergence(loss_type),
                'recent_std': np.std(self.loss_history[loss_type][-50:]) if len(self.loss_history[loss_type]) >= 50 else float('inf'),
                'trend': self._calculate_trend(loss_type)
            }
        
        return convergence_info
    
    def _calculate_trend(self, loss_type: str, window_size: int = 20) -> str:
        """Calculate trend direction for a loss type."""
        values = self.loss_history[loss_type]
        if len(values) < window_size:
            return 'insufficient_data'
        
        recent_values = values[-window_size:]
        first_half = np.mean(recent_values[:window_size//2])
        second_half = np.mean(recent_values[window_size//2:])
        
        if second_half < first_half * 0.95:
            return 'decreasing'
        elif second_half > first_half * 1.05:
            return 'increasing'
        else:
            return 'stable'