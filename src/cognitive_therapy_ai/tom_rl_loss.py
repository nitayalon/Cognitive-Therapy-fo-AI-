"""
ToM-RL Loss functions for mixed-motive game learning.

This module implements the loss functions described in the ToM-RL framework:
1. Reinforcement Learning Loss (LRL) - Policy gradient with advantage estimation
2. Opponent Prediction Loss (LOp) - Binary cross-entropy for opponent cooperation prediction

The total loss combines these two objectives to create ToM inductive bias while
optimizing for reward maximization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from .games import Action


class ToMRLLoss(nn.Module):
    """
    Theory of Mind Reinforcement Learning Loss Function.
    
    Combines reinforcement learning objective with opponent prediction auxiliary task
    to create inductive bias for understanding opponent behavior (Theory of Mind).
    
    Total Loss: L = LRL_norm + α * LOp_norm
    
    Where:
    - LRL_norm: Normalized Reinforcement Learning Loss (policy gradient with advantage)
    - LOp_norm: Normalized Opponent Prediction Loss (binary cross-entropy)
    - α: Weight balancing the two objectives
    
    Normalization ensures both loss components are on comparable scales for stable training.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 0.99,
        use_gae: bool = True,
        gae_lambda: float = 0.95,
        normalize_losses: bool = True,
        reward_scale: float = 1.0,
        temperature: float = 0.1
    ):
        """
        Initialize ToM-RL loss function.
        
        Args:
            alpha: Weight for opponent prediction loss (α in the paper)
            gamma: Discount factor for reward calculation
            use_gae: Whether to use Generalized Advantage Estimation
            gae_lambda: GAE lambda parameter for bias-variance tradeoff
            normalize_losses: Whether to normalize losses for comparable scales
            reward_scale: Expected scale of rewards (for normalization)
            temperature: Temperature for opponent policy softmax (low = sharp predictions)
        """
        super(ToMRLLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.normalize_losses = normalize_losses
        self.reward_scale = reward_scale
        self.temperature = temperature
        
        # Running statistics for adaptive normalization
        self.register_buffer('rl_loss_ema', torch.tensor(1.0))
        self.register_buffer('op_loss_ema', torch.tensor(1.0))
        self.register_buffer('update_count', torch.tensor(0))
        self.ema_decay = 0.99
        
        # Theoretical bounds for normalization
        self.max_bce_loss = torch.log(torch.tensor(2.0))  # ln(2) ≈ 0.693
        
        self.logger = logging.getLogger(__name__)
    
    def forward(
        self,
        policy_logits: torch.Tensor,
        opponent_policy_logits: torch.Tensor,
        value_estimates: torch.Tensor,
        actions_taken: torch.Tensor,
        rewards: torch.Tensor,
        opponent_actions: torch.Tensor,
        true_opponent_policy: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        next_value: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate ToM-RL composite loss.
        
        Args:
            policy_logits: Action logits from policy head (batch_size, num_actions)
            opponent_policy_logits: Predicted opponent policy logits (batch_size, 2)
            value_estimates: Value function estimates (batch_size, 1)
            actions_taken: Actions actually taken by the agent (batch_size,)
            rewards: Rewards received (batch_size,)
            opponent_actions: True opponent actions (batch_size,) - 1 for cooperate, 0 for defect
            true_opponent_policy: True opponent policy from opponent type (batch_size, 2)
            dones: Episode termination flags (batch_size,), optional
            next_value: Value estimate for next state (for advantage calculation)
            
        Returns:
            Dictionary containing individual losses and total loss
        """
        batch_size = policy_logits.size(0)
        
        # 1. Calculate Reinforcement Learning Loss (LRL)
        rl_loss, advantages = self._calculate_rl_loss(
            policy_logits, actions_taken, rewards, value_estimates, dones, next_value
        )
        
        # 2. Calculate Opponent Policy Loss (LOp)
        opponent_pred_loss = self._calculate_opponent_policy_loss(
            opponent_policy_logits, true_opponent_policy
        )
        
        # 3. Apply normalization if enabled
        if self.normalize_losses:
            rl_loss_norm, opponent_pred_loss_norm = self._normalize_losses(
                rl_loss, opponent_pred_loss
            )
        else:
            rl_loss_norm = rl_loss
            opponent_pred_loss_norm = opponent_pred_loss
        
        # 4. Combine normalized losses
        total_loss = rl_loss_norm + self.alpha * opponent_pred_loss_norm
        
        # Enhanced loss component analysis for debugging
        loss_ratio = (opponent_pred_loss_norm / (rl_loss_norm + 1e-8)).item()
        alpha_contribution = (self.alpha * opponent_pred_loss_norm / total_loss).item()
        
        # Log loss component analysis every few iterations
        if hasattr(self, 'logger') and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Loss Components - RL: {rl_loss.item():.6f}, OpPolicy: {opponent_pred_loss.item():.6f}")
            self.logger.debug(f"Loss Normalized - RL: {rl_loss_norm.item():.6f}, OpPolicy: {opponent_pred_loss_norm.item():.6f}")
            self.logger.debug(f"Loss Ratio (OpPolicy/RL): {loss_ratio:.4f}, Alpha Contribution: {alpha_contribution:.4f}")
        
        return {
            'total_loss': total_loss,
            'rl_loss': rl_loss,
            'rl_loss_normalized': rl_loss_norm,
            'opponent_policy_loss': opponent_pred_loss,
            'opponent_policy_loss_normalized': opponent_pred_loss_norm,
            'loss_ratio': loss_ratio,
            'alpha_contribution': alpha_contribution,
            'advantages': advantages,  # For logging/analysis
            'alpha': self.alpha,
            'temperature': self.temperature,
            'normalization_stats': {
                'rl_loss_scale': self.rl_loss_ema.item(),
                'op_loss_scale': self.op_loss_ema.item()
            }
        }
    
    def _calculate_rl_loss(
        self,
        policy_logits: torch.Tensor,
        actions_taken: torch.Tensor,
        rewards: torch.Tensor,
        value_estimates: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        next_value: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate reinforcement learning loss using policy gradient with advantage.
        
        LRL = -E_t[log π(a_t|s_t) * Â_t]
        
        Where Â_t is the advantage estimate: Â_t = R_t - V_θ(s_t)
        """
        # Convert logits to log probabilities
        log_probs = F.log_softmax(policy_logits, dim=-1)
        
        # Gather log probabilities for actions taken
        action_log_probs = log_probs.gather(1, actions_taken.unsqueeze(1)).squeeze(1)
        
        # Calculate advantages
        advantages = self._calculate_advantages(
            rewards, value_estimates, dones, next_value
        )
        
        # Policy gradient loss: -E[log π(a|s) * advantage]
        rl_loss = -(action_log_probs * advantages.detach()).mean()
        
        return rl_loss, advantages
    
    def _calculate_opponent_policy_loss(
        self,
        predicted_opponent_logits: torch.Tensor,
        true_opponent_policy: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate opponent policy loss using KL divergence.
        
        L_Op = KL(π_true || π_pred) = Σ π_true(a) * log(π_true(a) / π_pred(a))
        
        Where:
        - π_true(a) is the true opponent policy from opponent type parameters
        - π_pred(a) = softmax(opponent_logits / temperature) is the predicted policy
        
        Args:
            predicted_opponent_logits: Network predicted opponent policy logits (batch_size, 2)
            true_opponent_policy: True opponent policy distribution (batch_size, 2)
                                 Format: [p_defect, p_cooperate] where p_defect = 1-p_d, p_cooperate = p_d
        
        Returns:
            KL divergence loss (scalar)
        """
        # Convert predicted logits to policy distribution using temperature
        predicted_policy = F.softmax(predicted_opponent_logits / self.temperature, dim=-1)
        
        # Ensure true policy is normalized (should already be, but safety check)
        true_policy_normalized = true_opponent_policy / (true_opponent_policy.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Add small epsilon to avoid log(0) and division by 0
        eps = 1e-8
        predicted_policy = predicted_policy + eps
        true_policy_normalized = true_policy_normalized + eps
        
        # Calculate KL divergence: KL(true || pred) = Σ true * log(true / pred)
        # Note: We use log_softmax for numerical stability
        log_predicted_policy = F.log_softmax(predicted_opponent_logits / self.temperature, dim=-1)
        
        # KL(P || Q) = Σ P(x) * log(P(x)) - Σ P(x) * log(Q(x))
        # First term: entropy of true policy (constant, doesn't affect gradients)
        entropy_true = -torch.sum(true_policy_normalized * torch.log(true_policy_normalized), dim=-1)
        
        # Second term: negative cross-entropy between true and predicted
        cross_entropy = -torch.sum(true_policy_normalized * log_predicted_policy, dim=-1)
        
        # KL divergence = cross_entropy - entropy_true
        # But since entropy_true is constant, we can just use cross_entropy for gradient-based learning
        kl_loss = cross_entropy.mean()
        
        # Detailed logging for debugging opponent policy learning
        if hasattr(self, 'logger') and self.logger.isEnabledFor(logging.DEBUG):
            with torch.no_grad():
                pred_probs = F.softmax(predicted_opponent_logits / self.temperature, dim=-1)
                avg_pred_defect = pred_probs[:, 0].mean().item()
                avg_pred_cooperate = pred_probs[:, 1].mean().item()
                avg_true_defect = true_policy_normalized[:, 0].mean().item()
                avg_true_cooperate = true_policy_normalized[:, 1].mean().item()
                
                self.logger.debug(f"Opponent Policy Loss: {kl_loss.item():.6f}")
                self.logger.debug(f"Predicted: [defect={avg_pred_defect:.4f}, cooperate={avg_pred_cooperate:.4f}]")
                self.logger.debug(f"True: [defect={avg_true_defect:.4f}, cooperate={avg_true_cooperate:.4f}]")
        
        return kl_loss
    
    def _normalize_losses(
        self,
        rl_loss: torch.Tensor,
        opponent_pred_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize losses to make them comparable in scale.
        
        Uses exponential moving average to track typical loss magnitudes
        and normalizes current losses by their running scales.
        
        Args:
            rl_loss: Raw reinforcement learning loss
            opponent_pred_loss: Raw opponent prediction loss
            
        Returns:
            Tuple of (normalized_rl_loss, normalized_opponent_pred_loss)
        """
        # Update running averages
        if self.training:
            self.update_count += 1
            
            # Use smaller decay for initial updates
            decay = min(self.ema_decay, 1.0 - 1.0 / self.update_count.float())
            
            # Update exponential moving averages
            self.rl_loss_ema = decay * self.rl_loss_ema + (1 - decay) * rl_loss.detach()
            self.op_loss_ema = decay * self.op_loss_ema + (1 - decay) * opponent_pred_loss.detach()
        
        # Normalize losses by their running scales
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        
        # RL loss normalization: scale by running average
        rl_loss_normalized = rl_loss / (self.rl_loss_ema + epsilon)
        
        # Opponent prediction loss normalization: scale by running average
        op_loss_normalized = opponent_pred_loss / (self.op_loss_ema + epsilon)
        
        return rl_loss_normalized, op_loss_normalized
    
    def get_reward_scale_estimate(self, rewards: torch.Tensor) -> float:
        """
        Estimate reward scale from current batch for adaptive normalization.
        
        Args:
            rewards: Batch of rewards
            
        Returns:
            Estimated reward scale (standard deviation or range)
        """
        if len(rewards) <= 1:
            return self.reward_scale
        
        # Use standard deviation as scale estimate
        reward_std = torch.std(rewards)
        
        # Fallback to range if std is too small
        if reward_std < 1e-6:
            reward_range = torch.max(rewards) - torch.min(rewards)
            return max(reward_range.item(), self.reward_scale)
        
        return max(reward_std.item(), self.reward_scale)
    
    def reset_normalization_stats(self):
        """Reset running statistics for normalization."""
        self.rl_loss_ema.fill_(1.0)
        self.op_loss_ema.fill_(1.0)
        self.update_count.fill_(0)
    
    def _calculate_advantages(
        self,
        rewards: torch.Tensor,
        value_estimates: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        next_value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate advantage estimates.
        
        Two methods supported:
        1. Simple advantage: A_t = R_t - V(s_t)
        2. GAE (Generalized Advantage Estimation): More sophisticated bias-variance tradeoff
        """
        if dones is None:
            dones = torch.zeros_like(rewards)
        
        if next_value is None:
            next_value = torch.zeros_like(value_estimates[:1])
        
        if self.use_gae:
            return self._calculate_gae_advantages(rewards, value_estimates, dones, next_value)
        else:
            return self._calculate_simple_advantages(rewards, value_estimates, dones, next_value)
    
    def _calculate_simple_advantages(
        self,
        rewards: torch.Tensor,
        value_estimates: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> torch.Tensor:
        """Calculate simple advantage: A_t = R_t - V(s_t)"""
        # Calculate returns
        returns = self._calculate_returns(rewards, dones, next_value)
        
        # Advantage = Return - Value estimate
        advantages = returns - value_estimates.squeeze(1)
        
        return advantages
    
    def _calculate_gae_advantages(
        self,
        rewards: torch.Tensor,
        value_estimates: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> torch.Tensor:
        """Calculate GAE advantages with lambda-return."""
        batch_size = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        
        # Prepare tensors for concatenation
        values_squeezed = value_estimates.squeeze(1)  # Shape: (batch_size,)
        
        # Ensure next_value has compatible shape
        if next_value.dim() == 0:  # Scalar tensor
            next_value_tensor = next_value.unsqueeze(0)  # Shape: (1,)
        else:
            next_value_tensor = next_value.squeeze()  # Remove extra dimensions
            if next_value_tensor.dim() == 0:  # Still scalar after squeeze
                next_value_tensor = next_value_tensor.unsqueeze(0)  # Shape: (1,)
        
        # Extend value estimates with next_value
        values_extended = torch.cat([values_squeezed, next_value_tensor])
        
        gae = 0
        for t in reversed(range(batch_size)):
            delta = rewards[t] + self.gamma * values_extended[t + 1] * (1 - dones[t]) - values_extended[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages
    
    def _calculate_returns(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> torch.Tensor:
        """Calculate discounted returns."""
        batch_size = rewards.size(0)
        returns = torch.zeros_like(rewards)
        
        # Start with next_value for the last timestep - handle scalar case
        if next_value.dim() == 0:  # Scalar tensor
            R = next_value
        else:
            R = next_value.squeeze()
            if R.dim() > 0:  # If still multi-dimensional after squeeze, take first element
                R = R[0]
        
        for t in reversed(range(batch_size)):
            R = rewards[t] + self.gamma * R * (1 - dones[t])
            returns[t] = R
        
        return returns


class AdaptiveToMRLLoss(ToMRLLoss):
    """
    Adaptive version of ToM-RL loss that automatically adjusts α based on training progress.
    
    This version monitors the relative magnitudes of RL and opponent prediction losses
    and adjusts their balance to prevent one objective from dominating.
    """
    
    def __init__(
        self,
        initial_alpha: float = 1.0,
        gamma: float = 0.99,
        use_gae: bool = True,
        gae_lambda: float = 0.95,
        normalize_losses: bool = True,
        reward_scale: float = 1.0,
        adaptation_rate: float = 0.01,
        min_alpha: float = 0.1,
        max_alpha: float = 10.0
    ):
        """
        Initialize adaptive ToM-RL loss.
        
        Args:
            initial_alpha: Initial weight for opponent prediction loss
            gamma: Discount factor
            use_gae: Whether to use GAE
            gae_lambda: GAE lambda parameter
            normalize_losses: Whether to normalize losses for comparable scales
            reward_scale: Expected scale of rewards (for normalization)
            adaptation_rate: Rate of adaptation for α
            min_alpha: Minimum allowed α value
            max_alpha: Maximum allowed α value
        """
        super().__init__(initial_alpha, gamma, use_gae, gae_lambda, normalize_losses, reward_scale)
        
        self.adaptation_rate = adaptation_rate
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        
        # Track loss histories for adaptation
        self.rl_loss_history = []
        self.opponent_loss_history = []
        self.adaptation_steps = 0
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive α adjustment."""
        loss_dict = super().forward(*args, **kwargs)
        
        # Update adaptation
        self._update_alpha(loss_dict)
        
        return loss_dict
    
    def _update_alpha(self, loss_dict: Dict[str, torch.Tensor]):
        """Update α based on loss balance."""
        # Use normalized losses if available for better balance assessment
        if self.normalize_losses and 'rl_loss_normalized' in loss_dict:
            rl_loss = loss_dict['rl_loss_normalized'].item()
            opponent_loss = loss_dict['opponent_policy_loss_normalized'].item()
        else:
            rl_loss = loss_dict['rl_loss'].item()
            opponent_loss = loss_dict['opponent_policy_loss'].item()
            
        # Record current losses
        self.rl_loss_history.append(rl_loss)
        self.opponent_loss_history.append(opponent_loss)
        
        # Keep only recent history
        history_length = 50
        if len(self.rl_loss_history) > history_length:
            self.rl_loss_history = self.rl_loss_history[-history_length:]
            self.opponent_loss_history = self.opponent_loss_history[-history_length:]
        
        self.adaptation_steps += 1
        
        # Update α every 10 steps
        if self.adaptation_steps % 10 == 0 and len(self.rl_loss_history) >= 10:
            self._adapt_alpha()
    
    def _adapt_alpha(self):
        """Adapt α based on recent loss magnitudes."""
        # Calculate recent average losses
        recent_rl_loss = np.mean(self.rl_loss_history[-10:])
        recent_opponent_loss = np.mean(self.opponent_loss_history[-10:])
        
        if recent_opponent_loss > 0:
            # If RL loss is much larger than opponent loss, increase α
            # If opponent loss is much larger than RL loss, decrease α
            loss_ratio = recent_rl_loss / recent_opponent_loss
            
            if loss_ratio > 2.0:  # RL loss dominates
                target_alpha = self.alpha * 1.1
            elif loss_ratio < 0.5:  # Opponent loss dominates
                target_alpha = self.alpha * 0.9
            else:
                target_alpha = self.alpha  # Keep current α
            
            # Update α with momentum
            self.alpha += self.adaptation_rate * (target_alpha - self.alpha)
            
            # Clamp α to valid range
            self.alpha = np.clip(self.alpha, self.min_alpha, self.max_alpha)


class LossAnalyzer:
    """
    Utility class for analyzing ToM-RL loss components during training.
    """
    
    def __init__(self):
        self.loss_history = {
            'total': [],
            'rl': [],
            'opponent_policy': [],
            'alpha_values': [],
            'epoch': []
        }
    
    def record_loss(self, loss_dict: Dict[str, torch.Tensor], epoch: int):
        """Record loss values for analysis."""
        self.loss_history['total'].append(loss_dict['total_loss'].item())
        self.loss_history['rl'].append(loss_dict['rl_loss'].item())
        self.loss_history['opponent_policy'].append(loss_dict['opponent_policy_loss'].item())
        alpha_value = loss_dict.get('alpha', 1.0)
        if torch.is_tensor(alpha_value):
            alpha_value = alpha_value.item() if alpha_value.numel() == 1 else alpha_value.mean().item()
        self.loss_history['alpha_values'].append(alpha_value)
        self.loss_history['epoch'].append(epoch)
        
        # Record normalized losses if available
        if 'rl_loss_normalized' in loss_dict:
            if 'rl_normalized' not in self.loss_history:
                self.loss_history['rl_normalized'] = []
                self.loss_history['opponent_policy_normalized'] = []
                
            self.loss_history['rl_normalized'].append(loss_dict['rl_loss_normalized'].item())
            self.loss_history['opponent_policy_normalized'].append(
                loss_dict['opponent_policy_loss_normalized'].item()
            )
    
    def get_loss_balance_ratio(self, window_size: int = 50) -> float:
        """
        Get the ratio of RL loss to opponent prediction loss.
        
        This helps assess whether the two objectives are balanced.
        """
        if len(self.loss_history['rl']) < window_size:
            window_size = len(self.loss_history['rl'])
        
        if window_size == 0:
            return 1.0
        
        recent_rl = np.mean(self.loss_history['rl'][-window_size:])
        recent_opponent = np.mean(self.loss_history['opponent_policy'][-window_size:])
        
        if recent_opponent > 0:
            return recent_rl / recent_opponent
        else:
            return float('inf')
    
    def get_tom_contribution(self, window_size: int = 50) -> float:
        """
        Get the relative contribution of ToM component to total loss.
        
        Returns fraction of total loss that comes from opponent prediction.
        """
        if len(self.loss_history['total']) < window_size:
            window_size = len(self.loss_history['total'])
        
        if window_size == 0:
            return 0.0
        
        recent_total = np.mean(self.loss_history['total'][-window_size:])
        recent_opponent = np.mean(self.loss_history['opponent_policy'][-window_size:])
        recent_alpha = np.mean(self.loss_history['alpha_values'][-window_size:])
        
        if recent_total > 0:
            return (recent_alpha * recent_opponent) / recent_total
        else:
            return 0.0


def estimate_game_reward_scale(game_name: str) -> float:
    """
    Estimate appropriate reward scale for normalization based on game type.
    
    Args:
        game_name: Name of the game (e.g., 'prisoners-dilemma', 'hawk-dove')
        
    Returns:
        Estimated reward scale for the game
    """
    # Default payoff ranges for each game type
    game_scales = {
        'prisoners-dilemma': 5.0,  # T=5, R=3, P=1, S=0 -> range ~5
        'hawk-dove': 6.0,          # V=6, cost varies -> range ~6  
        'battle-of-sexes': 3.0,    # Preferred=3, dispreferred=2 -> range ~3
        'stag-hunt': 4.0           # Typical range for stag hunt games
    }
    
    # Normalize game name
    normalized_name = game_name.lower().replace('_', '-')
    
    return game_scales.get(normalized_name, 3.0)  # Default to 3.0


def create_normalized_loss(
    game_name: str,
    alpha: float = 1.0,
    adaptive: bool = False,
    **kwargs
) -> ToMRLLoss:
    """
    Factory function to create ToM-RL loss with appropriate normalization for a game.
    
    Args:
        game_name: Name of the game for reward scale estimation
        alpha: Weight for opponent prediction loss
        adaptive: Whether to use adaptive alpha adjustment
        **kwargs: Additional arguments for loss function
        
    Returns:
        Configured ToM-RL loss function with normalization
    """
    # Estimate reward scale for the game
    reward_scale = estimate_game_reward_scale(game_name)
    
    # Set default normalization parameters
    kwargs.setdefault('normalize_losses', True)
    kwargs.setdefault('reward_scale', reward_scale)
    
    if adaptive:
        return AdaptiveToMRLLoss(alpha, **kwargs)
    else:
        return ToMRLLoss(alpha, **kwargs)