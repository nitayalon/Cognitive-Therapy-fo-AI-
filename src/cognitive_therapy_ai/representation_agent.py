"""
Simplified single-layer LSTM agent for representation learning.

This architecture is designed for:
1. Discrete one-hot inputs (d_in = 8 or 11) → NO embeddings
2. Single LSTM layer with variable hidden size H
3. Direct linear readout: H → 2 policy logits, H → 1 value estimate
4. Exact FSM extraction via L* algorithm

Key differences from deprecated GameLSTM:
- NO embedding pathways (feed one-hot directly to LSTM)
- Single layer (NOT 2 layers)
- Smaller hidden sizes (H ≤ 32, NOT 128)
- No ToM opponent prediction head (focus on RL only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np


class RepresentationAgent(nn.Module):
    """
    Single-layer LSTM agent with direct one-hot input.
    
    Architecture:
    - Input: one-hot vector (d_in = 8 or 11)
    - LSTM: 1 layer, hidden size H
    - Policy head: Linear(H, 2) → logits for [Cooperate, Defect]
    - Value head: Linear(H, 1) → scalar value estimate
    
    Parameters:
        input_dim (int): Dimension of one-hot input (8 for no_game, 11 for game_tag)
        hidden_size (int): LSTM hidden size H (typically 2-32 for interpretability)
        
    Forward pass:
        obs (batch, d_in) → LSTM → h_t (batch, H) → {policy_logits, value}
        
    Hidden state management:
        - Persists across T=100 rounds within a session
        - Resets between sessions via reset_hidden_state()
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.device = device if device is not None else torch.device('cpu')
        
        # Single-layer LSTM (no embeddings)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Policy head: H → 2 logits (Cooperate, Defect)
        self.policy_head = nn.Linear(hidden_size, 2)
        
        # Value head: H → 1 scalar (for advantage estimation)
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Hidden state (h_t, c_t) - initialized lazily
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        
        self.to(self.device)
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM and heads.
        
        Args:
            obs: One-hot observation tensor (batch_size, input_dim)
            hidden_state: Optional (h_t, c_t) from previous step
                         If None, uses zeros (start of session)
        
        Returns:
            policy_logits: (batch_size, 2) logits for [Cooperate, Defect]
            value: (batch_size, 1) value estimate
            new_hidden_state: (h_t+1, c_t+1) for next step
        """
        batch_size = obs.shape[0]
        
        # Ensure obs is (batch, seq=1, input_dim) for LSTM
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, input_dim)
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            h = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
            c = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
            hidden_state = (h, c)
        
        # LSTM forward pass
        lstm_out, new_hidden_state = self.lstm(obs, hidden_state)
        # lstm_out: (batch, seq=1, H)
        
        # Extract final hidden state
        h_t = lstm_out[:, -1, :]  # (batch, H)
        
        # Policy logits
        policy_logits = self.policy_head(h_t)  # (batch, 2)
        
        # Value estimate
        value = self.value_head(h_t)  # (batch, 1)
        
        return policy_logits, value, new_hidden_state
    
    def select_action(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
        epsilon: float = 0.0
    ) -> Tuple[int, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Select action using current policy.
        
        Args:
            obs: One-hot observation (input_dim,) or (1, input_dim)
            hidden_state: Optional (h_t, c_t) from previous step
            deterministic: If True, select argmax action (for evaluation)
            epsilon: Epsilon-greedy exploration (for rollout phase)
        
        Returns:
            action: Integer action (0=Cooperate, 1=Defect)
            policy_probs: (2,) probability distribution over actions
            new_hidden_state: Updated (h_t+1, c_t+1)
        """
        # Ensure obs is batched
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # (1, input_dim)
        
        # Forward pass
        with torch.no_grad():
            policy_logits, _, new_hidden_state = self.forward(obs, hidden_state)
            policy_probs = F.softmax(policy_logits, dim=1).squeeze(0)  # (2,)
        
        # Epsilon-greedy exploration
        if epsilon > 0 and np.random.rand() < epsilon:
            action = np.random.choice([0, 1])
        elif deterministic:
            action = torch.argmax(policy_probs).item()
        else:
            # Sample from policy
            action = torch.multinomial(policy_probs, num_samples=1).item()
        
        return action, policy_probs, new_hidden_state
    
    def reset_hidden_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reset hidden state to zeros (start of new session).
        
        Args:
            batch_size: Batch size for hidden state
        
        Returns:
            (h_0, c_0): Zero-initialized hidden state
        """
        h = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        return (h, c)
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters in each component.
        
        Returns dictionary with:
            lstm_params: LSTM parameters
            policy_params: Policy head parameters
            value_params: Value head parameters
            total_params: Total trainable parameters
        """
        lstm_params = sum(p.numel() for p in self.lstm.parameters())
        policy_params = sum(p.numel() for p in self.policy_head.parameters())
        value_params = sum(p.numel() for p in self.value_head.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'lstm_params': lstm_params,
            'policy_params': policy_params,
            'value_params': value_params,
            'total_params': total_params
        }
    
    def get_analytic_parameter_count(self) -> int:
        """
        Compute expected parameter count using LSTM formula.
        
        LSTM parameters (single layer):
            For each gate (input, forget, cell, output):
                W_ih: (H, d_in)
                W_hh: (H, H)
                b_ih: (H,)
                b_hh: (H,)
            Total: 4 * (H*d_in + H*H + H + H)
                 = 4 * (H*d_in + H² + 2H)
        
        Policy head:
            W: (2, H)
            b: (2,)
            Total: 2H + 2
        
        Value head:
            W: (1, H)
            b: (1,)
            Total: H + 1
        
        Grand total: 4*(H*d_in + H² + 2H) + (2H + 2) + (H + 1)
                   = 4*H*d_in + 4*H² + 8H + 2H + 2 + H + 1
                   = 4*H*d_in + 4*H² + 11H + 3
        """
        H = self.hidden_size
        d_in = self.input_dim
        
        lstm_params = 4 * (H * d_in + H * H + 2 * H)
        policy_params = 2 * H + 2
        value_params = H + 1
        
        total = lstm_params + policy_params + value_params
        return total
    
    def verify_parameter_count(self) -> Tuple[bool, Dict[str, int]]:
        """
        Verify that actual parameter count matches analytic formula.
        
        Returns:
            is_correct: True if counts match
            details: Dictionary with actual vs expected counts
        """
        actual = self.count_parameters()
        expected = self.get_analytic_parameter_count()
        
        is_correct = (actual['total_params'] == expected)
        
        details = {
            'actual_total': actual['total_params'],
            'expected_total': expected,
            'match': is_correct,
            'breakdown': actual
        }
        
        return is_correct, details


def create_agent_from_config(
    config: Dict[str, Any],
    input_condition: str,
    device: Optional[torch.device] = None
) -> RepresentationAgent:
    """
    Factory function to create agent from configuration.
    
    Args:
        config: Configuration dictionary with agent parameters
        input_condition: "no_game" (d_in=8) or "game_tag" (d_in=11)
        device: Torch device
    
    Returns:
        RepresentationAgent instance
    """
    # Determine input dimension from encoding config
    from .encoding import ObservationEncoder
    encoder = ObservationEncoder(input_condition)
    input_dim = encoder.get_input_dim()
    
    # Get hidden size from config
    hidden_size = config.get('hidden_size', 8)
    
    return RepresentationAgent(
        input_dim=input_dim,
        hidden_size=hidden_size,
        device=device
    )
