"""
LSTM network implementation for mixed-motive game learning.

This module implements a multi-task LSTM network that learns to:
1. Choose actions to maximize cumulative reward (policy)
2. Predict opponent's next action
3. Predict opponent's type parameter (defection probability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

from .games import Action


class GameLSTM(nn.Module):
    """
    Multi-task LSTM network for mixed-motive game learning with Theory of Mind.
    
    The network takes payoff matrix and game metadata as input and outputs:
    1. Action probabilities for policy learning (RL component)
    2. Opponent action prediction probabilities (ToM component)
    3. Value function estimates (for advantage calculation)
    
    This architecture implements the ToM-RL framework where auxiliary prediction
    tasks create inductive bias for understanding opponent behavior.
    
    Input format: [payoff_matrix_flattened(4), round_number(1)] = 5 elements total
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_actions: int = 2
    ):
        """
        Initialize the GameLSTM network.
        
        Args:
            input_size: Size of input state vector
            hidden_size: Hidden size of LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            num_actions: Number of possible actions (typically 2)
        """
        super(GameLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_actions = num_actions
        
        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Policy head - outputs action logits for decision making
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        # Enhanced opponent policy prediction head - dedicated architecture for ToM
        # This creates stronger inductive bias for understanding opponent behavior patterns
        self.opponent_policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),  # Stabilize learning
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Less dropout in final layers
            nn.Linear(hidden_size // 4, 2),  # 2 actions: [defect, cooperate]
            # No activation - raw logits for policy distribution
        )
        
        # Value function head - estimates state value for advantage calculation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with specialized initialization for opponent policy head."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'opponent_policy_head' in name and 'weight' in name and param.dim() >= 2:
                # Specialized initialization for opponent policy prediction
                # Use smaller initialization to prevent saturation
                # Only apply to 2D+ tensors (weight matrices, not bias vectors)
                nn.init.xavier_uniform_(param.data, gain=0.5)
            elif 'opponent_policy_head' in name and 'bias' in name:
                # Initialize final layer bias to encourage balanced predictions
                if param.shape[0] == 2:  # Final layer bias
                    param.data.fill_(0.0)  # Start with balanced predictions
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            hidden: Optional hidden state tuple (h_0, c_0)
            
        Returns:
            Tuple of (policy_logits, opponent_policy_logits, value_estimate, new_hidden)
        """
        # LSTM forward pass
        lstm_out, new_hidden = self.lstm(x, hidden)
        
        # Use last timestep output for predictions
        if lstm_out.dim() == 3:  # (batch_size, sequence_length, hidden_size)
            last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        else:  # Single timestep: (batch_size, hidden_size)
            last_output = lstm_out
        
        # Multi-task outputs
        policy_logits = self.policy_head(last_output)
        opponent_policy_logits = self.opponent_policy_head(last_output)
        value_estimate = self.value_head(last_output)
        
        return policy_logits, opponent_policy_logits, value_estimate, new_hidden
    
    def get_action_probabilities(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Get action probabilities for policy.
        
        Args:
            x: Input tensor
            hidden: Optional hidden state
            
        Returns:
            Action probabilities tensor
        """
        policy_logits, _, _, _ = self.forward(x, hidden)
        return F.softmax(policy_logits, dim=-1)
    
    def sample_action(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[Action, torch.Tensor]:
        """
        Sample an action from the policy.
        
        Args:
            x: Input tensor
            hidden: Optional hidden state
            
        Returns:
            Tuple of (sampled_action, action_log_prob)
        """
        policy_logits, _, _, _ = self.forward(x, hidden)
        action_probs = F.softmax(policy_logits, dim=-1)
        
        # Sample action
        action_dist = torch.distributions.Categorical(action_probs)
        action_idx = action_dist.sample()
        action_log_prob = action_dist.log_prob(action_idx)
        
        # Convert to Action enum
        action = Action.COOPERATE if action_idx.item() == 0 else Action.DEFECT
        
        return action, action_log_prob
    
    def get_value_estimate(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Get value function estimate for advantage calculation.
        
        Args:
            x: Input tensor
            hidden: Optional hidden state
            
        Returns:
            Value estimate tensor
        """
        _, _, value_estimate, _ = self.forward(x, hidden)
        return value_estimate
    
    def get_all_predictions(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Get all network predictions.
        
        Args:
            x: Input tensor
            hidden: Optional hidden state
            
        Returns:
            Dictionary with all predictions
        """
        policy_logits, opponent_policy_logits, value_estimate, new_hidden = self.forward(x, hidden)
        
        return {
            'policy_probs': F.softmax(policy_logits, dim=-1),
            'policy_logits': policy_logits,
            'opponent_policy_logits': opponent_policy_logits,
            'value_estimate': value_estimate,
            'hidden': new_hidden
        }
    
    def analyze_predictions(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Analyze network predictions for debugging opponent policy learning.
        
        Args:
            x: Input tensor
            hidden: Optional hidden state
            
        Returns:
            Dictionary with detailed prediction analysis
        """
        with torch.no_grad():
            policy_logits, opponent_policy_logits, value_estimate, new_hidden = self.forward(x, hidden)
            
            # Agent policy analysis
            agent_probs = F.softmax(policy_logits, dim=-1)
            
            # Opponent policy analysis with different temperatures
            opp_probs_raw = F.softmax(opponent_policy_logits, dim=-1)
            opp_probs_temp01 = F.softmax(opponent_policy_logits / 0.1, dim=-1)
            opp_probs_temp10 = F.softmax(opponent_policy_logits / 1.0, dim=-1)
            
            return {
                'agent_policy_probs': agent_probs,
                'opponent_policy_logits': opponent_policy_logits,
                'opponent_policy_probs_raw': opp_probs_raw,
                'opponent_policy_probs_temp_0.1': opp_probs_temp01,
                'opponent_policy_probs_temp_1.0': opp_probs_temp10,
                'value_estimate': value_estimate,
                'hidden_state_norm': torch.norm(new_hidden[0], dim=-1) if new_hidden else None
            }
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state for LSTM.
        
        Args:
            batch_size: Batch size
            device: Device to place tensors on
            
        Returns:
            Tuple of (h_0, c_0) hidden states
        """
        if device is None:
            device = next(self.parameters()).device
        
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        
        return h_0, c_0



class NetworkManager:
    """
    Utility class for managing network operations.
    """
    
    def __init__(self, network: GameLSTM, device: torch.device = None):
        """
        Initialize network manager.
        
        Args:
            network: The GameLSTM network to manage
            device: Device to place network on
        """
        self.network = network
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        
        self.logger = logging.getLogger(__name__)
    
    def predict(self, state_history: np.ndarray, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Make predictions for a given state history.
        
        Args:
            state_history: Numpy array of state history
            hidden: Optional hidden state
            
        Returns:
            Dictionary with predictions and updated hidden state
        """
        self.network.eval()
        
        with torch.no_grad():
            # Convert to tensor and add batch dimension if needed
            if isinstance(state_history, np.ndarray):
                x = torch.from_numpy(state_history).float().to(self.device)
            else:
                x = state_history.to(self.device)
            
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Add batch dimension
            if x.dim() == 2:
                x = x.unsqueeze(0)  # Add sequence dimension
            
            predictions = self.network.get_all_predictions(x, hidden)
            
            return {
                'policy_probs': predictions['policy_probs'].cpu().numpy(),
                'opponent_policy_logits': predictions['opponent_policy_logits'].cpu().numpy(),
                'value_estimate': predictions['value_estimate'].cpu().numpy(),
                'hidden': predictions['hidden']
            }
    
    def save_checkpoint(self, filepath: str, optimizer_state: Optional[Dict] = None, 
                       epoch: int = None, loss: float = None):
        """
        Save network checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            optimizer_state: Optional optimizer state dict
            epoch: Optional epoch number
            loss: Optional loss value
        """
        checkpoint = {
            'model_state_dict': self.network.state_dict(),
            'model_config': {
                'input_size': self.network.input_size,
                'hidden_size': self.network.hidden_size,
                'num_layers': self.network.num_layers,
                'num_actions': self.network.num_actions,
            }
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Load network checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint data
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint
    
    def get_network_summary(self) -> str:
        """Get a summary of the network architecture."""
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        summary = f"""
GameLSTM Network Summary:
- Input size: {self.network.input_size}
- Hidden size: {self.network.hidden_size}
- Number of layers: {self.network.num_layers}
- Number of actions: {self.network.num_actions}
- Total parameters: {total_params:,}
- Trainable parameters: {trainable_params:,}
- Device: {self.device}
        """
        
        return summary.strip()