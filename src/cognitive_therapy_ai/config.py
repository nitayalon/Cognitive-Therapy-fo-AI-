"""
Configuration management for the cognitive therapy AI experiments.
"""

from dataclasses import dataclass
from typing import Dict, Any
import json
import os


@dataclass
class GameConfig:
    """Configuration for a specific game."""
    name: str
    payoff_matrix: Dict[str, Dict[str, float]]
    action_space: list
    
    
@dataclass 
class NetworkConfig:
    """Configuration for the LSTM network."""
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    input_size: int = 5  # Default for payoff matrix + round number representation
    

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    num_games_per_partner: int = 100  # T parameter
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 2000
    convergence_threshold: float = 1e-4
    patience: int = 100  # Early stopping patience
    
    # Loss function weights
    reward_loss_weight: float = 1.0
    action_prediction_loss_weight: float = 1.0
    type_prediction_loss_weight: float = 1.0
    

@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    opponent_defection_probs: list = None  # List of p values to test
    random_seed: int = 42
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.opponent_defection_probs is None:
            self.opponent_defection_probs = [0.1, 0.3, 0.5, 0.7, 0.9]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)
        

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to JSON file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)