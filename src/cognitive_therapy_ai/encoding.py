"""
Observation encoding for discrete, one-hot game inputs.

This module implements the canonical encode_observation function used throughout
the codebase. The encoding is deliberately discrete and one-hot to enable exact
FSM extraction via L* and geometric methods.

CRITICAL: Do not duplicate encoding logic elsewhere. All encoding must go through
this module to ensure consistency between training, rollout, and extraction.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from enum import Enum

# Import Action from games module to avoid duplication
from .games import Action


class Outcome(Enum):
    """Possible outcomes of a game round."""
    CC = 0  # Both cooperate
    CD = 1  # Agent cooperates, opponent defects
    DC = 2  # Agent defects, opponent cooperates
    DD = 3  # Both defect


class ObservationEncoder:
    """
    Encodes game history into discrete one-hot vectors for FSM-extractable agents.
    
    The encoding is:
    - **Discrete**: finite alphabet of symbols
    - **One-hot**: each symbol represented as a one-hot vector
    - **History-based**: encodes (agent_last_action, opponent_last_action, outcome)
    - **Stationary**: no timestep information (required for FSM extraction)
    
    Input formats:
    1. **no_game (8D)**: agent_last(2) + opp_last(2) + outcome(4)
    2. **game_tag (11D)**: history(8) + game_one_hot(3)
    
    At t=0, the start token (all zeros) is used.
    """
    
    def __init__(
        self,
        input_condition: str = "no_game",
        include_timestep: bool = False,
        include_reward: bool = False
    ):
        """
        Initialize the observation encoder.
        
        Args:
            input_condition: "no_game" (8D) or "game_tag" (11D)
            include_timestep: If True, append timestep (breaks FSM extraction)
            include_reward: If True, append rewards (breaks discrete alphabet)
        """
        self.input_condition = input_condition
        self.include_timestep = include_timestep
        self.include_reward = include_reward
        
        # Base dimensions
        self.agent_action_dim = 2  # One-hot: [cooperate, defect]
        self.opp_action_dim = 2    # One-hot: [cooperate, defect]
        self.outcome_dim = 4       # One-hot: [CC, CD, DC, DD]
        self.history_dim = 8       # Total history encoding
        
        # Game tag dimension (only for game_tag condition)
        self.game_tag_dim = 3  # One-hot: [PD, SH, HD]
        
        # Calculate total input dimension
        self.input_dim = self.history_dim
        if self.input_condition == "game_tag":
            self.input_dim += self.game_tag_dim
        if self.include_timestep:
            self.input_dim += 1
        if self.include_reward:
            self.input_dim += 2  # agent_reward + opponent_reward
        
        # Alphabet enumeration for FSM extraction
        self.history_symbols = ["START", "CC", "CD", "DC", "DD"]
        
        if self.input_condition == "game_tag":
            # Full alphabet = history_symbols × games = 5 × 3 = 15
            self.games = ["PD", "SH", "HD"]
            self.full_alphabet = [
                f"{game}_{symbol}" 
                for game in self.games 
                for symbol in self.history_symbols
            ]
        else:
            # No-game condition: alphabet = history symbols only
            self.full_alphabet = self.history_symbols
    
    def encode_observation(
        self,
        agent_last_action: Optional[Action] = None,
        opponent_last_action: Optional[Action] = None,
        game_name: Optional[str] = None,
        timestep: int = 0,
        agent_reward: float = 0.0,
        opponent_reward: float = 0.0
    ) -> np.ndarray:
        """
        Encode a single observation as a discrete one-hot vector.
        
        Args:
            agent_last_action: Agent's previous action (None at t=0)
            opponent_last_action: Opponent's previous action (None at t=0)
            game_name: Game identifier "PD", "SH", or "HD" (required for game_tag)
            timestep: Current timestep (only used if include_timestep=True)
            agent_reward: Agent's last reward (only used if include_reward=True)
            opponent_reward: Opponent's last reward (only used if include_reward=True)
        
        Returns:
            One-hot encoded observation vector
        """
        # Start with all zeros (start token if no history)
        encoding = np.zeros(self.input_dim, dtype=np.float32)
        
        # If no previous actions, return start token (all zeros)
        if agent_last_action is None or opponent_last_action is None:
            return encoding
        
        # Encode agent's last action (one-hot)
        agent_idx = agent_last_action.value
        encoding[agent_idx] = 1.0
        
        # Encode opponent's last action (one-hot)
        opp_idx = self.agent_action_dim + opponent_last_action.value
        encoding[opp_idx] = 1.0
        
        # Encode outcome (one-hot)
        outcome = self._get_outcome(agent_last_action, opponent_last_action)
        outcome_idx = self.agent_action_dim + self.opp_action_dim + outcome.value
        encoding[outcome_idx] = 1.0
        
        # Add game tag if required
        if self.input_condition == "game_tag":
            if game_name is None:
                raise ValueError("game_name required for input_condition='game_tag'")
            game_idx = self._get_game_index(game_name)
            game_offset = self.history_dim
            encoding[game_offset + game_idx] = 1.0
        
        # Add timestep if required (breaks FSM extraction)
        if self.include_timestep:
            timestep_offset = self.history_dim
            if self.input_condition == "game_tag":
                timestep_offset += self.game_tag_dim
            # Normalize timestep to [0, 1] range (assuming T=100)
            encoding[timestep_offset] = timestep / 100.0
        
        # Add rewards if required (breaks discrete alphabet)
        if self.include_reward:
            reward_offset = self.history_dim
            if self.input_condition == "game_tag":
                reward_offset += self.game_tag_dim
            if self.include_timestep:
                reward_offset += 1
            encoding[reward_offset] = agent_reward
            encoding[reward_offset + 1] = opponent_reward
        
        return encoding
    
    def encode_start_token(self, game_name: Optional[str] = None) -> np.ndarray:
        """
        Encode the start token (t=0, no history).
        
        Args:
            game_name: Game identifier (required for game_tag condition)
        
        Returns:
            Start token encoding (all zeros except possibly game tag)
        """
        encoding = np.zeros(self.input_dim, dtype=np.float32)
        
        # Add game tag if required
        if self.input_condition == "game_tag":
            if game_name is None:
                raise ValueError("game_name required for input_condition='game_tag'")
            game_idx = self._get_game_index(game_name)
            game_offset = self.history_dim
            encoding[game_offset + game_idx] = 1.0
        
        return encoding
    
    def _get_outcome(self, agent_action: Action, opponent_action: Action) -> Outcome:
        """Determine the outcome from both actions."""
        if agent_action == Action.COOPERATE and opponent_action == Action.COOPERATE:
            return Outcome.CC
        elif agent_action == Action.COOPERATE and opponent_action == Action.DEFECT:
            return Outcome.CD
        elif agent_action == Action.DEFECT and opponent_action == Action.COOPERATE:
            return Outcome.DC
        else:  # Both defect
            return Outcome.DD
    
    def _get_game_index(self, game_name: str) -> int:
        """Get the index for a game name."""
        game_map = {"PD": 0, "SH": 1, "HD": 2}
        if game_name not in game_map:
            raise ValueError(f"Unknown game: {game_name}. Must be one of {list(game_map.keys())}")
        return game_map[game_name]
    
    def get_input_dim(self) -> int:
        """Get the total input dimension."""
        return self.input_dim
    
    def get_alphabet(self) -> list:
        """Get the discrete alphabet for FSM extraction."""
        return self.full_alphabet
    
    def is_one_hot_valid(self, encoding: np.ndarray) -> bool:
        """
        Validate that an encoding is a proper one-hot vector.
        
        Checks:
        1. Correct dimensionality
        2. History portion has exactly 3 active bits (agent_action + opp_action + outcome)
           OR is all zeros (start token)
        3. Game tag (if present) has exactly 1 active bit
        4. All values are 0 or 1
        
        Args:
            encoding: Encoded observation to validate
        
        Returns:
            True if valid, False otherwise
        """
        # Check dimensionality
        if len(encoding) != self.input_dim:
            return False
        
        # Check all values are 0 or 1
        if not np.all(np.isin(encoding, [0, 1])):
            return False
        
        # Check history portion (first 8 elements)
        history_part = encoding[:self.history_dim]
        history_sum = np.sum(history_part)
        
        # History should be all zeros (start token) or have exactly 3 active bits
        if history_sum not in [0, 3]:
            return False
        
        # If not start token, check structure:
        # - Exactly 1 bit in agent_action (positions 0-1)
        # - Exactly 1 bit in opp_action (positions 2-3)
        # - Exactly 1 bit in outcome (positions 4-7)
        if history_sum == 3:
            agent_sum = np.sum(history_part[0:2])
            opp_sum = np.sum(history_part[2:4])
            outcome_sum = np.sum(history_part[4:8])
            if not (agent_sum == 1 and opp_sum == 1 and outcome_sum == 1):
                return False
        
        # Check game tag if present
        if self.input_condition == "game_tag":
            game_part = encoding[self.history_dim:self.history_dim + self.game_tag_dim]
            if np.sum(game_part) != 1:
                return False
        
        return True
