"""
Mixed-motive games implementation for cognitive therapy AI experiments.

This module implements three classic mixed-motive games:
- Hawk-Dove Game
- Prisoner's Dilemma
- Battle of the Sexes

Each game is implemented as a class with standard payoff matrices and game logic.
"""

import numpy as np
from typing import Tuple, Dict, List
from abc import ABC, abstractmethod
from enum import Enum


class Action(Enum):
    """Standard actions for mixed-motive games."""
    COOPERATE = 0
    DEFECT = 1


class MixedMotiveGame(ABC):
    """Abstract base class for mixed-motive games."""
    
    def __init__(self, name: str):
        self.name = name
        self.action_space = [Action.COOPERATE, Action.DEFECT]
        self.num_actions = len(self.action_space)
        self.reset()
    
    @abstractmethod
    def get_payoff_matrix(self) -> np.ndarray:
        """Return the payoff matrix for this game."""
        pass
    
    def play_round(
        self, 
        player_action: Action, 
        opponent_action: Action
    ) -> Tuple[float, float]:
        """
        Play one round of the game.
        
        Args:
            player_action: Action chosen by the player
            opponent_action: Action chosen by the opponent
            
        Returns:
            Tuple of (player_payoff, opponent_payoff)
        """
        payoff_matrix = self.get_payoff_matrix()
        player_payoff = payoff_matrix[player_action.value, opponent_action.value]
        opponent_payoff = payoff_matrix[opponent_action.value, player_action.value]
        
        # Update game history
        self.history.append({
            'player_action': player_action,
            'opponent_action': opponent_action,
            'player_payoff': player_payoff,
            'opponent_payoff': opponent_payoff
        })
        
        return player_payoff, opponent_payoff
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get a simplified state vector representation with only payoff matrix and round number.
        
        Returns:
            State vector with payoff matrix (4 elements) + round number (1 element) = 5 elements total
        """
        # Get the payoff matrix (2x2 for all mixed-motive games)
        payoff_matrix = self.get_payoff_matrix()
        
        # Flatten the payoff matrix to create base state vector
        # Matrix format: [[cooperate_vs_cooperate, cooperate_vs_defect],
        #                 [defect_vs_cooperate, defect_vs_defect]]
        state_vector = payoff_matrix.flatten().tolist()  # Length: 4
        
        # Add current round number (normalized)
        round_number = len(self.history)
        state_vector.append(round_number / 100.0)  # Normalize to reasonable range
        
        return np.array(state_vector, dtype=np.float32)
    
    def get_state_size(self) -> int:
        """
        Get the size of the state vector.
        
        Returns:
            Size of the state vector (payoff matrix + round number = 5 elements)
        """
        return 5  # 4 elements for payoff matrix + 1 for round number
    
    def reset(self):
        """Reset the game state."""
        self.history = []
    
    def get_name(self) -> str:
        """Get the name of the game."""
        return self.name
    
    def get_cumulative_payoff(self) -> float:
        """Get cumulative payoff for the player."""
        return sum(round_data['player_payoff'] for round_data in self.history)
    
    def get_cooperation_rate(self) -> float:
        """Get the player's cooperation rate."""
        if not self.history:
            return 0.0
        
        cooperations = sum(
            1 for round_data in self.history 
            if round_data['player_action'] == Action.COOPERATE
        )
        return cooperations / len(self.history)


class HawkDove(MixedMotiveGame):
    """
    Hawk-Dove Game implementation.
    
    In this game:
    - Dove vs Dove: Both get moderate payoff (share resource)
    - Hawk vs Dove: Hawk gets high payoff, Dove gets low payoff
    - Dove vs Hawk: Dove gets low payoff, Hawk gets high payoff  
    - Hawk vs Hawk: Both get very low payoff (costly conflict)
    
    Payoff Matrix (Cooperate=Dove, Defect=Hawk):
    """
    
    def __init__(self, resource_value: float = 6.0, cost_of_conflict: float = 10.0):
        """
        Initialize Hawk-Dove game.
        
        Args:
            resource_value: Value of the contested resource
            cost_of_conflict: Cost when both players are aggressive (Hawks)
        """
        super().__init__("Hawk-Dove")
        self.resource_value = resource_value
        self.cost_of_conflict = cost_of_conflict
    
    def get_payoff_matrix(self) -> np.ndarray:
        """
        Payoff matrix for Hawk-Dove game.
        
        Returns:
            2x2 numpy array where matrix[i][j] is payoff for action i against action j
        """
        V = self.resource_value
        C = self.cost_of_conflict
        
        # Rows: player actions (Cooperate=Dove, Defect=Hawk)
        # Cols: opponent actions (Cooperate=Dove, Defect=Hawk)
        return np.array([
            [V/2, 0],        # Dove vs [Dove, Hawk]
            [V, (V-C)/2]     # Hawk vs [Dove, Hawk]
        ], dtype=np.float32)


class PrisonersDilemma(MixedMotiveGame):
    """
    Prisoner's Dilemma implementation.
    
    Classic 2x2 game where mutual cooperation gives moderate payoff,
    but defection against cooperation gives highest individual payoff.
    """
    
    def __init__(self, T: float = 5.0, R: float = 3.0, P: float = 1.0, S: float = 0.0):
        """
        Initialize Prisoner's Dilemma.
        
        Args:
            T: Temptation payoff (defect while opponent cooperates)
            R: Reward payoff (mutual cooperation)
            P: Punishment payoff (mutual defection)
            S: Sucker's payoff (cooperate while opponent defects)
            
        Note: Standard ordering requires T > R > P > S and 2R > T + S
        """
        super().__init__("Prisoners-Dilemma")
        self.T, self.R, self.P, self.S = T, R, P, S
        
        # Validate payoff ordering
        assert T > R > P > S, "Invalid payoff ordering for Prisoner's Dilemma"
        assert 2*R > T + S, "Invalid payoff structure for Prisoner's Dilemma"
    
    def get_payoff_matrix(self) -> np.ndarray:
        """
        Payoff matrix for Prisoner's Dilemma.
        
        Returns:
            2x2 numpy array where matrix[i][j] is payoff for action i against action j
        """
        # Rows: player actions (Cooperate, Defect)
        # Cols: opponent actions (Cooperate, Defect)
        return np.array([
            [self.R, self.S],  # Cooperate vs [Cooperate, Defect]
            [self.T, self.P]   # Defect vs [Cooperate, Defect]
        ], dtype=np.float32)


class BattleOfSexes(MixedMotiveGame):
    """
    Battle of the Sexes implementation.
    
    Coordination game where players prefer to coordinate but disagree
    on which outcome to coordinate on.
    """
    
    def __init__(self, player_preferred: float = 3.0, player_dispreferred: float = 2.0, 
                 miscoordination: float = 0.0):
        """
        Initialize Battle of the Sexes.
        
        Args:
            player_preferred: Payoff when coordinating on player's preferred outcome
            player_dispreferred: Payoff when coordinating on opponent's preferred outcome  
            miscoordination: Payoff when failing to coordinate
        """
        super().__init__("Battle-of-Sexes")
        self.player_preferred = player_preferred
        self.player_dispreferred = player_dispreferred
        self.miscoordination = miscoordination
    
    def get_payoff_matrix(self) -> np.ndarray:
        """
        Payoff matrix for Battle of the Sexes.
        
        In this interpretation:
        - Cooperate = Player's preferred choice
        - Defect = Opponent's preferred choice
        
        Returns:
            2x2 numpy array where matrix[i][j] is payoff for action i against action j
        """
        # Rows: player actions (Player's pref, Opponent's pref)
        # Cols: opponent actions (Player's pref, Opponent's pref)
        return np.array([
            [self.player_preferred, self.miscoordination],      # Player's pref vs [Player's pref, Opponent's pref]
            [self.miscoordination, self.player_dispreferred]    # Opponent's pref vs [Player's pref, Opponent's pref]
        ], dtype=np.float32)


class StagHunt(MixedMotiveGame):
    """
    Stag Hunt Game implementation.
    
    Coordination game where players can either hunt stag (cooperate) or hunt hare (defect).
    Hunting stag requires both players to cooperate and yields the highest mutual payoff,
    but hunting hare is a safer individual strategy that doesn't depend on the other player.
    
    This game illustrates the tension between mutual benefit through cooperation
    and individual security through independent action.
    """
    
    def __init__(self, stag_payoff: float = 4.0, hare_payoff: float = 2.0, 
                 stag_failure: float = 0.0):
        """
        Initialize Stag Hunt game.
        
        Args:
            stag_payoff: Payoff when both players hunt stag (mutual cooperation)
            hare_payoff: Payoff when hunting hare (individual defection)
            stag_failure: Payoff when trying to hunt stag but partner hunts hare
        """
        super().__init__("Stag-Hunt")
        self.stag_payoff = stag_payoff
        self.hare_payoff = hare_payoff
        self.stag_failure = stag_failure
        
        # Validate payoff structure for Stag Hunt
        # Requirement: stag_payoff > hare_payoff > stag_failure
        if not (stag_payoff > hare_payoff > stag_failure):
            raise ValueError(
                f"Invalid payoff structure for Stag Hunt. "
                f"Required: stag_payoff ({stag_payoff}) > hare_payoff ({hare_payoff}) > "
                f"stag_failure ({stag_failure})"
            )
    
    def get_payoff_matrix(self) -> np.ndarray:
        """
        Payoff matrix for Stag Hunt.
        
        In this interpretation:
        - Cooperate = Hunt Stag (requires cooperation)
        - Defect = Hunt Hare (safe individual strategy)
        
        Returns:
            2x2 numpy array where matrix[i][j] is payoff for action i against action j
        """
        # Rows: player actions (Hunt Stag, Hunt Hare)
        # Cols: opponent actions (Hunt Stag, Hunt Hare)
        return np.array([
            [self.stag_payoff, self.stag_failure],    # Hunt Stag vs [Hunt Stag, Hunt Hare]
            [self.hare_payoff, self.hare_payoff]      # Hunt Hare vs [Hunt Stag, Hunt Hare]
        ], dtype=np.float32)


class GameFactory:
    """Factory class for creating game instances."""
    
    @staticmethod
    def create_game(game_name: str, **kwargs) -> MixedMotiveGame:
        """
        Create a game instance by name.
        
        Args:
            game_name: Name of the game to create
            **kwargs: Additional parameters for game initialization
            
        Returns:
            Game instance
        """
        game_classes = {
            'hawk-dove': HawkDove,
            'prisoners-dilemma': PrisonersDilemma,
            'battle-of-sexes': BattleOfSexes,
            'stag-hunt': StagHunt
        }
        
        if game_name.lower() not in game_classes:
            raise ValueError(f"Unknown game: {game_name}. Available: {list(game_classes.keys())}")
        
        return game_classes[game_name.lower()](**kwargs)
    
    @staticmethod
    def get_available_games() -> List[str]:
        """Get list of available game names."""
        return ['hawk-dove', 'prisoners-dilemma', 'battle-of-sexes', 'stag-hunt']