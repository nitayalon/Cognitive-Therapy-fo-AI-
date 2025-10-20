"""
Opponent model implementation for mixed-motive games.

This module implements various opponent strategies, including the main
probabilistic opponent used in the cognitive therapy experiments.
"""

import numpy as np
from typing import List, Optional
from abc import ABC, abstractmethod
import random

from .games import Action, MixedMotiveGame


class OpponentStrategy(ABC):
    """Abstract base class for opponent strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.action_history = []
    
    @abstractmethod
    def choose_action(self, game_history: List[dict], round_number: int) -> Action:
        """
        Choose an action based on game history.
        
        Args:
            game_history: List of previous rounds with actions and payoffs
            round_number: Current round number (0-indexed)
            
        Returns:
            Action to take this round
        """
        pass
    
    def reset(self):
        """Reset the opponent's internal state."""
        self.action_history = []


class ProbabilisticOpponent(OpponentStrategy):
    """
    Probabilistic opponent with fixed defection probability.
    
    This is the main opponent type used in the cognitive therapy experiments.
    The opponent has a fixed probability 'p' of defecting on each round.
    """
    
    def __init__(self, defection_probability: float, name: Optional[str] = None):
        """
        Initialize probabilistic opponent.
        
        Args:
            defection_probability: Probability of defecting (0.0 to 1.0)
            name: Optional name for the opponent
        """
        if not 0.0 <= defection_probability <= 1.0:
            raise ValueError("Defection probability must be between 0.0 and 1.0")
        
        self.defection_probability = defection_probability
        if name is None:
            name = f"Probabilistic-p{defection_probability:.2f}"
        
        super().__init__(name)
    
    def choose_action(self, game_history: List[dict], round_number: int) -> Action:
        """
        Choose action based on defection probability.
        
        Args:
            game_history: List of previous rounds (not used by this strategy)
            round_number: Current round number (not used by this strategy)
            
        Returns:
            Action.DEFECT with probability p, Action.COOPERATE otherwise
        """
        action = Action.DEFECT if random.random() < self.defection_probability else Action.COOPERATE
        self.action_history.append(action)
        return action
    
    def get_type_parameter(self) -> float:
        """Get the defection probability parameter p."""
        return self.defection_probability


class TitForTatOpponent(OpponentStrategy):
    """
    Tit-for-Tat strategy: cooperate first, then copy opponent's last action.
    """
    
    def __init__(self):
        super().__init__("Tit-for-Tat")
    
    def choose_action(self, game_history: List[dict], round_number: int) -> Action:
        """
        Choose action based on tit-for-tat strategy.
        
        Args:
            game_history: List of previous rounds
            round_number: Current round number
            
        Returns:
            Action.COOPERATE on first round, then copy opponent's last action
        """
        if round_number == 0 or not game_history:
            action = Action.COOPERATE
        else:
            # Copy the player's last action
            last_round = game_history[-1]
            action = last_round['player_action']
        
        self.action_history.append(action)
        return action


class AlwaysCooperateOpponent(OpponentStrategy):
    """Always cooperate strategy."""
    
    def __init__(self):
        super().__init__("Always-Cooperate")
    
    def choose_action(self, game_history: List[dict], round_number: int) -> Action:
        """Always choose to cooperate."""
        action = Action.COOPERATE
        self.action_history.append(action)
        return action


class AlwaysDefectOpponent(OpponentStrategy):
    """Always defect strategy."""
    
    def __init__(self):
        super().__init__("Always-Defect")
    
    def choose_action(self, game_history: List[dict], round_number: int) -> Action:
        """Always choose to defect."""
        action = Action.DEFECT
        self.action_history.append(action)
        return action


class RandomOpponent(OpponentStrategy):
    """Random strategy with 50% probability of each action."""
    
    def __init__(self):
        super().__init__("Random")
    
    def choose_action(self, game_history: List[dict], round_number: int) -> Action:
        """Choose action randomly."""
        action = Action.DEFECT if random.random() < 0.5 else Action.COOPERATE
        self.action_history.append(action)
        return action


class Opponent:
    """
    Main opponent class that wraps strategy and provides additional functionality.
    
    This class is the primary interface for creating opponents in experiments.
    """
    
    def __init__(self, strategy: OpponentStrategy, opponent_id: Optional[str] = None):
        """
        Initialize opponent with a strategy.
        
        Args:
            strategy: The strategy this opponent will use
            opponent_id: Optional unique identifier for this opponent
        """
        self.strategy = strategy
        self.opponent_id = opponent_id or f"{strategy.name}-{id(self)}"
        self.games_played = 0
        self.total_payoff = 0.0
        self.action_counts = {Action.COOPERATE: 0, Action.DEFECT: 0}
    
    def play_action(self, game_history: List[dict], round_number: int) -> Action:
        """
        Choose and execute an action.
        
        Args:
            game_history: List of previous rounds
            round_number: Current round number
            
        Returns:
            Action chosen by the opponent's strategy
        """
        action = self.strategy.choose_action(game_history, round_number)
        self.action_counts[action] += 1
        return action
    
    def update_payoff(self, payoff: float):
        """Update the opponent's total payoff."""
        self.total_payoff += payoff
        self.games_played += 1
    
    def get_cooperation_rate(self) -> float:
        """Get the opponent's cooperation rate."""
        total_actions = sum(self.action_counts.values())
        if total_actions == 0:
            return 0.0
        return self.action_counts[Action.COOPERATE] / total_actions
    
    def get_defection_rate(self) -> float:
        """Get the opponent's defection rate."""
        return 1.0 - self.get_cooperation_rate()
    
    def get_average_payoff(self) -> float:
        """Get the opponent's average payoff per game."""
        if self.games_played == 0:
            return 0.0
        return self.total_payoff / self.games_played
    
    def reset(self):
        """Reset opponent state for a new experiment."""
        self.strategy.reset()
        self.games_played = 0
        self.total_payoff = 0.0
        self.action_counts = {Action.COOPERATE: 0, Action.DEFECT: 0}
    
    def get_strategy_name(self) -> str:
        """Get the name of the opponent's strategy."""
        return self.strategy.name
    
    def get_type_parameter(self) -> Optional[float]:
        """
        Get the type parameter if applicable (e.g., defection probability).
        
        Returns:
            Type parameter value or None if not applicable
        """
        if hasattr(self.strategy, 'get_type_parameter'):
            return self.strategy.get_type_parameter()
        return None


class OpponentFactory:
    """Factory class for creating opponent instances."""
    
    @staticmethod
    def create_probabilistic_opponent(defection_probability: float, 
                                    opponent_id: Optional[str] = None) -> Opponent:
        """
        Create a probabilistic opponent with given defection probability.
        
        Args:
            defection_probability: Probability of defecting (0.0 to 1.0)
            opponent_id: Optional unique identifier
            
        Returns:
            Opponent instance with probabilistic strategy
        """
        strategy = ProbabilisticOpponent(defection_probability)
        return Opponent(strategy, opponent_id)
    
    @staticmethod
    def create_opponent_set(defection_probabilities: List[float]) -> List[Opponent]:
        """
        Create a set of probabilistic opponents with different defection probabilities.
        
        Args:
            defection_probabilities: List of defection probabilities
            
        Returns:
            List of opponent instances
        """
        opponents = []
        for i, p in enumerate(defection_probabilities):
            opponent_id = f"prob_opponent_{i}_p{p:.2f}"
            opponents.append(OpponentFactory.create_probabilistic_opponent(p, opponent_id))
        return opponents
    
    @staticmethod
    def create_standard_opponents() -> List[Opponent]:
        """
        Create a standard set of opponents for testing.
        
        Returns:
            List of opponents with different strategies
        """
        strategies = [
            AlwaysCooperateOpponent(),
            AlwaysDefectOpponent(),
            RandomOpponent(),
            TitForTatOpponent(),
            ProbabilisticOpponent(0.25),
            ProbabilisticOpponent(0.75),
        ]
        
        return [Opponent(strategy) for strategy in strategies]