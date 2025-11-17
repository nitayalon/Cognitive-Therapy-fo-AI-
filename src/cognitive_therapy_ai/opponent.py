"""
Opponent model implementation for mixed-motive games.

This module implements various opponent strategies, including the main
probabilistic opponent used in the cognitive therapy experiments.
"""

import numpy as np
from typing import List, Optional, Dict, Any
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
    def create_equally_spaced_opponents(
        defection_probabilities: List[float], 
        num_opponents: int = 11,
        include_boundaries: bool = True
    ) -> List[Opponent]:
        """
        Create equally spaced opponents across a defection probability range.
        
        If specific probabilities are given, uses the min and max to create equally
        spaced opponents across that range. This ensures training and testing use
        the same comprehensive opponent set rather than just the specified points.
        
        Args:
            defection_probabilities: List of defection probabilities to define range
            num_opponents: Number of equally spaced opponents to generate (default: 11)
            include_boundaries: Whether to include exact 0.0 and 1.0 probabilities
            
        Returns:
            List of equally spaced opponent instances
            
        Example:
            Input: [0.1, 0.3, 0.9] with num_opponents=11 
            Output: Opponents with defection probs [0.1, 0.18, 0.26, 0.34, 0.42, 0.5, 0.58, 0.66, 0.74, 0.82, 0.9]
        """
        if not defection_probabilities:
            raise ValueError("Must provide at least one defection probability")
        
        # Find the range from provided probabilities
        min_prob = min(defection_probabilities)
        max_prob = max(defection_probabilities)
        
        # Handle edge case where min and max are the same
        if min_prob == max_prob:
            return [OpponentFactory.create_probabilistic_opponent(min_prob, f"prob_opponent_fixed_p{min_prob:.3f}")]
        
        # Generate equally spaced probabilities
        equally_spaced_probs = np.linspace(min_prob, max_prob, num_opponents)
        
        # Optionally include exact boundaries if requested and not already included
        if include_boundaries:
            prob_set = set(equally_spaced_probs)
            if 0.0 not in prob_set and min_prob > 0.0:
                equally_spaced_probs = np.concatenate([[0.0], equally_spaced_probs])
            if 1.0 not in prob_set and max_prob < 1.0:
                equally_spaced_probs = np.concatenate([equally_spaced_probs, [1.0]])
        
        # Create opponent instances
        opponents = []
        for i, p in enumerate(equally_spaced_probs):
            # Ensure probability is within valid bounds
            p = max(0.0, min(1.0, float(p)))
            opponent_id = f"equally_spaced_{i}_p{p:.3f}"
            opponents.append(OpponentFactory.create_probabilistic_opponent(p, opponent_id))
        
        return opponents
    
    @staticmethod
    def create_segmented_experiment_configs(
        defection_probabilities: List[float],
        num_opponents_per_segment: int = 11,
        include_boundaries: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Create separate experiment configurations for each adjacent pair of defection probabilities.
        
        This creates multiple experiments where each experiment covers one segment of the
        probability range with equally spaced opponents within that segment.
        
        Args:
            defection_probabilities: Sorted list of defection probabilities defining segment boundaries
            num_opponents_per_segment: Number of equally spaced opponents per segment
            include_boundaries: Whether to include exact 0.0 and 1.0 probabilities
            
        Returns:
            List of experiment configurations, each containing opponents for one segment
            
        Example:
            Input: [0.1, 0.3, 0.9] with num_opponents_per_segment=5
            Output: 
                - Experiment 1: [0.1, 0.15, 0.2, 0.25, 0.3] (segment 0.1-0.3)
                - Experiment 2: [0.3, 0.45, 0.6, 0.75, 0.9] (segment 0.3-0.9)
        """
        if len(defection_probabilities) < 2:
            raise ValueError("Must provide at least two defection probabilities to create segments")
        
        # Sort probabilities to ensure proper segmentation
        sorted_probs = sorted(defection_probabilities)
        
        experiment_configs = []
        
        # Create experiment for each adjacent pair
        for i in range(len(sorted_probs) - 1):
            start_prob = sorted_probs[i]
            end_prob = sorted_probs[i + 1]
            
            # Generate equally spaced opponents for this segment
            segment_probs = np.linspace(start_prob, end_prob, num_opponents_per_segment)
            
            # Create opponents for this segment
            segment_opponents = []
            for j, p in enumerate(segment_probs):
                p = max(0.0, min(1.0, float(p)))
                opponent_id = f"segment_{i}_opponent_{j}_p{p:.3f}"
                segment_opponents.append(OpponentFactory.create_probabilistic_opponent(p, opponent_id))
            
            # Create experiment configuration
            experiment_config = {
                'experiment_id': f"segment_{i}_{start_prob:.2f}_to_{end_prob:.2f}",
                'segment_range': [start_prob, end_prob],
                'opponents': segment_opponents,
                'num_opponents': len(segment_opponents),
                'description': f"Experiment for defection probability range [{start_prob:.3f}, {end_prob:.3f}]"
            }
            
            experiment_configs.append(experiment_config)
        
        return experiment_configs
    
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