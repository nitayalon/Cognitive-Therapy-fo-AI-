"""
Analytic best response calculations for memoryless opponents.

Since opponents are memoryless (fixed P(cooperate) = p_coop, independent of history),
the best response in every (game, opponent) cell is a fixed action that can be
computed in closed form.

This module provides ground truth for:
1. Training sanity checks (does RL recover the optimal policy?)
2. Validating extracted FSMs (does the FSM match the best response?)

Expected qualitative patterns (to be empirically verified):
- PD: Best response is DEFECT for all p_coop (dominant strategy)
  → Optimal policy ignores opponent entirely
- SH and HD: Best response depends on p_coop
  → Optimal policy must condition on opponent behavior
"""

import numpy as np
from typing import Tuple
from enum import Enum

# Import Action from games module to avoid duplication
from .games import Action


def analytic_best_response(
    payoff_matrix: np.ndarray,
    p_coop: float
) -> Action:
    """
    Compute the closed-form best response against a memoryless opponent.
    
    Since the opponent plays Cooperate with probability p_coop (independent of
    all history), the expected payoff for each of the agent's actions can be
    computed directly:
    
    E[payoff | agent plays C] = p_coop * M[C,C] + (1-p_coop) * M[C,D]
    E[payoff | agent plays D] = p_coop * M[D,C] + (1-p_coop) * M[D,D]
    
    Best response = argmax over {C, D}.
    
    Args:
        payoff_matrix: 2x2 numpy array, payoff_matrix[i,j] = payoff for action i vs action j
                      where 0=Cooperate, 1=Defect
        p_coop: Opponent's cooperation probability (memoryless)
    
    Returns:
        Best response action (COOPERATE or DEFECT)
    """
    # Expected payoff for cooperating
    ev_cooperate = p_coop * payoff_matrix[0, 0] + (1 - p_coop) * payoff_matrix[0, 1]
    
    # Expected payoff for defecting
    ev_defect = p_coop * payoff_matrix[1, 0] + (1 - p_coop) * payoff_matrix[1, 1]
    
    # Return argmax
    if ev_cooperate > ev_defect:
        return Action.COOPERATE
    else:
        # Tie-breaking: prefer defect (conservative)
        return Action.DEFECT


def analytic_best_response_value(
    payoff_matrix: np.ndarray,
    p_coop: float
) -> float:
    """
    Compute the expected payoff of the best response.
    
    Args:
        payoff_matrix: 2x2 numpy array, payoff_matrix[i,j] = payoff for action i vs action j
        p_coop: Opponent's cooperation probability (memoryless)
    
    Returns:
        Expected payoff under the best response policy
    """
    # Expected payoff for cooperating
    ev_cooperate = p_coop * payoff_matrix[0, 0] + (1 - p_coop) * payoff_matrix[0, 1]
    
    # Expected payoff for defecting
    ev_defect = p_coop * payoff_matrix[1, 0] + (1 - p_coop) * payoff_matrix[1, 1]
    
    # Return the max
    return max(ev_cooperate, ev_defect)


def compute_expected_payoffs(
    payoff_matrix: np.ndarray,
    p_coop: float
) -> Tuple[float, float]:
    """
    Compute expected payoffs for both actions.
    
    Args:
        payoff_matrix: 2x2 numpy array
        p_coop: Opponent's cooperation probability
    
    Returns:
        Tuple of (ev_cooperate, ev_defect)
    """
    ev_cooperate = p_coop * payoff_matrix[0, 0] + (1 - p_coop) * payoff_matrix[0, 1]
    ev_defect = p_coop * payoff_matrix[1, 0] + (1 - p_coop) * payoff_matrix[1, 1]
    return ev_cooperate, ev_defect


def generate_best_response_table(
    payoff_matrix: np.ndarray,
    p_coop_values: np.ndarray
) -> dict:
    """
    Generate a table of best responses across opponent cooperation probabilities.
    
    Useful for visualizing how the best response changes with p_coop.
    
    Args:
        payoff_matrix: 2x2 numpy array
        p_coop_values: Array of cooperation probabilities to evaluate
    
    Returns:
        Dictionary with keys:
        - 'p_coop': Array of cooperation probabilities
        - 'best_response': Array of best response actions (as integers)
        - 'ev_cooperate': Array of expected values for cooperating
        - 'ev_defect': Array of expected values for defecting
        - 'ev_best': Array of expected values under best response
    """
    best_responses = []
    ev_cooperates = []
    ev_defects = []
    ev_bests = []
    
    for p in p_coop_values:
        br = analytic_best_response(payoff_matrix, p)
        ev_c, ev_d = compute_expected_payoffs(payoff_matrix, p)
        ev_best = max(ev_c, ev_d)
        
        best_responses.append(br.value)
        ev_cooperates.append(ev_c)
        ev_defects.append(ev_d)
        ev_bests.append(ev_best)
    
    return {
        'p_coop': p_coop_values,
        'best_response': np.array(best_responses),
        'ev_cooperate': np.array(ev_cooperates),
        'ev_defect': np.array(ev_defects),
        'ev_best': np.array(ev_bests)
    }


def verify_pd_dominance(payoff_matrix: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Verify that defection is the dominant strategy in a Prisoner's Dilemma.
    
    For PD, DEFECT should be the best response for all p_coop ∈ [0, 1].
    This is a testable property of the PD payoff matrix.
    
    Args:
        payoff_matrix: 2x2 payoff matrix to check
        tolerance: Numerical tolerance for comparisons
    
    Returns:
        True if defection is dominant, False otherwise
    """
    # Check best response at several p_coop values
    test_probs = np.linspace(0, 1, 21)  # 0.0, 0.05, 0.1, ..., 1.0
    
    for p in test_probs:
        br = analytic_best_response(payoff_matrix, p)
        if br != Action.DEFECT:
            return False
    
    return True


def verify_no_dominance(payoff_matrix: np.ndarray) -> bool:
    """
    Verify that neither action is dominant (best response depends on p_coop).
    
    This should be true for Stag Hunt and Hawk-Dove.
    
    Args:
        payoff_matrix: 2x2 payoff matrix to check
    
    Returns:
        True if best response varies with p_coop, False if one action dominates
    """
    # Check best response at several p_coop values
    test_probs = np.linspace(0, 1, 21)
    
    responses = [analytic_best_response(payoff_matrix, p) for p in test_probs]
    
    # If all responses are the same, one action is dominant
    if all(r == responses[0] for r in responses):
        return False
    
    # Otherwise, best response varies
    return True
