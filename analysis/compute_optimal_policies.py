"""
Compute optimal mixed policies for each game against different opponent types.

For each game and opponent cooperation probability, compute the optimal 
probability to cooperate that maximizes expected payoff.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from cognitive_therapy_ai.games import GameFactory


def compute_expected_payoff(p_agent, p_opp, payoff_matrix):
    """
    Compute expected payoff for agent using mixed strategy.
    
    Args:
        p_agent: Agent's probability to cooperate
        p_opp: Opponent's probability to cooperate
        payoff_matrix: 2x2 numpy array [[R, S], [T, P]]
            R: Both cooperate
            S: Agent cooperates, opponent defects
            T: Agent defects, opponent cooperates
            P: Both defect
    
    Returns:
        Expected payoff for the agent
    """
    R = payoff_matrix[0, 0]  # Both cooperate
    S = payoff_matrix[0, 1]  # Agent coop, opponent defect
    T = payoff_matrix[1, 0]  # Agent defect, opponent coop
    P = payoff_matrix[1, 1]  # Both defect
    
    # Expected payoff = sum over all joint action probabilities
    expected_payoff = (
        p_agent * p_opp * R +           # Both cooperate
        p_agent * (1 - p_opp) * S +     # Agent coop, opp defect
        (1 - p_agent) * p_opp * T +     # Agent defect, opp coop
        (1 - p_agent) * (1 - p_opp) * P # Both defect
    )
    
    return expected_payoff


def find_optimal_policy(p_opp, payoff_matrix, resolution=1000):
    """
    Find optimal cooperation probability against given opponent.
    
    Args:
        p_opp: Opponent's cooperation probability
        payoff_matrix: 2x2 payoff matrix
        resolution: Number of points to search over [0, 1]
    
    Returns:
        Optimal p_agent (cooperation probability)
    """
    # Search over possible agent cooperation probabilities
    p_agent_values = np.linspace(0, 1, resolution)
    payoffs = [compute_expected_payoff(p, p_opp, payoff_matrix) for p in p_agent_values]
    
    # Find p_agent that maximizes payoff
    optimal_idx = np.argmax(payoffs)
    optimal_p_agent = p_agent_values[optimal_idx]
    
    return optimal_p_agent


def main():
    # Configuration
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    game_display_names = {
        'prisoners-dilemma': 'PD',
        'hawk-dove': 'HD',
        'stag-hunt': 'SH'
    }
    
    # Opponent cooperation probabilities (bins of 0.1)
    opponent_coop_probs = np.arange(0.0, 1.1, 0.1)
    
    # Storage for results
    results = {}
    
    print("="*80)
    print("OPTIMAL MIXED POLICIES FOR EACH GAME")
    print("="*80)
    print()
    
    for game_name in games:
        game = GameFactory.create_game(game_name)
        payoff_matrix = game.get_payoff_matrix()
        
        game_display = game_display_names[game_name]
        print(f"Game: {game_display} ({game_name})")
        print(f"Payoff Matrix:")
        print(f"           Opp Coop  Opp Defect")
        print(f"  Ag Coop:  {payoff_matrix[0,0]:6.2f}    {payoff_matrix[0,1]:6.2f}")
        print(f"  Ag Defect:{payoff_matrix[1,0]:6.2f}    {payoff_matrix[1,1]:6.2f}")
        print()
        
        # Compute optimal policies for each opponent type
        optimal_policies = []
        
        for p_opp_coop in opponent_coop_probs:
            # Note: opponent cooperation = 1 - defection probability
            optimal_p_coop = find_optimal_policy(p_opp_coop, payoff_matrix)
            optimal_policies.append(optimal_p_coop)
        
        results[game_display] = optimal_policies
        print(f"{game_display} - Optimal cooperation probabilities:")
        print("-" * 60)
        print("Opp Coop Prob | Optimal Agent Coop Prob")
        print("-" * 60)
        for p_opp, p_opt in zip(opponent_coop_probs, optimal_policies):
            print(f"    {p_opp:.1f}       |        {p_opt:.4f}")
        print()
        print()
    
    # Create DataFrame (rows = games, columns = opponent coop probs)
    df = pd.DataFrame(results).T
    df.columns = [f'opp_coop_{p:.1f}' for p in opponent_coop_probs]
    
    print("="*80)
    print("OPTIMAL POLICIES TABLE (Games × Opponent Cooperation Probabilities)")
    print("="*80)
    print()
    print(df.to_string(float_format='%.4f'))
    print()
    
    # Save to CSV
    output_dir = Path('Results/optimal_policies')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'optimal_mixed_policies.csv'
    df.to_csv(output_file)
    
    print(f"Saved to: {output_file}")
    print()


if __name__ == '__main__':
    main()
