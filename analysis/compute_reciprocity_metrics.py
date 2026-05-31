"""
Compute game complexity metrics based on optimal policy patterns.

Metrics:
- Reciprocity: Correlation between opponent cooperation and optimal agent cooperation
- Mutual Information: How informative opponent's policy is about ego agent's optimal policy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150


def compute_reciprocity(optimal_policies):
    """
    Compute reciprocity metric: correlation between opponent cooperation 
    and optimal agent cooperation.
    
    Args:
        optimal_policies: Array of optimal cooperation probabilities for each opponent type
    
    Returns:
        Pearson correlation coefficient
    """
    opponent_coop_probs = np.arange(0.0, 1.1, 0.1)
    
    # Compute correlation
    if np.std(optimal_policies) == 0:
        # Constant policy - no reciprocity
        return 0.0
    
    corr, _ = pearsonr(opponent_coop_probs, optimal_policies)
    return corr


def compute_mutual_information(optimal_policies, num_bins=5):
    """
    Compute mutual information between opponent's policy and ego agent's optimal policy.
    
    Measures how much knowing the opponent's cooperation probability tells us
    about the ego agent's optimal cooperation probability.
    
    Higher MI = opponent's policy is more informative about optimal response.
    
    Args:
        optimal_policies: Array of optimal cooperation probabilities for each opponent type
        num_bins: Number of bins for discretizing continuous probabilities
    
    Returns:
        Mutual information in bits
    """
    opponent_coop_probs = np.arange(0.0, 1.1, 0.1)
    
    # Discretize both variables for MI computation
    # Bin opponent cooperation probabilities
    opp_bins = np.linspace(0, 1, num_bins + 1)
    opp_binned = np.digitize(opponent_coop_probs, opp_bins[:-1]) - 1
    opp_binned = np.clip(opp_binned, 0, num_bins - 1)
    
    # Bin agent optimal cooperation probabilities
    agent_bins = np.linspace(0, 1, num_bins + 1)
    agent_binned = np.digitize(optimal_policies, agent_bins[:-1]) - 1
    agent_binned = np.clip(agent_binned, 0, num_bins - 1)
    
    # Compute mutual information
    mi = mutual_info_score(opp_binned, agent_binned)
    
    return mi


def compute_transition_count(optimal_policies, threshold=0.1):
    """
    Count number of strategy transitions in optimal policy.
    
    A transition occurs when policy changes by more than threshold.
    
    Args:
        optimal_policies: Array of optimal cooperation probabilities
        threshold: Minimum change to count as transition
    
    Returns:
        Number of transitions
    """
    transitions = 0
    for i in range(1, len(optimal_policies)):
        if abs(optimal_policies[i] - optimal_policies[i-1]) > threshold:
            transitions += 1
    return transitions


def main():
    # Load optimal policies
    optimal_policies_file = Path('Results/optimal_policies/optimal_mixed_policies.csv')
    
    if not optimal_policies_file.exists():
        print("ERROR: Optimal policies file not found. Run compute_optimal_policies.py first.")
        return
    
    df_policies = pd.read_csv(optimal_policies_file, index_col=0)
    
    print("="*80)
    print("GAME COMPLEXITY METRICS FROM OPTIMAL POLICIES")
    print("="*80)
    print()
    
    # Compute metrics for each game
    results = {
        'game': [],
        'reciprocity': [],
        'mutual_information': [],
        'num_transitions': []
    }
    
    for game in df_policies.index:
        optimal_policies = df_policies.loc[game].values
        
        # Compute metrics
        reciprocity = compute_reciprocity(optimal_policies)
        mi = compute_mutual_information(optimal_policies)
        transitions = compute_transition_count(optimal_policies)
        
        results['game'].append(game)
        results['reciprocity'].append(reciprocity)
        results['mutual_information'].append(mi)
        results['num_transitions'].append(transitions)
        
        print(f"Game: {game}")
        print(f"  Reciprocity (correlation):     {reciprocity:7.4f}")
        print(f"  Mutual Information (bits):     {mi:7.4f}")
        print(f"  Number of Transitions:         {transitions}")
        print()
    
    df_results = pd.DataFrame(results)
    
    print("="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(df_results.to_string(index=False, float_format='%.4f'))
    print()
    
    # Create scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = {'PD': 'orange', 'HD': 'blue', 'SH': 'green'}
    
    for _, row in df_results.iterrows():
        game = row['game']
        ax.scatter(row['mutual_information'], row['reciprocity'], 
                  s=200, c=colors[game], 
                  edgecolors='black', linewidths=2, label=game, 
                  alpha=0.7, zorder=3)
        
        # Add game label
        ax.annotate(game, (row['mutual_information'], row['reciprocity']), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Mutual Information (bits)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reciprocity (Correlation with Opponent)', fontsize=12, fontweight='bold')
    ax.set_title('Game Complexity: Mutual Information vs. Reciprocity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    
    # Add horizontal line at reciprocity=0 (no reciprocity)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Set axis limits with some padding
    x_range = df_results['mutual_information'].max() - df_results['mutual_information'].min()
    y_range = df_results['reciprocity'].max() - df_results['reciprocity'].min()
    
    ax.set_xlim(-0.05, df_results['mutual_information'].max() * 1.2)
    ax.set_ylim(df_results['reciprocity'].min() - 0.1 * y_range - 0.1,
                df_results['reciprocity'].max() + 0.1 * y_range + 0.1)
    
    plt.tight_layout()
    
    # Save results
    output_dir = Path('Results/game_complexity')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_results.to_csv(output_dir / 'reciprocity_complexity_metrics.csv', index=False)
    fig.savefig(output_dir / 'reciprocity_complexity_scatter.png', dpi=300, bbox_inches='tight')
    
    print(f"Saved results to: {output_dir}")
    print(f"  - reciprocity_complexity_metrics.csv")
    print(f"  - reciprocity_complexity_scatter.png")
    
    plt.show()


if __name__ == '__main__':
    main()
