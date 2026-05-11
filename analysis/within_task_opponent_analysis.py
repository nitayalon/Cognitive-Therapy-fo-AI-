"""
Within-Task Opponent Generalization Analysis
===========================================
For each training task, analyze generalization to different opponent ranges.
Creates 4×4 matrices (training opponent × test opponent) for each game.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Configuration
DATA_DIR = Path("experiments/generalization_matrix_834222")
OUTPUT_DIR = Path("experiments/results/within_task_opponent_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create subdirectories
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(exist_ok=True)
(OUTPUT_DIR / "data").mkdir(exist_ok=True)

# Opponent range ordering for consistent plotting
OPPONENT_ORDER = ['low', 'mid_low', 'mid_high', 'high']
GAME_NAMES = {
    'prisoners-dilemma': 'PD',
    'stag-hunt': 'SH', 
    'battle-of-sexes': 'BoS',
    'hawk-dove': 'HD'
}

# Payoff matrices for reward normalization
PAYOFF_MATRICES = {
    'prisoners-dilemma': {'R': 3, 'S': 0, 'T': 5, 'P': 1},
    'stag-hunt': {'R': 4, 'S': 0, 'T': 3, 'P': 2},
    'battle-of-sexes': {'R': 3, 'S': 0, 'T': 2, 'P': 1},
    'hawk-dove': {'R': 2, 'S': 1, 'T': 3, 'P': 0}
}

def calculate_reward_bounds(payoff_matrix):
    """Calculate min/max/range for reward normalization."""
    values = list(payoff_matrix.values())
    min_val = min(values)
    max_val = max(values)
    return {
        'min': min_val,
        'max': max_val,
        'range': max_val - min_val
    }

# Calculate reward bounds for each game
REWARD_BOUNDS = {
    game: calculate_reward_bounds(matrix)
    for game, matrix in PAYOFF_MATRICES.items()
}

def normalize_reward(reward, game):
    """Normalize reward to [0, 1] scale using game-specific bounds."""
    bounds = REWARD_BOUNDS[game]
    if bounds['range'] == 0:
        return 0
    normalized = (reward - bounds['min']) / bounds['range']
    return np.clip(normalized, 0, 1)

def extract_baseline_metrics(results_dict):
    """Extract baseline performance (same task, same opponent)."""
    training_cond = results_dict['training_condition']
    eval_results = results_dict['evaluation_results']
    
    # Find the baseline test condition (same game, same opponent)
    for test_cond, test_data in eval_results.items():
        if (test_cond['game'] == training_cond['game'] and 
            test_cond['opponent_range'] == training_cond['opponent_range']):
            
            # Extract metrics from opponent-level results
            opponent_results = test_data['results']
            rewards = []
            coop_rates = []
            mispredictions = []
            
            for opp_key, opp_data in opponent_results.items():
                rewards.append(opp_data['average_reward'])
                coop_rates.append(opp_data['cooperation_rate'])
                
                # Calculate misprediction
                agent_coop = opp_data['cooperation_rate']
                opponent_coop = opp_data['opponent_cooperation_rate']
                mispred = abs(agent_coop - opponent_coop)
                mispredictions.append(mispred)
            
            return {
                'raw_reward': np.mean(rewards),
                'normalized_reward': normalize_reward(np.mean(rewards), training_cond['game']),
                'cooperation': np.mean(coop_rates),
                'misprediction': np.mean(mispredictions),
                'raw_reward_std': np.std(rewards),
                'normalized_reward_std': np.std([normalize_reward(r, training_cond['game']) for r in rewards]),
                'cooperation_std': np.std(coop_rates),
                'misprediction_std': np.std(mispredictions)
            }
    
    return None

def extract_generalization_metrics(results_dict, test_opponent_range):
    """Extract generalization metrics for specific test opponent range (same game)."""
    training_cond = results_dict['training_condition']
    eval_results = results_dict['evaluation_results']
    
    # Find test condition: same game, different opponent
    for test_cond, test_data in eval_results.items():
        if (test_cond['game'] == training_cond['game'] and 
            test_cond['opponent_range'] == test_opponent_range):
            
            opponent_results = test_data['results']
            rewards = []
            coop_rates = []
            mispredictions = []
            
            for opp_key, opp_data in opponent_results.items():
                rewards.append(opp_data['average_reward'])
                coop_rates.append(opp_data['cooperation_rate'])
                
                agent_coop = opp_data['cooperation_rate']
                opponent_coop = opp_data['opponent_cooperation_rate']
                mispred = abs(agent_coop - opponent_coop)
                mispredictions.append(mispred)
            
            return {
                'raw_reward': np.mean(rewards),
                'normalized_reward': normalize_reward(np.mean(rewards), training_cond['game']),
                'cooperation': np.mean(coop_rates),
                'misprediction': np.mean(mispredictions),
                'raw_reward_std': np.std(rewards),
                'normalized_reward_std': np.std([normalize_reward(r, training_cond['game']) for r in rewards]),
                'cooperation_std': np.std(coop_rates),
                'misprediction_std': np.std(mispredictions)
            }
    
    return None

def load_all_results():
    """Load all 16 task results."""
    results = []
    
    for pkl_file in sorted(DATA_DIR.glob("*.pkl")):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
    
    print(f"Loaded {len(results)} task results")
    return results

def create_opponent_generalization_matrix(game_name):
    """
    Create 4×4 matrix showing generalization from training opponent to test opponent.
    Rows = training opponent range, Columns = test opponent range.
    """
    
    # Initialize matrices
    reward_shift_matrix = np.zeros((4, 4))  # Will store reward ratio (test/baseline)
    mispred_change_matrix = np.zeros((4, 4))  # Will store misprediction change
    
    # Load all results
    all_results = load_all_results()
    
    # Filter for specific game
    game_results = [r for r in all_results if r['training_condition']['game'] == game_name]
    
    print(f"\n{game_name} ({GAME_NAMES[game_name]}): Found {len(game_results)} training tasks")
    
    # Process each training task
    for results_dict in game_results:
        training_opp = results_dict['training_condition']['opponent_range']
        train_idx = OPPONENT_ORDER.index(training_opp)
        
        # Get baseline metrics
        baseline = extract_baseline_metrics(results_dict)
        if baseline is None:
            continue
        
        # Get generalization to each test opponent
        for test_opp in OPPONENT_ORDER:
            test_idx = OPPONENT_ORDER.index(test_opp)
            
            if test_opp == training_opp:
                # Diagonal: same opponent (baseline)
                reward_shift_matrix[train_idx, test_idx] = 1.0
                mispred_change_matrix[train_idx, test_idx] = 0.0
            else:
                # Off-diagonal: different opponent
                gen_metrics = extract_generalization_metrics(results_dict, test_opp)
                if gen_metrics is not None:
                    # Reward ratio
                    if baseline['normalized_reward'] > 0:
                        reward_ratio = gen_metrics['normalized_reward'] / baseline['normalized_reward']
                    else:
                        reward_ratio = 0
                    reward_shift_matrix[train_idx, test_idx] = reward_ratio
                    
                    # Misprediction change (absolute)
                    mispred_change = gen_metrics['misprediction'] - baseline['misprediction']
                    mispred_change_matrix[train_idx, test_idx] = mispred_change
    
    return reward_shift_matrix, mispred_change_matrix

def plot_generalization_matrices(game_name, reward_matrix, mispred_matrix):
    """Plot side-by-side heatmaps for reward shift and misprediction change."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    game_abbrev = GAME_NAMES[game_name]
    
    # Plot 1: Reward Shift (Ratio)
    ax1 = axes[0]
    sns.heatmap(reward_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='RdYlGn',
                center=1.0,
                vmin=0.5, 
                vmax=1.2,
                xticklabels=OPPONENT_ORDER,
                yticklabels=OPPONENT_ORDER,
                cbar_kws={'label': 'Reward Ratio (Test/Baseline)'},
                ax=ax1)
    ax1.set_xlabel('Test Opponent Range', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Opponent Range', fontsize=12, fontweight='bold')
    ax1.set_title(f'{game_abbrev}: Normalized Reward Retention\n(1.0 = baseline, <1.0 = worse, >1.0 = better)', 
                  fontsize=13, fontweight='bold')
    
    # Plot 2: Misprediction Change
    ax2 = axes[1]
    
    # Calculate symmetric vmin/vmax for diverging colormap
    max_abs = max(abs(mispred_matrix.min()), abs(mispred_matrix.max()))
    
    sns.heatmap(mispred_matrix,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn_r',  # Reversed: green for negative (improvement)
                center=0.0,
                vmin=-max_abs,
                vmax=max_abs,
                xticklabels=OPPONENT_ORDER,
                yticklabels=OPPONENT_ORDER,
                cbar_kws={'label': 'Misprediction Change'},
                ax=ax2)
    ax2.set_xlabel('Test Opponent Range', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Opponent Range', fontsize=12, fontweight='bold')
    ax2.set_title(f'{game_abbrev}: Behavior Prediction Change\n(0.0 = baseline, <0 = improved, >0 = worsened)',
                  fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    filename = OUTPUT_DIR / "figures" / f"{game_abbrev}_opponent_generalization_matrix.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def analyze_opponent_generalization_patterns():
    """Analyze which opponent transitions are easiest/hardest."""
    
    all_transitions = []
    
    for game_name in PAYOFF_MATRICES.keys():
        all_results = load_all_results()
        game_results = [r for r in all_results if r['training_condition']['game'] == game_name]
        
        for results_dict in game_results:
            training_opp = results_dict['training_condition']['opponent_range']
            baseline = extract_baseline_metrics(results_dict)
            
            if baseline is None:
                continue
            
            for test_opp in OPPONENT_ORDER:
                if test_opp == training_opp:
                    continue  # Skip baseline
                
                gen_metrics = extract_generalization_metrics(results_dict, test_opp)
                if gen_metrics is None:
                    continue
                
                # Calculate metrics
                reward_ratio = gen_metrics['normalized_reward'] / baseline['normalized_reward'] if baseline['normalized_reward'] > 0 else 0
                mispred_change = gen_metrics['misprediction'] - baseline['misprediction']
                
                all_transitions.append({
                    'game': game_name,
                    'game_abbrev': GAME_NAMES[game_name],
                    'train_opponent': training_opp,
                    'test_opponent': test_opp,
                    'baseline_reward': baseline['normalized_reward'],
                    'test_reward': gen_metrics['normalized_reward'],
                    'reward_ratio': reward_ratio,
                    'reward_pct_change': (reward_ratio - 1.0) * 100,
                    'baseline_mispred': baseline['misprediction'],
                    'test_mispred': gen_metrics['misprediction'],
                    'mispred_change': mispred_change
                })
    
    df = pd.DataFrame(all_transitions)
    
    # Save detailed data
    df.to_csv(OUTPUT_DIR / "tables" / "all_opponent_transitions.csv", index=False)
    print(f"\nSaved: all_opponent_transitions.csv ({len(df)} transitions)")
    
    # Aggregate by transition type
    transition_summary = df.groupby(['train_opponent', 'test_opponent']).agg({
        'reward_ratio': ['mean', 'std'],
        'reward_pct_change': ['mean', 'std'],
        'mispred_change': ['mean', 'std']
    }).round(4)
    
    transition_summary.to_csv(OUTPUT_DIR / "tables" / "opponent_transition_summary.csv")
    print(f"Saved: opponent_transition_summary.csv")
    
    # Find best/worst transitions
    print("\n" + "="*70)
    print("BEST OPPONENT GENERALIZATIONS (Highest Reward Retention)")
    print("="*70)
    best_transitions = df.nlargest(10, 'reward_ratio')
    for _, row in best_transitions.iterrows():
        print(f"{row['game_abbrev']}: {row['train_opponent']} → {row['test_opponent']}: "
              f"Reward={row['reward_ratio']:.3f}, Mispred Δ={row['mispred_change']:+.3f}")
    
    print("\n" + "="*70)
    print("WORST OPPONENT GENERALIZATIONS (Lowest Reward Retention)")
    print("="*70)
    worst_transitions = df.nsmallest(10, 'reward_ratio')
    for _, row in worst_transitions.iterrows():
        print(f"{row['game_abbrev']}: {row['train_opponent']} → {row['test_opponent']}: "
              f"Reward={row['reward_ratio']:.3f}, Mispred Δ={row['mispred_change']:+.3f}")
    
    return df

def create_aggregate_opponent_analysis():
    """Create aggregate analysis across all games."""
    
    all_data = []
    
    for game_name in PAYOFF_MATRICES.keys():
        all_results = load_all_results()
        game_results = [r for r in all_results if r['training_condition']['game'] == game_name]
        
        for results_dict in game_results:
            training_opp = results_dict['training_condition']['opponent_range']
            baseline = extract_baseline_metrics(results_dict)
            
            if baseline is None:
                continue
            
            # Get average generalization performance across all 3 new opponents
            gen_rewards = []
            gen_mispreds = []
            
            for test_opp in OPPONENT_ORDER:
                if test_opp == training_opp:
                    continue
                
                gen_metrics = extract_generalization_metrics(results_dict, test_opp)
                if gen_metrics is not None:
                    gen_rewards.append(gen_metrics['normalized_reward'])
                    gen_mispreds.append(gen_metrics['misprediction'])
            
            if gen_rewards:
                all_data.append({
                    'game': game_name,
                    'game_abbrev': GAME_NAMES[game_name],
                    'train_opponent': training_opp,
                    'baseline_reward': baseline['normalized_reward'],
                    'avg_gen_reward': np.mean(gen_rewards),
                    'avg_reward_ratio': np.mean(gen_rewards) / baseline['normalized_reward'] if baseline['normalized_reward'] > 0 else 0,
                    'baseline_mispred': baseline['misprediction'],
                    'avg_gen_mispred': np.mean(gen_mispreds),
                    'avg_mispred_change': np.mean(gen_mispreds) - baseline['misprediction']
                })
    
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_DIR / "tables" / "per_task_opponent_generalization.csv", index=False)
    print(f"\nSaved: per_task_opponent_generalization.csv")
    
    # Summary by training opponent range
    summary = df.groupby('train_opponent').agg({
        'avg_reward_ratio': ['mean', 'std'],
        'avg_mispred_change': ['mean', 'std']
    }).round(4)
    
    print("\n" + "="*70)
    print("OPPONENT GENERALIZATION BY TRAINING OPPONENT RANGE")
    print("="*70)
    print(summary)
    
    return df

def create_executive_summary(transition_df, task_df):
    """Create markdown executive summary."""
    
    summary_path = OUTPUT_DIR / "executive_summary.md"
    
    with open(summary_path, 'w') as f:
        f.write("# Within-Task Opponent Generalization Analysis\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Research Question\n")
        f.write("Within each training task, **which new opponent ranges are easiest vs hardest to generalize to?**\n\n")
        
        f.write("## Matrix Structure\n")
        f.write("For each game, we create a **4×4 matrix**:\n")
        f.write("- **Rows**: Training opponent range (low, mid_low, mid_high, high)\n")
        f.write("- **Columns**: Test opponent range (low, mid_low, mid_high, high)\n")
        f.write("- **Diagonal**: Baseline performance (same opponent)\n")
        f.write("- **Off-diagonal**: Generalization performance (new opponent)\n\n")
        
        f.write("## Metrics\n")
        f.write("1. **Reward Ratio**: test_reward / baseline_reward (1.0 = baseline, <1.0 = worse)\n")
        f.write("2. **Misprediction Change**: test_mispred - baseline_mispred (0.0 = baseline, >0 = worsened)\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Best transitions
        f.write("### Easiest Opponent Transitions (Top 5)\n")
        best = transition_df.nlargest(5, 'reward_ratio')
        for i, row in best.iterrows():
            f.write(f"{i+1}. **{row['game_abbrev']}: {row['train_opponent']} → {row['test_opponent']}**\n")
            f.write(f"   - Reward Retention: {row['reward_ratio']:.3f} ({row['reward_pct_change']:+.1f}%)\n")
            f.write(f"   - Misprediction Change: {row['mispred_change']:+.3f}\n\n")
        
        # Worst transitions
        f.write("### Hardest Opponent Transitions (Bottom 5)\n")
        worst = transition_df.nsmallest(5, 'reward_ratio')
        for i, row in worst.iterrows():
            f.write(f"{i+1}. **{row['game_abbrev']}: {row['train_opponent']} → {row['test_opponent']}**\n")
            f.write(f"   - Reward Retention: {row['reward_ratio']:.3f} ({row['reward_pct_change']:+.1f}%)\n")
            f.write(f"   - Misprediction Change: {row['mispred_change']:+.3f}\n\n")
        
        # Summary by training opponent
        f.write("### Average Generalization by Training Opponent Range\n")
        summary = task_df.groupby('train_opponent').agg({
            'avg_reward_ratio': 'mean',
            'avg_mispred_change': 'mean'
        }).round(3)
        
        for opp in OPPONENT_ORDER:
            if opp in summary.index:
                f.write(f"- **{opp}**: Avg Reward Ratio = {summary.loc[opp, 'avg_reward_ratio']:.3f}, "
                       f"Avg Mispred Change = {summary.loc[opp, 'avg_mispred_change']:+.3f}\n")
        
        f.write("\n## Files Generated\n")
        f.write("- **Figures**: 4 matrices (one per game) showing reward shift and misprediction change\n")
        f.write("- **Tables**: \n")
        f.write("  - `all_opponent_transitions.csv`: All 48 transitions (4 games × 4 train × 3 test)\n")
        f.write("  - `opponent_transition_summary.csv`: Aggregated by transition type\n")
        f.write("  - `per_task_opponent_generalization.csv`: Average across 3 new opponents per task\n")
    
    print(f"\nSaved: {summary_path}")

def main():
    """Main analysis pipeline."""
    
    print("="*70)
    print("WITHIN-TASK OPPONENT GENERALIZATION ANALYSIS")
    print("="*70)
    
    # Create matrices for each game
    print("\nGenerating 4×4 matrices for each game...")
    
    for game_name in PAYOFF_MATRICES.keys():
        print(f"\nProcessing {game_name}...")
        reward_matrix, mispred_matrix = create_opponent_generalization_matrix(game_name)
        plot_generalization_matrices(game_name, reward_matrix, mispred_matrix)
    
    # Analyze transition patterns
    print("\nAnalyzing opponent transition patterns...")
    transition_df = analyze_opponent_generalization_patterns()
    
    # Aggregate analysis
    print("\nCreating aggregate analysis...")
    task_df = create_aggregate_opponent_analysis()
    
    # Executive summary
    create_executive_summary(transition_df, task_df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"- Figures: {OUTPUT_DIR / 'figures'}")
    print(f"- Tables: {OUTPUT_DIR / 'tables'}")
    print(f"- Executive Summary: {OUTPUT_DIR / 'executive_summary.md'}")

if __name__ == "__main__":
    main()
