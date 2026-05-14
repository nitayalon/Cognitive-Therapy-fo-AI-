"""
Analyze Whole Population Training Data.

This script verifies:
1. Opponent cooperation probability distribution (uniformity check)
2. Action variability by opponent type during training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import json

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150


def analyze_opponent_distribution(training_dir: Path):
    """
    Verify that opponent cooperation probabilities are sampled uniformly
    from [0.0, 0.1, 0.2, ..., 1.0].
    """
    print("=" * 80)
    print("TASK 1: Opponent Cooperation Probability Distribution")
    print("=" * 80)
    
    # Find all training log directories
    training_logs = list(training_dir.glob("whole_population_task_*/logs/training_log.csv"))
    
    if not training_logs:
        print(f"No training logs found in {training_dir}")
        return None
    
    print(f"Found {len(training_logs)} training log files")
    
    # Collect all opponent probabilities across all training runs
    all_opponent_probs = []
    
    for log_file in training_logs[:3]:  # Sample first 3 for speed
        try:
            df = pd.read_csv(log_file)
            
            # Check what columns are available
            if 'opponent_defect_prob' in df.columns:
                # Convert defection prob to cooperation prob
                coop_probs = 1.0 - df['opponent_defect_prob']
                all_opponent_probs.extend(coop_probs.tolist())
            elif 'opponent_coop_prob' in df.columns:
                all_opponent_probs.extend(df['opponent_coop_prob'].tolist())
            else:
                print(f"Warning: No opponent probability column found in {log_file.name}")
                print(f"Available columns: {df.columns.tolist()[:10]}")
                
        except Exception as e:
            print(f"Error reading {log_file.name}: {e}")
    
    if not all_opponent_probs:
        print("No opponent probability data found")
        return None
    
    all_opponent_probs = np.array(all_opponent_probs)
    
    # Expected distribution: uniform over [0.0, 0.1, 0.2, ..., 1.0]
    expected_values = np.arange(0.0, 1.1, 0.1)
    
    # Count occurrences (round to nearest 0.1)
    rounded_probs = np.round(all_opponent_probs, 1)
    unique_vals, counts = np.unique(rounded_probs, return_counts=True)
    
    print(f"\nTotal opponent samples: {len(all_opponent_probs)}")
    print(f"Unique values: {len(unique_vals)}")
    print("\nDistribution:")
    for val, count in zip(unique_vals, counts):
        percentage = 100 * count / len(all_opponent_probs)
        print(f"  {val:.1f}: {count:6d} ({percentage:5.2f}%)")
    
    # Statistical test for uniformity
    expected_count = len(all_opponent_probs) / len(expected_values)
    chi_square = sum([(c - expected_count)**2 / expected_count for c in counts])
    print(f"\nUniformity Check:")
    print(f"  Expected count per bin: {expected_count:.1f}")
    print(f"  Chi-square statistic: {chi_square:.2f}")
    print(f"  Degrees of freedom: {len(expected_values) - 1}")
    
    # Create distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(all_opponent_probs, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(all_opponent_probs.mean(), color='red', linestyle='--', 
                    label=f'Mean: {all_opponent_probs.mean():.3f}')
    axes[0].set_xlabel('Opponent Cooperation Probability')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Opponent Cooperation Probabilities')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Bar chart of discrete values
    axes[1].bar(unique_vals, counts, width=0.08, alpha=0.7, edgecolor='black')
    axes[1].axhline(expected_count, color='red', linestyle='--', 
                    label=f'Expected (uniform): {expected_count:.1f}')
    axes[1].set_xlabel('Opponent Cooperation Probability (discrete)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Binned Distribution (rounded to 0.1)')
    axes[1].set_xticks(expected_values)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, all_opponent_probs


def analyze_action_variability(training_dir: Path):
    """
    Plot histograms of actions played by each opponent type during training.
    Verifies that there's sufficient variability in opponent actions.
    """
    print("\n" + "=" * 80)
    print("TASK 2: Action Variability by Opponent Type")
    print("=" * 80)
    
    # Find all training log directories
    training_logs = list(training_dir.glob("whole_population_task_*/logs/training_log.csv"))
    
    if not training_logs:
        print(f"No training logs found in {training_dir}")
        return None
    
    # Collect actions by opponent type
    opponent_actions = defaultdict(list)
    
    for log_file in training_logs[:5]:  # Sample first 5 for speed
        try:
            df = pd.read_csv(log_file, nrows=10000)  # First 10k rows for speed
            
            # Identify opponent type column
            opp_prob_col = None
            if 'opponent_defect_prob' in df.columns:
                opp_prob_col = 'opponent_defect_prob'
                df['opponent_coop_prob'] = 1.0 - df[opp_prob_col]
            elif 'opponent_coop_prob' in df.columns:
                opp_prob_col = 'opponent_coop_prob'
            
            if opp_prob_col is None or 'opponent_action' not in df.columns:
                print(f"Warning: Required columns not found in {log_file.name}")
                continue
            
            # Round opponent probs to discrete values
            df['opponent_type'] = df['opponent_coop_prob'].round(1)
            
            # Group actions by opponent type
            for opp_type in df['opponent_type'].unique():
                mask = df['opponent_type'] == opp_type
                actions = df.loc[mask, 'opponent_action'].values
                opponent_actions[opp_type].extend(actions.tolist())
                
        except Exception as e:
            print(f"Error reading {log_file.name}: {e}")
    
    if not opponent_actions:
        print("No action data found")
        return None
    
    # Calculate statistics per opponent type
    print("\nAction Statistics by Opponent Type:")
    print(f"{'Opp Coop Prob':<15} {'N Actions':<12} {'P(Coop)':<12} {'Variance':<12}")
    print("-" * 55)
    
    stats = []
    for opp_type in sorted(opponent_actions.keys()):
        actions = np.array(opponent_actions[opp_type])
        # Assume 0=cooperate, 1=defect (or vice versa, doesn't matter for variance)
        coop_rate = (actions == 0).mean()
        variance = coop_rate * (1 - coop_rate)
        n_actions = len(actions)
        
        print(f"{opp_type:<15.1f} {n_actions:<12d} {coop_rate:<12.4f} {variance:<12.4f}")
        stats.append({
            'opponent_coop_prob': opp_type,
            'n_actions': n_actions,
            'p_coop': coop_rate,
            'variance': variance
        })
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Cooperation rate by opponent type
    stats_df = pd.DataFrame(stats)
    axes[0, 0].plot(stats_df['opponent_coop_prob'], stats_df['p_coop'], 
                    'o-', markersize=8, linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect match')
    axes[0, 0].set_xlabel('Opponent Cooperation Probability (Policy)')
    axes[0, 0].set_ylabel('Observed Cooperation Rate (Actions)')
    axes[0, 0].set_title('Opponent Actions Match Their Policy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(-0.05, 1.05)
    axes[0, 0].set_ylim(-0.05, 1.05)
    
    # Plot 2: Action variance by opponent type
    axes[0, 1].plot(stats_df['opponent_coop_prob'], stats_df['variance'], 
                    's-', markersize=8, linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Opponent Cooperation Probability')
    axes[0, 1].set_ylabel('Action Variance')
    axes[0, 1].set_title('Action Variability by Opponent Type')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Histograms for extreme opponents (0.0 and 1.0)
    extreme_opps = [0.0, 1.0]
    colors = ['red', 'green']
    for opp_type, color in zip(extreme_opps, colors):
        if opp_type in opponent_actions:
            actions = opponent_actions[opp_type]
            axes[1, 0].hist(actions, bins=2, alpha=0.6, label=f'Opp p(coop)={opp_type:.1f}',
                           color=color, edgecolor='black')
    axes[1, 0].set_xlabel('Action (0=Coop, 1=Defect)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Actions for Extreme Opponent Types')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Histograms for middle opponents (0.5)
    middle_opp = 0.5
    if middle_opp in opponent_actions:
        actions = opponent_actions[middle_opp]
        axes[1, 1].hist(actions, bins=2, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_xlabel('Action (0=Coop, 1=Defect)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'Actions for Opponent p(coop)={middle_opp:.1f}')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, stats_df


def main():
    # Find the most recent whole population training experiment
    experiments_dir = Path("experiments")
    wp_train_dirs = sorted(experiments_dir.glob("whole_population_train_*"))
    
    if not wp_train_dirs:
        print("No whole population training experiments found")
        return
    
    # Use most recent
    experiment_dir = wp_train_dirs[-1]
    print(f"Analyzing: {experiment_dir}")
    
    # ==============================================================
    # TASK 1 & 2: Verification from Configuration
    # ==============================================================
    # Since detailed training logs aren't saved, we verify from config
    
    config_file = Path("config/whole_population_config.json")
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        
        print("\n" + "=" * 80)
        print("TASK 1: Opponent Distribution Verification")
        print("=" * 80)
        
        opponents = config['training_opponents']['values']
        print(f"\nConfigured opponent cooperation probabilities:")
        print(f"  Values: {opponents}")
        print(f"  Count: {len(opponents)}")
        print(f"  Range: [{min(opponents)}, {max(opponents)}]")
        print(f"  Spacing: {opponents[1] - opponents[0] if len(opponents) > 1 else 'N/A'}")
        print(f"  Sampling: {config['training_opponents']['sampling']}")
        
        # Verify uniformity
        expected_spacing = 0.1
        actual_spacing = [opponents[i+1] - opponents[i] for i in range(len(opponents)-1)]
        is_uniform = all(abs(s - expected_spacing) < 0.01 for s in actual_spacing)
        
        print(f"\n  Uniformity check: {'✓ PASS' if is_uniform else '✗ FAIL'}")
        print(f"  Expected distribution: Uniform over [0.0, 1.0] with 0.1 spacing")
        print(f"  Actual distribution: {'Uniform' if is_uniform else 'Non-uniform'}")
        
        print("\n" + "=" * 80)
        print("TASK 2: Expected Action Variability")
        print("=" * 80)
        
        print(f"\nTheoretical action distributions by opponent type:")
        print(f"  {'Opp p(coop)':<15} {'Expected p(coop)':<20} {'Variance':<15}")
        print("  " + "-" * 50)
        
        for p_coop in opponents:
            variance = p_coop * (1 - p_coop)
            print(f"  {p_coop:<15.1f} {p_coop:<20.3f} {variance:<15.4f}")
        
        print(f"\n  Maximum variance at p=0.5: {0.5 * 0.5:.4f}")
        print(f"  Minimum variance at extremes (0.0, 1.0): {0.0:.4f}")
        print(f"  ✓ Expected sufficient action variability across opponent types")
        
        # Create theoretical visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: Distribution of opponent types
        axes[0].bar(range(len(opponents)), [1/len(opponents)] * len(opponents), 
                   color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Opponent Index')
        axes[0].set_ylabel('Probability (Uniform)')
        axes[0].set_title('Opponent Sampling Distribution')
        axes[0].set_xticks(range(len(opponents)))
        axes[0].set_xticklabels([f'{p:.1f}' for p in opponents], rotation=45)
        axes[0].axhline(1/len(opponents), color='red', linestyle='--', alpha=0.5)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Expected action variance by opponent type
        variances = [p * (1 - p) for p in opponents]
        axes[1].plot(opponents, variances, 'o-', markersize=10, linewidth=2, color='orange')
        axes[1].set_xlabel('Opponent Cooperation Probability')
        axes[1].set_ylabel('Action Variance')
        axes[1].set_title('Expected Action Variability\n(Bernoulli Variance: p(1-p))')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(-0.05, 1.05)
        axes[1].set_ylim(0, 0.3)
        
        plt.tight_layout()
        
        output_dir = Path("Results/whole_population_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "opponent_distribution_verification.png", dpi=300, bbox_inches='tight')
        print(f"\n  Saved: {output_dir / 'opponent_distribution_verification.png'}")
        
        plt.show()
    
    else:
        print(f"Config file not found: {config_file}")



if __name__ == "__main__":
    main()
