#!/usr/bin/env python3
"""
Plot globally normalized rewards across all test conditions for whole population agents.

This script creates plots showing normalized reward (y-axis) vs test conditions (x-axis)
with the same layout as the cooperation probability plots. Rewards are normalized
globally across all conditions to [0, 1] for comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'output' / 'whole_population_generalization' / 'data'
OUTPUT_DIR = BASE_DIR / 'output' / 'whole_population_generalization' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Game and opponent labels
GAME_NAMES = {
    'prisoners-dilemma': 'PD',
    'hawk-dove': 'HD',
    'stag-hunt': 'SH'
}

OPPONENT_LABELS = {
    0.1: 'VL',  # Very Low
    0.3: 'L',   # Low
    0.5: 'M',   # Mid
    0.7: 'H',   # High
    0.9: 'VH'   # Very High
}

GAME_COLORS = {
    'prisoners-dilemma': '#1f77b4',  # Blue
    'hawk-dove': '#ff7f0e',          # Orange
    'stag-hunt': '#2ca02c'           # Green
}


def load_per_agent_test_data():
    """Load the per-agent test data."""
    print(f"Loading per-agent test data from {DATA_DIR / 'test_data_per_agent.csv'}...")
    df = pd.read_csv(DATA_DIR / 'test_data_per_agent.csv')
    print(f"  Loaded {len(df)} test records")
    return df


def compute_global_normalization(df):
    """
    Compute global min and max rewards across all conditions for normalization.
    
    Returns:
        tuple: (min_reward, max_reward)
    """
    print("\nComputing global normalization bounds...")
    min_reward = df['mean_reward'].min()
    max_reward = df['mean_reward'].max()
    print(f"  Global reward range: [{min_reward:.4f}, {max_reward:.4f}]")
    return min_reward, max_reward


def normalize_rewards(df, min_reward, max_reward):
    """
    Normalize rewards globally to [0, 1].
    
    Args:
        df: DataFrame with mean_reward column
        min_reward: Global minimum reward
        max_reward: Global maximum reward
        
    Returns:
        DataFrame with normalized_reward column added
    """
    print("\nNormalizing rewards globally...")
    
    # Handle edge case where all rewards are the same
    if max_reward == min_reward:
        df['normalized_reward'] = 0.5
    else:
        df['normalized_reward'] = (df['mean_reward'] - min_reward) / (max_reward - min_reward)
    
    print(f"  Normalized reward range: [{df['normalized_reward'].min():.4f}, {df['normalized_reward'].max():.4f}]")
    return df


def aggregate_by_condition(df):
    """
    Aggregate normalized rewards across the 5 seeds for each test condition.
    
    Returns:
        DataFrame with columns: train_game, test_game, opponent_prob, 
                                normalized_reward_mean, normalized_reward_std
    """
    print("\nAggregating normalized rewards across seeds...")
    
    agg_df = df.groupby(['train_game', 'test_game', 'opponent_prob']).agg({
        'normalized_reward': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = ['train_game', 'test_game', 'opponent_prob', 
                      'normalized_reward_mean', 'normalized_reward_std', 'count']
    
    print(f"  Aggregated into {len(agg_df)} conditions")
    print(f"  Seeds per condition: {agg_df['count'].min()}-{agg_df['count'].max()}")
    
    return agg_df


def create_test_condition_labels(df):
    """
    Create test condition labels matching KLD plot format.
    
    Returns:
        DataFrame with 'condition_label' column added
    """
    df['condition_label'] = df.apply(
        lambda row: f"{GAME_NAMES[row['test_game']]},{OPPONENT_LABELS[row['opponent_prob']]}",
        axis=1
    )
    
    # Create ordering for x-axis (PD,VL to HD,VH)
    game_order = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    opp_order = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    df['game_idx'] = df['test_game'].map({g: i for i, g in enumerate(game_order)})
    df['opp_idx'] = df['opponent_prob'].map({o: i for i, o in enumerate(opp_order)})
    df['condition_idx'] = df['game_idx'] * 5 + df['opp_idx']
    
    df = df.sort_values('condition_idx')
    
    return df


def plot_normalized_reward_by_test_condition_faceted(df):
    """
    Create faceted plot with one subplot per trained game-agent.
    
    Args:
        df: DataFrame with aggregated normalized rewards and condition labels
    """
    print("\nGenerating normalized reward by test condition plot (faceted)...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Normalized Reward Across Test Conditions (Whole Population Agents)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    game_order = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    
    for idx, train_game in enumerate(game_order):
        ax = axes[idx]
        
        # Filter data for this trained agent
        game_df = df[df['train_game'] == train_game].copy()
        
        # Plot line with error band
        x = range(len(game_df))
        y_mean = game_df['normalized_reward_mean'].values
        y_std = game_df['normalized_reward_std'].values
        
        color = GAME_COLORS[train_game]
        
        ax.plot(x, y_mean, marker='o', linewidth=2, markersize=6, 
                color=color, label=f'{GAME_NAMES[train_game]} Agent')
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, 
                        alpha=0.2, color=color)
        
        # Reference line at 0.5
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, 
                   label='Mid-range')
        
        # Vertical lines separating test games
        ax.axvline(x=4.5, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax.axvline(x=9.5, color='black', linestyle='-', linewidth=1, alpha=0.3)
        
        # Styling
        ax.set_ylabel('Normalized Reward', fontsize=11, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # Add game region labels at top of first subplot
        if idx == 0:
            ax.text(2, 1.02, 'PD Test', ha='center', va='bottom', 
                   fontsize=9, color='gray')
            ax.text(7, 1.02, 'SH Test', ha='center', va='bottom', 
                   fontsize=9, color='gray')
            ax.text(12, 1.02, 'HD Test', ha='center', va='bottom', 
                   fontsize=9, color='gray')
    
    # X-axis labels on bottom subplot
    axes[-1].set_xlabel('Test Condition', fontsize=11, fontweight='bold')
    axes[-1].set_xticks(range(len(game_df)))
    axes[-1].set_xticklabels(game_df['condition_label'], rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'normalized_reward_by_test_condition_faceted.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


def plot_normalized_reward_by_test_condition_combined(df):
    """
    Create combined plot with all agents overlaid.
    
    Args:
        df: DataFrame with aggregated normalized rewards and condition labels
    """
    print("\nGenerating normalized reward by test condition plot (combined)...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    game_order = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    markers = ['o', 's', '^']
    
    for idx, train_game in enumerate(game_order):
        # Filter data for this trained agent
        game_df = df[df['train_game'] == train_game].copy()
        
        # Plot line
        x = range(len(game_df))
        y_mean = game_df['normalized_reward_mean'].values
        
        color = GAME_COLORS[train_game]
        marker = markers[idx]
        
        ax.plot(x, y_mean, marker=marker, linewidth=2, markersize=8, 
                color=color, label=f'{GAME_NAMES[train_game]} Agent',
                markeredgewidth=1.5, markeredgecolor='white')
    
    # Reference line at 0.5
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, 
               label='Mid-range')
    
    # Vertical lines separating test games
    ax.axvline(x=4.5, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.axvline(x=9.5, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    # Game region labels
    ax.text(2, 1.02, 'PD Test', ha='center', va='bottom', 
           fontsize=10, color='gray', fontweight='bold')
    ax.text(7, 1.02, 'SH Test', ha='center', va='bottom', 
           fontsize=10, color='gray', fontweight='bold')
    ax.text(12, 1.02, 'HD Test', ha='center', va='bottom', 
           fontsize=10, color='gray', fontweight='bold')
    
    # Styling
    ax.set_xlabel('Test Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Reward', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(range(15))
    
    # Use first agent's labels (all have same test conditions)
    sample_df = df[df['train_game'] == game_order[0]]
    ax.set_xticklabels(sample_df['condition_label'], rotation=45, ha='right')
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    plt.title('Normalized Reward Across Test Conditions (All Whole Population Agents)', 
             fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'normalized_reward_by_test_condition_combined.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


def main():
    """Main execution function."""
    print("=" * 60)
    print("NORMALIZED REWARD PLOTTER")
    print("=" * 60)
    
    # Load data
    df = load_per_agent_test_data()
    
    # Global normalization
    min_reward, max_reward = compute_global_normalization(df)
    df = normalize_rewards(df, min_reward, max_reward)
    
    # Aggregate across seeds
    agg_df = aggregate_by_condition(df)
    
    # Create condition labels and ordering
    agg_df = create_test_condition_labels(agg_df)
    
    # Generate plots
    plot_normalized_reward_by_test_condition_faceted(agg_df)
    plot_normalized_reward_by_test_condition_combined(agg_df)
    
    print("\n" + "=" * 60)
    print("PLOTS COMPLETE")
    print("=" * 60)
    print(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
