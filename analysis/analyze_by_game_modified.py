#!/usr/bin/env python3
"""
Separate analysis for each game (PD, HD, SH) from modified architecture training.
Generates individual plots and statistics for each game across all opponent ranges.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Game and condition mapping
GAME_MAPPING = {
    'prisoners-dilemma': {
        'name': "Prisoner's Dilemma",
        'conditions': [0, 1, 2, 3, 4],
        'opponent_ranges': ['very_low', 'low', 'mid', 'high', 'very_high']
    },
    'hawk-dove': {
        'name': 'Hawk-Dove',
        'conditions': [5, 6, 7, 8, 9],
        'opponent_ranges': ['very_low', 'low', 'mid', 'high', 'very_high']
    },
    'stag-hunt': {
        'name': 'Stag-Hunt',
        'conditions': [10, 11, 12, 13, 14],
        'opponent_ranges': ['very_low', 'low', 'mid', 'high', 'very_high']
    }
}

OPPONENT_RANGE_LABELS = {
    'very_low': 'Very Low (0.0-0.2)',
    'low': 'Low (0.2-0.4)',
    'mid': 'Mid (0.4-0.6)',
    'high': 'High (0.6-0.8)',
    'very_high': 'Very High (0.8-1.0)'
}

# Colors for opponent ranges
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def load_game_data(base_dir: Path, game_key: str, num_seeds: int = 5) -> pd.DataFrame:
    """Load training data for a specific game."""
    conditions = GAME_MAPPING[game_key]['conditions']
    all_data = []
    
    print(f"\nLoading data for {GAME_MAPPING[game_key]['name']}...")
    
    for condition_id in conditions:
        for seed in range(num_seeds):
            task_dir = base_dir / f'condition_{condition_id}_seed_{seed}'
            
            # Find the actual experiment directory
            subdirs = [d for d in task_dir.iterdir() if d.is_dir()]
            if not subdirs:
                print(f"  Warning: No subdirectory in {task_dir}")
                continue
            
            exp_dir = subdirs[0]
            # Look in results folder, not logs
            metrics_file = exp_dir / 'results' / f'training_task_{condition_id}_metrics.csv'
            
            if not metrics_file.exists():
                print(f"  Warning: Missing {metrics_file}")
                continue
            
            # Load with selective columns for efficiency
            df = pd.read_csv(metrics_file)
            
            df['condition'] = condition_id
            df['seed'] = seed
            df['game'] = game_key
            df['opponent_range'] = GAME_MAPPING[game_key]['opponent_ranges'][conditions.index(condition_id)]
            
            all_data.append(df)
            print(f"  ✓ Loaded condition_{condition_id}_seed_{seed}: {len(df)} rows")
    
    if not all_data:
        raise ValueError(f"No data loaded for {game_key}")
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Total rows for {GAME_MAPPING[game_key]['name']}: {len(combined):,}")
    
    return combined


def plot_game_convergence(df: pd.DataFrame, game_key: str, output_dir: Path):
    """Plot training convergence for a single game across all opponent ranges."""
    game_name = GAME_MAPPING[game_key]['name']
    opponent_ranges = GAME_MAPPING[game_key]['opponent_ranges']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{game_name} - Training Convergence Across Opponent Ranges', 
                 fontsize=16, fontweight='bold')
    
    metrics = [
        ('epoch_average_cooperation_rate', 'Cooperation Probability', axes[0, 0]),
        ('total_loss', 'Total Loss', axes[0, 1]),
        ('rl_loss', 'RL Loss', axes[1, 0]),
        ('opponent_policy_loss', 'Opponent Policy Loss', axes[1, 1])
    ]
    
    for idx, opp_range in enumerate(opponent_ranges):
        data_subset = df[df['opponent_range'] == opp_range]
        color = COLORS[idx]
        label = OPPONENT_RANGE_LABELS[opp_range]
        
        for metric, title, ax in metrics:
            # Aggregate across seeds
            grouped = data_subset.groupby('epoch')[metric].agg(['mean', 'std']).reset_index()
            
            ax.plot(grouped['epoch'], grouped['mean'], 
                   label=label, color=color, linewidth=2, alpha=0.8)
            ax.fill_between(grouped['epoch'], 
                           grouped['mean'] - grouped['std'],
                           grouped['mean'] + grouped['std'],
                           color=color, alpha=0.2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    output_file = output_dir / f'{game_key.replace("-", "_")}_convergence_modified.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_game_cooperation_only(df: pd.DataFrame, game_key: str, output_dir: Path):
    """Plot cooperation probability only (like the attached image)."""
    game_name = GAME_MAPPING[game_key]['name']
    opponent_ranges = GAME_MAPPING[game_key]['opponent_ranges']
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_title(game_name, fontsize=14, fontweight='bold')
    
    for idx, opp_range in enumerate(opponent_ranges):
        data_subset = df[df['opponent_range'] == opp_range]
        color = COLORS[idx]
        label = OPPONENT_RANGE_LABELS[opp_range]
        
        # Aggregate across seeds
        grouped = data_subset.groupby('epoch')['epoch_average_cooperation_rate'].agg(['mean', 'std']).reset_index()
        
        ax.plot(grouped['epoch'], grouped['mean'], 
               label=label, color=color, linewidth=2, alpha=0.8)
        ax.fill_between(grouped['epoch'], 
                       grouped['mean'] - grouped['std'],
                       grouped['mean'] + grouped['std'],
                       color=color, alpha=0.2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cooperation Probability', fontsize=12)
    max_epoch = df['epoch'].max()
    ax.set_xlim(0, min(500, max_epoch))
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(title='Opponent Range', loc='best', fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / f'{game_key.replace("-", "_")}_cooperation_modified.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def compute_game_statistics(df: pd.DataFrame, game_key: str) -> pd.DataFrame:
    """Compute final statistics for a game."""
    game_name = GAME_MAPPING[game_key]['name']
    opponent_ranges = GAME_MAPPING[game_key]['opponent_ranges']
    
    # Get last 100 epochs (convergence period)
    max_epoch = df['epoch'].max()
    final_data = df[df['epoch'] >= max_epoch - 100]
    
    stats_list = []
    
    for opp_range in opponent_ranges:
        subset = final_data[final_data['opponent_range'] == opp_range]
        
        stats = {
            'game': game_name,
            'opponent_range': OPPONENT_RANGE_LABELS[opp_range],
            'cooperation_rate_mean': subset['epoch_average_cooperation_rate'].mean(),
            'cooperation_rate_std': subset['epoch_average_cooperation_rate'].std(),
            'total_loss_mean': subset['total_loss'].mean(),
            'total_loss_std': subset['total_loss'].std(),
            'rl_loss_mean': subset['rl_loss'].mean(),
            'rl_loss_std': subset['rl_loss'].std(),
            'opponent_policy_loss_mean': subset['opponent_policy_loss'].mean(),
            'opponent_policy_loss_std': subset['opponent_policy_loss'].std(),
            'cumulative_reward_mean': subset['epoch_cumulative_reward'].mean(),
            'cumulative_reward_std': subset['epoch_cumulative_reward'].std(),
            'num_samples': len(subset)
        }
        
        stats_list.append(stats)
    
    return pd.DataFrame(stats_list)


def main():
    """Main analysis function."""
    print("=" * 80)
    print("GAME-SPECIFIC ANALYSIS - Modified Architecture (6-element input)")
    print("=" * 80)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent / 'generalization_matrix_train_910969' / 'training'
    output_dir = Path(__file__).parent / 'output' / 'task_opponent_modified_by_game'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not base_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {base_dir}")
    
    print(f"\nInput directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    # Analyze each game separately
    all_stats = []
    
    for game_key in ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']:
        print("\n" + "=" * 80)
        print(f"Analyzing {GAME_MAPPING[game_key]['name']}")
        print("=" * 80)
        
        # Load data for this game
        df = load_game_data(base_dir, game_key)
        
        # Generate plots
        print(f"\nGenerating plots for {GAME_MAPPING[game_key]['name']}...")
        plot_game_convergence(df, game_key, output_dir)
        plot_game_cooperation_only(df, game_key, output_dir)
        
        # Compute statistics
        print(f"\nComputing statistics for {GAME_MAPPING[game_key]['name']}...")
        stats = compute_game_statistics(df, game_key)
        all_stats.append(stats)
        
        # Save game-specific data
        data_file = output_dir / f'{game_key.replace("-", "_")}_data_modified.csv'
        df.to_csv(data_file, index=False)
        print(f"  Saved: {data_file}")
        
        # Save game-specific statistics
        stats_file = output_dir / f'{game_key.replace("-", "_")}_statistics_modified.csv'
        stats.to_csv(stats_file, index=False)
        print(f"  Saved: {stats_file}")
        
        # Print statistics summary
        print(f"\nStatistics Summary for {GAME_MAPPING[game_key]['name']}:")
        print(stats.to_string(index=False))
    
    # Save combined statistics
    combined_stats = pd.concat(all_stats, ignore_index=True)
    combined_stats_file = output_dir / 'all_games_statistics_modified.csv'
    combined_stats.to_csv(combined_stats_file, index=False)
    print(f"\n✓ Saved combined statistics: {combined_stats_file}")
    
    # Create summary report
    summary_file = output_dir / 'analysis_summary_by_game.txt'
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GAME-SPECIFIC ANALYSIS SUMMARY\n")
        f.write("Modified Architecture (6-element input with opponent's previous action)\n")
        f.write("=" * 80 + "\n\n")
        
        for game_key in ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']:
            game_name = GAME_MAPPING[game_key]['name']
            game_stats = combined_stats[combined_stats['game'] == game_name]
            
            f.write(f"\n{game_name}\n")
            f.write("-" * 80 + "\n")
            f.write(game_stats.to_string(index=False))
            f.write("\n\n")
            
            # Key findings
            f.write(f"Key Findings for {game_name}:\n")
            f.write(f"  - Cooperation ranges from {game_stats['cooperation_rate_mean'].min():.3f} to {game_stats['cooperation_rate_mean'].max():.3f}\n")
            f.write(f"  - Cumulative reward ranges from {game_stats['cumulative_reward_mean'].min():.3f} to {game_stats['cumulative_reward_mean'].max():.3f}\n")
            f.write(f"  - RL loss: {game_stats['rl_loss_mean'].mean():.6f} ± {game_stats['rl_loss_std'].mean():.6f}\n")
            f.write(f"  - Opponent policy loss: {game_stats['opponent_policy_loss_mean'].mean():.6f} ± {game_stats['opponent_policy_loss_std'].mean():.6f}\n")
            f.write("\n")
    
    print(f"\n✓ Saved summary report: {summary_file}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - Individual game convergence plots (4-panel)")
    print("  - Individual game cooperation plots (single panel)")
    print("  - Individual game data CSVs")
    print("  - Individual game statistics CSVs")
    print("  - Combined statistics CSV")
    print("  - Summary report TXT")


if __name__ == '__main__':
    main()
