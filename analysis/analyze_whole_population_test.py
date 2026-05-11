#!/usr/bin/env python3
"""
Whole Population Test Analysis
Analyzes test results from whole population paradigm (task-only or task-opponent).
Generates heatmaps showing generalization across games and opponent types.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Payoff matrices for reward normalization
PAYOFF_MATRICES = {
    'prisoners-dilemma': {'R': 3, 'S': 0, 'T': 5, 'P': 1},
    'stag-hunt': {'R': 4, 'S': 0, 'T': 3, 'P': 2},
    'hawk-dove': {'R': 2, 'S': 1, 'T': 3, 'P': 0}
}

GAME_DISPLAY_NAMES = {
    'prisoners-dilemma': "Prisoner's Dilemma",
    'hawk-dove': 'Hawk-Dove',
    'stag-hunt': 'Stag-Hunt'
}

def normalize_reward(reward: float, game: str) -> float:
    """Normalize reward to [0, 1] based on game payoff matrix."""
    if game not in PAYOFF_MATRICES:
        return 0.5
    
    payoffs = PAYOFF_MATRICES[game]
    all_payoffs = [payoffs['R'], payoffs['S'], payoffs['T'], payoffs['P']]
    min_r = min(all_payoffs)
    max_r = max(all_payoffs)
    rng = max_r - min_r
    
    if rng == 0:
        return 0.5
    
    norm = (reward - min_r) / rng
    return np.clip(norm, 0, 1)


def load_test_results(test_dir: Path) -> pd.DataFrame:
    """Load all test results from the whole population test directory."""
    print(f"\nLoading test results from {test_dir}...")
    
    all_results = []
    
    test_root = test_dir / 'testing'
    if not test_root.exists():
        raise FileNotFoundError(f"Testing directory not found: {test_root}")
    
    task_dirs = sorted([d for d in test_root.iterdir() if d.is_dir()])
    print(f"  Found {len(task_dirs)} test task directories")
    
    for task_dir in task_dirs:
        # Parse task directory name: whole_population_task_X_timestamp
        task_name = task_dir.name
        if not task_name.startswith('whole_population_task_'):
            continue
        
        try:
            task_id = int(task_name.split('_')[3])
        except (IndexError, ValueError):
            print(f"  Warning: Could not parse {task_name}")
            continue
        
        # Load config to get train/test info
        config_file = task_dir / 'experiment_config.json'
        if not config_file.exists():
            continue
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        train_game = config.get('train_game', 'unknown')
        test_game = config.get('test_game', 'unknown')
        
        # Test opponent is single value
        test_opp_probs = config.get('test_opponent_probabilities', [])
        if not test_opp_probs:
            continue
        test_opp = test_opp_probs[0]
        
        # Determine seed from task_id
        # Whole population: 3 games × 5 seeds = 15 models
        # Each model tested on 3 games × 5 opponents = 15 conditions
        # Total: 15 × 15 = 225 tasks
        model_id = task_id // 15  # Which of the 15 trained models
        test_cond = task_id % 15   # Which of the 15 test conditions
        
        # Extract seed from model_id (5 seeds per game)
        seed = model_id % 5
        
        # Load results
        results_dir = task_dir / 'results'
        if not results_dir.exists():
            continue
        
        # Find JSON report files (more reliable than pickle)
        report_files = list(results_dir.glob('eval_model_*_report.json'))
        if not report_files:
            continue
        
        for report_file in report_files:
            try:
                with open(report_file, 'r') as f:
                    test_data = json.load(f)
                
                # Extract metrics
                mean_reward = test_data.get('mean_reward', 0.0)
                cooperation_rate = test_data.get('mean_cooperation_rate', 0.0)
                
                # Verify games match
                actual_train_game = test_data.get('trained_game', train_game)
                actual_test_game = test_data.get('test_game', test_game)
                
                # Normalize reward
                norm_reward = normalize_reward(mean_reward, actual_test_game)
                
                all_results.append({
                    'task_id': task_id,
                    'model_id': model_id,
                    'seed': seed,
                    'train_game': actual_train_game,
                    'test_game': actual_test_game,
                    'test_opponent': test_opp,
                    'mean_reward': mean_reward,
                    'normalized_reward': norm_reward,
                    'cooperation_rate': cooperation_rate
                })
                
            except Exception as e:
                print(f"  Warning: Error loading {report_file}: {e}")
                continue
    
    if not all_results:
        raise ValueError("No test results found")
    
    df = pd.DataFrame(all_results)
    print(f"\n✓ Loaded {len(df)} test results")
    print(f"  Training games: {sorted(df['train_game'].unique())}")
    print(f"  Test games: {sorted(df['test_game'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")
    print(f"  Test opponents: {sorted(df['test_opponent'].unique())}")
    
    return df


def plot_generalization_heatmaps(df: pd.DataFrame, output_dir: Path):
    """
    Generate heatmaps showing generalization performance for each trained game.
    Each heatmap shows test games (rows) × test opponents (columns).
    """
    print("\nGenerating generalization heatmaps...")
    
    game_order = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    opp_order = sorted(df['test_opponent'].unique())
    
    for train_game in game_order:
        print(f"\n  Processing {GAME_DISPLAY_NAMES[train_game]}...")
        
        # Filter data for this training game
        train_data = df[df['train_game'] == train_game].copy()
        
        if len(train_data) == 0:
            print(f"    Warning: No data for {train_game}")
            continue
        
        # Aggregate across seeds
        agg_data = train_data.groupby(['test_game', 'test_opponent']).agg({
            'normalized_reward': ['mean', 'std'],
            'cooperation_rate': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        agg_data.columns = ['test_game', 'test_opponent', 
                           'reward_mean', 'reward_std',
                           'coop_mean', 'coop_std']
        
        # Create pivot tables
        reward_pivot = agg_data.pivot_table(
            index='test_game',
            columns='test_opponent',
            values='reward_mean',
            aggfunc='mean'
        )
        
        coop_pivot = agg_data.pivot_table(
            index='test_game',
            columns='test_opponent',
            values='coop_mean',
            aggfunc='mean'
        )
        
        # Reindex to ensure consistent ordering
        reward_pivot = reward_pivot.reindex(index=game_order, columns=opp_order)
        coop_pivot = coop_pivot.reindex(index=game_order, columns=opp_order)
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap 1: Normalized Reward
        sns.heatmap(
            reward_pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Mean Normalized Reward'},
            ax=axes[0],
            linewidths=0.5,
            center=0.5
        )
        axes[0].set_title(f'Trained on: {GAME_DISPLAY_NAMES[train_game]}\nNormalized Reward', 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Test Game', fontsize=12)
        axes[0].set_xlabel('Test Opponent Defection Probability', fontsize=12)
        axes[0].set_yticklabels([GAME_DISPLAY_NAMES[g] for g in game_order], rotation=0)
        
        # Heatmap 2: Cooperation Rate
        sns.heatmap(
            coop_pivot,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Cooperation Rate'},
            ax=axes[1],
            linewidths=0.5,
            center=0.5
        )
        axes[1].set_title(f'Trained on: {GAME_DISPLAY_NAMES[train_game]}\nCooperation Rate', 
                         fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Test Game', fontsize=12)
        axes[1].set_xlabel('Test Opponent Defection Probability', fontsize=12)
        axes[1].set_yticklabels([GAME_DISPLAY_NAMES[g] for g in game_order], rotation=0)
        
        plt.tight_layout()
        output_file = output_dir / f'{train_game.replace("-", "_")}_generalization_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved: {output_file}")
        plt.close()


def plot_combined_generalization_matrix(df: pd.DataFrame, output_dir: Path):
    """Create combined heatmap showing all training×test combinations."""
    print("\nGenerating combined generalization matrix...")
    
    game_order = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    opp_order = sorted(df['test_opponent'].unique())
    
    # Aggregate across seeds
    agg_df = df.groupby(['train_game', 'test_game', 'test_opponent']).agg({
        'normalized_reward': 'mean',
        'cooperation_rate': 'mean'
    }).reset_index()
    
    # Create test condition labels
    test_conditions = []
    for game in game_order:
        for opp in opp_order:
            test_conditions.append(f'{game}_{opp:.1f}')
    
    # Create matrix for each training game
    matrices = {}
    for train_game in game_order:
        train_data = agg_df[agg_df['train_game'] == train_game]
        matrix = np.zeros((len(game_order), len(opp_order)))
        
        for i, test_game in enumerate(game_order):
            for j, test_opp in enumerate(opp_order):
                subset = train_data[
                    (train_data['test_game'] == test_game) &
                    (train_data['test_opponent'] == test_opp)
                ]
                if len(subset) > 0:
                    matrix[i, j] = subset['normalized_reward'].values[0]
        
        matrices[train_game] = matrix
    
    # Create 3-row combined plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    
    for idx, train_game in enumerate(game_order):
        ax = axes[idx]
        sns.heatmap(
            matrices[train_game],
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Normalized Reward'},
            ax=ax,
            linewidths=0.5,
            center=0.5
        )
        
        ax.set_title(f'Trained on: {GAME_DISPLAY_NAMES[train_game]}', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Game', fontsize=10)
        if idx == 2:
            ax.set_xlabel('Test Opponent Defection Probability', fontsize=10)
        ax.set_yticklabels([GAME_DISPLAY_NAMES[g] for g in game_order], rotation=0)
        ax.set_xticklabels([f'{o:.1f}' for o in opp_order], rotation=0)
    
    plt.suptitle('Whole Population Generalization Matrix\n(Averaged across 5 seeds)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / 'combined_generalization_matrix.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def compute_summary_statistics(df: pd.DataFrame, output_dir: Path):
    """Compute and save summary statistics."""
    print("\nComputing summary statistics...")
    
    # Overall statistics by training game
    summary = df.groupby('train_game').agg({
        'normalized_reward': ['mean', 'std', 'min', 'max'],
        'cooperation_rate': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    summary_file = output_dir / 'summary_statistics.csv'
    summary.to_csv(summary_file)
    print(f"  ✓ Saved: {summary_file}")
    
    # Detailed statistics by train_game × test_game
    detailed = df.groupby(['train_game', 'test_game']).agg({
        'normalized_reward': ['mean', 'std'],
        'cooperation_rate': ['mean', 'std']
    }).round(4)
    
    detailed_file = output_dir / 'detailed_statistics.csv'
    detailed.to_csv(detailed_file)
    print(f"  ✓ Saved: {detailed_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(summary)
    
    return summary


def main():
    """Main analysis function."""
    print("=" * 80)
    print("WHOLE POPULATION TEST ANALYSIS")
    print("=" * 80)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Analyze whole population test results')
    parser.add_argument('--test-job-id', type=str, required=True,
                       help='Test job ID (e.g., 912631)')
    parser.add_argument('--train-job-id', type=str, default='911034',
                       help='Training job ID (default: 911034)')
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    test_dir = base_dir / f'whole_population_test_{args.test_job_id}'
    output_dir = Path(__file__).parent / 'output' / f'whole_population_test_{args.test_job_id}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTest directory: {test_dir}")
    print(f"Training job ID: {args.train_job_id}")
    print(f"Output directory: {output_dir}")
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Load test results
    df = load_test_results(test_dir)
    
    # Save raw data
    csv_file = output_dir / 'test_results_all.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n✓ Saved raw test data: {csv_file}")
    
    # Generate analyses
    plot_generalization_heatmaps(df, output_dir)
    plot_combined_generalization_matrix(df, output_dir)
    compute_summary_statistics(df, output_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - Per-game generalization heatmaps (reward + cooperation)")
    print("  - Combined 3-panel generalization matrix")
    print("  - Summary statistics CSV")
    print("  - Detailed statistics CSV")
    print("  - Raw test results CSV")


if __name__ == '__main__':
    main()
