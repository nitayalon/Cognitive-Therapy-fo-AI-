#!/usr/bin/env python3
"""
Test Heatmap and KLD Analysis for Modified Architecture (6-element input)
Generates generalization heatmaps and KLD analysis for test results.
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

# Game and condition mapping
CONDITION_TO_GAME = {
    0: 'prisoners-dilemma', 1: 'prisoners-dilemma', 2: 'prisoners-dilemma', 3: 'prisoners-dilemma', 4: 'prisoners-dilemma',
    5: 'hawk-dove', 6: 'hawk-dove', 7: 'hawk-dove', 8: 'hawk-dove', 9: 'hawk-dove',
    10: 'stag-hunt', 11: 'stag-hunt', 12: 'stag-hunt', 13: 'stag-hunt', 14: 'stag-hunt'
}

CONDITION_TO_OPP_RANGE = {
    0: 'very_low', 1: 'low', 2: 'mid', 3: 'high', 4: 'very_high',
    5: 'very_low', 6: 'low', 7: 'mid', 8: 'high', 9: 'very_high',
    10: 'very_low', 11: 'low', 12: 'mid', 13: 'high', 14: 'very_high'
}

OPP_RANGE_CENTERS = {
    'very_low': 0.1, 'low': 0.3, 'mid': 0.5, 'high': 0.7, 'very_high': 0.9
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
    """Load all test results from the test directory."""
    print(f"\nLoading test results from {test_dir}...")
    
    all_results = []
    
    # Check for MASTER_TEST_REGISTRY.csv
    registry_file = test_dir / 'MASTER_TEST_REGISTRY.csv'
    if registry_file.exists():
        print(f"  Found registry: {registry_file}")
        registry = pd.read_csv(registry_file)
        print(f"  Registry contains {len(registry)} test tasks")
    
    # Load individual test results
    test_root = test_dir / 'testing'
    if not test_root.exists():
        raise FileNotFoundError(f"Testing directory not found: {test_root}")
    
    for condition_dir in sorted(test_root.iterdir()):
        if not condition_dir.is_dir():
            continue
        
        # Parse condition directory name: test_condition_X_seed_Y
        parts = condition_dir.name.split('_')
        if len(parts) < 5 or parts[0] != 'test':
            continue
        
        try:
            train_condition = int(parts[2])
            seed = int(parts[4])
        except (IndexError, ValueError):
            print(f"  Warning: Could not parse {condition_dir.name}")
            continue
        
        # Find test subdirectories
        for test_subdir in condition_dir.iterdir():
            if not test_subdir.is_dir():
                continue
            
            # Load test results
            results_file = test_subdir / 'results' / 'test_results.pkl'
            if not results_file.exists():
                continue
            
            try:
                with open(results_file, 'rb') as f:
                    test_data = pickle.load(f)
                
                # Extract test condition info
                test_game = test_data.get('test_game', 'unknown')
                test_opponent_range = test_data.get('test_opponent_range', 'unknown')
                mean_reward = test_data.get('mean_reward', 0.0)
                cooperation_rate = test_data.get('final_cooperation_rate', 0.0)
                
                # Normalize reward
                norm_reward = normalize_reward(mean_reward, test_game)
                
                all_results.append({
                    'train_condition': train_condition,
                    'train_game': CONDITION_TO_GAME[train_condition],
                    'train_opp_range': CONDITION_TO_OPP_RANGE[train_condition],
                    'seed': seed,
                    'test_game': test_game,
                    'test_opp_range': test_opponent_range,
                    'mean_reward': mean_reward,
                    'normalized_reward': norm_reward,
                    'cooperation_rate': cooperation_rate
                })
                
            except Exception as e:
                print(f"  Warning: Error loading {results_file}: {e}")
                continue
    
    if not all_results:
        raise ValueError("No test results found")
    
    df = pd.DataFrame(all_results)
    print(f"\n✓ Loaded {len(df)} test results")
    print(f"  Training conditions: {sorted(df['train_condition'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")
    print(f"  Test games: {sorted(df['test_game'].unique())}")
    
    return df


def plot_generalization_heatmaps_by_training_game(df: pd.DataFrame, output_dir: Path):
    """
    Generate one heatmap per training game showing generalization performance.
    Each heatmap is 3 test games (rows) × 5 test opponent ranges (columns).
    """
    print("\nGenerating generalization heatmaps by training game...")
    
    game_order = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    opp_range_order = ['very_low', 'low', 'mid', 'high', 'very_high']
    
    for train_game in game_order:
        print(f"\n  Processing {GAME_DISPLAY_NAMES[train_game]}...")
        
        # Filter data for this training game (averaged across all opponent ranges)
        train_data = df[df['train_game'] == train_game].copy()
        
        if len(train_data) == 0:
            print(f"    Warning: No data for {train_game}")
            continue
        
        # Aggregate across training opponent ranges and seeds
        agg_data = train_data.groupby(['test_game', 'test_opp_range']).agg({
            'normalized_reward': 'mean',
            'cooperation_rate': 'mean'
        }).reset_index()
        
        # Create pivot tables
        reward_pivot = agg_data.pivot_table(
            index='test_game',
            columns='test_opp_range',
            values='normalized_reward',
            aggfunc='mean'
        )
        
        coop_pivot = agg_data.pivot_table(
            index='test_game',
            columns='test_opp_range',
            values='cooperation_rate',
            aggfunc='mean'
        )
        
        # Reindex to ensure consistent ordering
        reward_pivot = reward_pivot.reindex(index=game_order, columns=opp_range_order)
        coop_pivot = coop_pivot.reindex(index=game_order, columns=opp_range_order)
        
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
        axes[0].set_title(f'Generalization: {GAME_DISPLAY_NAMES[train_game]}\nNormalized Reward', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Test Game', fontsize=12)
        axes[0].set_xlabel('Test Opponent Range', fontsize=12)
        axes[0].set_yticklabels([GAME_DISPLAY_NAMES[g] for g in game_order], rotation=0)
        axes[0].set_xticklabels(['Very Low\n(0.0-0.2)', 'Low\n(0.2-0.4)', 'Mid\n(0.4-0.6)', 
                                  'High\n(0.6-0.8)', 'Very High\n(0.8-1.0)'], rotation=0)
        
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
        axes[1].set_title(f'Generalization: {GAME_DISPLAY_NAMES[train_game]}\nCooperation Rate', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Test Game', fontsize=12)
        axes[1].set_xlabel('Test Opponent Range', fontsize=12)
        axes[1].set_yticklabels([GAME_DISPLAY_NAMES[g] for g in game_order], rotation=0)
        axes[1].set_xticklabels(['Very Low\n(0.0-0.2)', 'Low\n(0.2-0.4)', 'Mid\n(0.4-0.6)', 
                                  'High\n(0.6-0.8)', 'Very High\n(0.8-1.0)'], rotation=0)
        
        plt.tight_layout()
        output_file = output_dir / f'{train_game.replace("-", "_")}_generalization_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved: {output_file}")
        plt.close()


def compute_kld(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Kullback-Leibler Divergence: KL(P||Q) = sum(p * log(p/q))
    
    Args:
        p: Target distribution (e.g., optimal policy)
        q: Predicted distribution (e.g., learned policy)
        epsilon: Small value to avoid log(0)
    
    Returns:
        KL divergence value
    """
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    
    # Normalize to ensure valid probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return np.sum(p * np.log(p / q))


def plot_kld_analysis(df: pd.DataFrame, output_dir: Path):
    """
    Plot KLD between learned policy and optimal policy.
    For each training condition, compute KLD on test conditions.
    """
    print("\nGenerating KLD analysis plots...")
    
    game_order = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    opp_range_order = ['very_low', 'low', 'mid', 'high', 'very_high']
    
    # For simplicity, use cooperation rate as proxy for policy
    # Optimal policy would be game-theory optimal for each game+opponent combination
    # This is a simplified version - full KLD would require actual policy distributions
    
    print("  Note: This is a simplified KLD analysis using cooperation rate as policy proxy")
    print("  Full KLD would require policy logits from the model")
    
    # Define heuristic optimal cooperation rates for each game-opponent combo
    optimal_coop = {
        ('prisoners-dilemma', 'very_low'): 0.0,  # Always defect in PD
        ('prisoners-dilemma', 'low'): 0.0,
        ('prisoners-dilemma', 'mid'): 0.0,
        ('prisoners-dilemma', 'high'): 0.0,
        ('prisoners-dilemma', 'very_high'): 0.0,
        ('hawk-dove', 'very_low'): 0.0,  # Defect against cooperative opponents
        ('hawk-dove', 'low'): 0.0,
        ('hawk-dove', 'mid'): 0.5,  # Mixed against balanced opponents
        ('hawk-dove', 'high'): 1.0,  # Cooperate against aggressive opponents
        ('hawk-dove', 'very_high'): 1.0,
        ('stag-hunt', 'very_low'): 1.0,  # Cooperate with cooperative opponents
        ('stag-hunt', 'low'): 1.0,
        ('stag-hunt', 'mid'): 0.5,  # Risky with balanced
        ('stag-hunt', 'high'): 0.0,  # Defect against defectors
        ('stag-hunt', 'very_high'): 0.0,
    }
    
    # Compute KLD for each training condition
    kld_data = []
    
    for train_cond in sorted(df['train_condition'].unique()):
        train_game = CONDITION_TO_GAME[train_cond]
        train_opp_range = CONDITION_TO_OPP_RANGE[train_cond]
        
        train_subset = df[df['train_condition'] == train_cond]
        
        for test_game in game_order:
            for test_opp_range in opp_range_order:
                test_subset = train_subset[
                    (train_subset['test_game'] == test_game) &
                    (train_subset['test_opp_range'] == test_opp_range)
                ]
                
                if len(test_subset) == 0:
                    continue
                
                # Get learned cooperation rate
                learned_coop = test_subset['cooperation_rate'].mean()
                
                # Get optimal cooperation rate
                optimal_key = (test_game, test_opp_range)
                optimal = optimal_coop.get(optimal_key, 0.5)
                
                # Compute simplified KLD (binary distribution)
                p = np.array([optimal, 1 - optimal])  # Optimal policy
                q = np.array([learned_coop, 1 - learned_coop])  # Learned policy
                
                kld = compute_kld(p, q)
                
                kld_data.append({
                    'train_game': train_game,
                    'train_opp_range': train_opp_range,
                    'test_game': test_game,
                    'test_opp_range': test_opp_range,
                    'kld': kld,
                    'learned_coop': learned_coop,
                    'optimal_coop': optimal
                })
    
    kld_df = pd.DataFrame(kld_data)
    
    # Save KLD data
    kld_csv = output_dir / 'kld_analysis_modified.csv'
    kld_df.to_csv(kld_csv, index=False)
    print(f"  ✓ Saved KLD data: {kld_csv}")
    
    # Plot KLD heatmap for each training game
    for train_game in game_order:
        train_data = kld_df[kld_df['train_game'] == train_game]
        
        if len(train_data) == 0:
            continue
        
        # Aggregate across training opponent ranges
        agg_kld = train_data.groupby(['test_game', 'test_opp_range'])['kld'].mean().reset_index()
        
        # Create pivot
        kld_pivot = agg_kld.pivot_table(
            index='test_game',
            columns='test_opp_range',
            values='kld',
            aggfunc='mean'
        )
        
        kld_pivot = kld_pivot.reindex(index=game_order, columns=opp_range_order)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            kld_pivot,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            cbar_kws={'label': 'KL Divergence'},
            ax=ax,
            linewidths=0.5
        )
        
        ax.set_title(f'Policy KLD: {GAME_DISPLAY_NAMES[train_game]}\n(Learned vs Optimal Policy)', 
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('Test Game', fontsize=12)
        ax.set_xlabel('Test Opponent Range', fontsize=12)
        ax.set_yticklabels([GAME_DISPLAY_NAMES[g] for g in game_order], rotation=0)
        ax.set_xticklabels(['Very Low', 'Low', 'Mid', 'High', 'Very High'], rotation=0)
        
        plt.tight_layout()
        output_file = output_dir / f'{train_game.replace("-", "_")}_kld_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()


def plot_combined_generalization_matrix(df: pd.DataFrame, output_dir: Path):
    """Create a single large heatmap showing all training×test combinations."""
    print("\nGenerating combined generalization matrix...")
    
    # Aggregate across seeds
    agg_df = df.groupby(['train_condition', 'test_game', 'test_opp_range']).agg({
        'normalized_reward': 'mean',
        'cooperation_rate': 'mean'
    }).reset_index()
    
    # Create matrix: 15 training conditions × 15 test conditions
    test_conditions = []
    for game in ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']:
        for opp in ['very_low', 'low', 'mid', 'high', 'very_high']:
            test_conditions.append(f'{game}_{opp}')
    
    matrix = np.zeros((15, 15))
    
    for _, row in agg_df.iterrows():
        train_idx = int(row['train_condition'])
        test_key = f"{row['test_game']}_{row['test_opp_range']}"
        if test_key in test_conditions:
            test_idx = test_conditions.index(test_key)
            matrix[train_idx, test_idx] = row['normalized_reward']
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Normalized Reward'},
        ax=ax,
        linewidths=0.5,
        center=0.5
    )
    
    # Labels
    train_labels = [f'{CONDITION_TO_GAME[i][:2].upper()}-{CONDITION_TO_OPP_RANGE[i][:2].upper()}' 
                    for i in range(15)]
    test_labels = [f'{tc.split("_")[0][:2].upper()}-{tc.split("_")[1][:2].upper()}' 
                   for tc in test_conditions]
    
    ax.set_title('Generalization Matrix: Modified Architecture (6-element input)\nAll Training × All Test Conditions', 
                 fontsize=16, fontweight='bold')
    ax.set_ylabel('Training Condition', fontsize=12)
    ax.set_xlabel('Test Condition', fontsize=12)
    ax.set_yticklabels(train_labels, rotation=0, fontsize=8)
    ax.set_xticklabels(test_labels, rotation=90, fontsize=8)
    
    plt.tight_layout()
    output_file = output_dir / 'combined_generalization_matrix_modified.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def main():
    """Main analysis function."""
    print("=" * 80)
    print("TEST HEATMAP & KLD ANALYSIS - Modified Architecture (6-element input)")
    print("=" * 80)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Analyze test results for modified architecture')
    parser.add_argument('--test-job-id', type=str, required=True,
                       help='Test job ID (e.g., 912345)')
    parser.add_argument('--train-job-id', type=str, default='910969',
                       help='Training job ID (default: 910969)')
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    test_dir = base_dir / f'generalization_matrix_test_{args.test_job_id}'
    output_dir = Path(__file__).parent / 'output' / f'test_analysis_modified_{args.test_job_id}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTest directory: {test_dir}")
    print(f"Output directory: {output_dir}")
    
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}\n"
            f"Please run the test phase first:\n"
            f"  TRAINING_JOB_ID={args.train_job_id} sbatch run_generalization_matrix_test.sh\n"
            f"  TRAINING_JOB_ID={args.train_job_id} sbatch run_generalization_matrix_test_part2.sh"
        )
    
    # Load test results
    df = load_test_results(test_dir)
    
    # Save raw data
    csv_file = output_dir / 'test_results_all_modified.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n✓ Saved raw test data: {csv_file}")
    
    # Generate analyses
    plot_generalization_heatmaps_by_training_game(df, output_dir)
    plot_kld_analysis(df, output_dir)
    plot_combined_generalization_matrix(df, output_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - Per-game generalization heatmaps (reward + cooperation)")
    print("  - Per-game KLD heatmaps")
    print("  - Combined 15×15 generalization matrix")
    print("  - KLD analysis CSV")
    print("  - Raw test results CSV")
    print("\nNext step: Compare with baseline results (job 888509/902267)")


if __name__ == '__main__':
    main()
