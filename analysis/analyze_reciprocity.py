#!/usr/bin/env python3
"""
Reciprocity Analysis for Mixed-Motive Game Experiments

Analyzes whether agents learn social dynamics (reciprocity) or pure Nash equilibrium strategies.
Computes P(agent_coop_t | opponent_coop_t-1) to measure reciprocal cooperation.

Author: Research Team
Date: May 2026
"""

import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def load_detailed_test_logs(test_dir: Path, experiment_type: str) -> pd.DataFrame:
    """
    Load detailed testing logs from all test tasks.
    
    Args:
        test_dir: Path to testing directory
        experiment_type: 'whole_population' or 'generalization_matrix'
    
    Returns:
        DataFrame with all episode-level data
    """
    all_logs = []
    
    if experiment_type == 'whole_population':
        # Structure: testing/whole_population_task_X_*/logs/eval_task_X/detailed_testing_log.csv
        task_dirs = sorted(test_dir.glob('whole_population_task_*'))
        
        for task_dir in task_dirs:
            task_id = int(task_dir.name.split('_')[3])
            log_file = task_dir / 'logs' / f'eval_task_{task_id}' / 'detailed_testing_log.csv'
            
            if log_file.exists():
                try:
                    df = pd.read_csv(log_file)
                    df['task_id'] = task_id
                    all_logs.append(df)
                except Exception as e:
                    print(f"  Warning: Error loading {log_file}: {e}")
    
    elif experiment_type == 'generalization_matrix':
        # Structure: testing/model_X_test_cond_Y/generalization_matrix_task_Z_*/logs/eval_cond_Y/detailed_testing_log.csv
        test_cond_dirs = sorted(test_dir.glob('model_*_test_cond_*'))
        
        for cond_dir in test_cond_dirs:
            # Parse model_X_test_cond_Y
            parts = cond_dir.name.split('_')
            model_id = int(parts[1])
            test_cond = int(parts[4])
            
            # Find the task directory inside
            task_dirs = list(cond_dir.glob('generalization_matrix_task_*'))
            if not task_dirs:
                continue
            
            task_dir = task_dirs[0]
            log_file = task_dir / 'logs' / f'eval_cond_{test_cond}' / 'detailed_testing_log.csv'
            
            if log_file.exists():
                try:
                    df = pd.read_csv(log_file)
                    df['model_id'] = model_id
                    df['test_cond'] = test_cond
                    all_logs.append(df)
                except Exception as e:
                    print(f"  Warning: Error loading {log_file}: {e}")
    
    if not all_logs:
        raise ValueError(f"No detailed logs found in {test_dir}")
    
    return pd.concat(all_logs, ignore_index=True)


def compute_reciprocity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute reciprocity metrics from episode-level data.
    
    For each game, computes:
    - P(agent_coop_t | opponent_coop_t-1): Probability agent cooperates given opponent cooperated
    - P(agent_coop_t | opponent_defect_t-1): Probability agent cooperates given opponent defected
    - reciprocity_strength: Difference between the two (positive = reciprocal)
    
    Args:
        df: DataFrame with columns: agent_sampled_action, opponent_actual_action, 
            test_session, session_game_number, game_step_in_session
    
    Returns:
        DataFrame with reciprocity metrics per game
    """
    results = []
    
    # Group by session and game
    for (session, game_num), game_df in df.groupby(['test_session', 'session_game_number']):
        # Sort by game step to ensure correct temporal order
        game_df = game_df.sort_values('game_step_in_session')
        
        # Skip first trial (no previous opponent action)
        if len(game_df) < 2:
            continue
        
        # Create lagged opponent action
        game_df = game_df.copy()
        game_df['opponent_prev_action'] = game_df['opponent_actual_action'].shift(1)
        
        # Remove first row (no previous action)
        game_df = game_df.iloc[1:].copy()
        
        # Convert actions: 0=cooperate, 1=defect
        # Agent cooperates if action == 0
        game_df['agent_cooperates'] = (game_df['agent_sampled_action'] == 0).astype(int)
        game_df['opponent_prev_cooperates'] = (game_df['opponent_prev_action'] == 0).astype(int)
        
        # Compute conditional probabilities
        # P(agent_coop | opponent_prev_coop)
        opp_coop_mask = game_df['opponent_prev_cooperates'] == 1
        if opp_coop_mask.sum() > 0:
            p_coop_given_opp_coop = game_df.loc[opp_coop_mask, 'agent_cooperates'].mean()
            n_opp_coop = opp_coop_mask.sum()
        else:
            p_coop_given_opp_coop = np.nan
            n_opp_coop = 0
        
        # P(agent_coop | opponent_prev_defect)
        opp_defect_mask = game_df['opponent_prev_cooperates'] == 0
        if opp_defect_mask.sum() > 0:
            p_coop_given_opp_defect = game_df.loc[opp_defect_mask, 'agent_cooperates'].mean()
            n_opp_defect = opp_defect_mask.sum()
        else:
            p_coop_given_opp_defect = np.nan
            n_opp_defect = 0
        
        # Reciprocity strength: positive = reciprocal, negative = anti-reciprocal
        if not np.isnan(p_coop_given_opp_coop) and not np.isnan(p_coop_given_opp_defect):
            reciprocity = p_coop_given_opp_coop - p_coop_given_opp_defect
        else:
            reciprocity = np.nan
        
        # Extract metadata from first row
        metadata = game_df.iloc[0]
        
        results.append({
            'test_session': session,
            'session_game_number': game_num,
            'game_name': metadata.get('game_name', 'unknown'),
            'opponent_type': metadata.get('opponent_type', metadata.get('true_opponent_defect_prob', 'unknown')),
            'p_coop_given_opp_coop': p_coop_given_opp_coop,
            'p_coop_given_opp_defect': p_coop_given_opp_defect,
            'reciprocity_strength': reciprocity,
            'n_trials_after_opp_coop': n_opp_coop,
            'n_trials_after_opp_defect': n_opp_defect,
            'total_trials': len(game_df)
        })
    
    return pd.DataFrame(results)


def aggregate_reciprocity_by_condition(reciprocity_df: pd.DataFrame, metadata_df: pd.DataFrame, 
                                       experiment_type: str) -> pd.DataFrame:
    """
    Aggregate reciprocity metrics by training game and test opponent.
    
    Args:
        reciprocity_df: Per-game reciprocity metrics
        metadata_df: Metadata with training game info
        experiment_type: Type of experiment
    
    Returns:
        Aggregated metrics
    """
    # Merge with metadata to get training game
    if experiment_type == 'whole_population':
        # Get training game from task_id
        # Task structure: task_id // 15 = model_id, model_id % 5 = seed
        # Models 0-4: PD, 5-9: HD, 10-14: SH (hypothetical)
        pass  # Will be added from metadata
    
    # Extract opponent defection probability
    reciprocity_df['opponent_defect_prob'] = reciprocity_df['opponent_type'].astype(float)
    
    # Group by game and opponent
    agg = reciprocity_df.groupby(['game_name', 'opponent_defect_prob']).agg({
        'p_coop_given_opp_coop': ['mean', 'std', 'count'],
        'p_coop_given_opp_defect': ['mean', 'std'],
        'reciprocity_strength': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                   for col in agg.columns.values]
    
    return agg


def plot_reciprocity_by_game(reciprocity_agg: pd.DataFrame, output_dir: Path, 
                             experiment_name: str, train_game: str = None):
    """
    Plot reciprocity metrics by test game and opponent.
    
    Args:
        reciprocity_agg: Aggregated reciprocity metrics
        output_dir: Directory for output plots
        experiment_name: Name for plot titles
        train_game: Training game name (optional)
    """
    # Map game names
    game_name_map = {
        'Prisoners-Dilemma': "Prisoner's Dilemma",
        'prisoners-dilemma': "Prisoner's Dilemma",
        'Hawk-Dove': "Hawk-Dove",
        'hawk-dove': "Hawk-Dove",
        'Stag-Hunt': "Stag-Hunt",
        'stag-hunt': "Stag-Hunt"
    }
    
    reciprocity_agg['game_display'] = reciprocity_agg['game_name'].map(
        lambda x: game_name_map.get(x, x)
    )
    
    games = reciprocity_agg['game_display'].unique()
    
    fig, axes = plt.subplots(1, len(games), figsize=(6*len(games), 5))
    if len(games) == 1:
        axes = [axes]
    
    for idx, game in enumerate(sorted(games)):
        ax = axes[idx]
        game_data = reciprocity_agg[reciprocity_agg['game_display'] == game].sort_values('opponent_defect_prob')
        
        # Plot P(coop | opp_coop) and P(coop | opp_defect)
        ax.errorbar(game_data['opponent_defect_prob'], 
                   game_data['p_coop_given_opp_coop_mean'],
                   yerr=game_data['p_coop_given_opp_coop_std'],
                   marker='o', label='After Opp Cooperates', 
                   linewidth=2, markersize=8, capsize=5)
        
        ax.errorbar(game_data['opponent_defect_prob'], 
                   game_data['p_coop_given_opp_defect_mean'],
                   yerr=game_data['p_coop_given_opp_defect_std'],
                   marker='s', label='After Opp Defects',
                   linewidth=2, markersize=8, capsize=5)
        
        ax.set_xlabel('Opponent Defection Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('P(Agent Cooperates)', fontsize=12, fontweight='bold')
        ax.set_title(f'{game}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.05, 1.0)
    
    title = f'Reciprocity Analysis: {experiment_name}'
    if train_game:
        title += f'\nTrained on {train_game}'
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename = f'reciprocity_{experiment_name.lower().replace(" ", "_")}'
    if train_game:
        filename += f'_{train_game.lower().replace(" ", "_")}'
    filename += '.png'
    
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_dir / filename}")


def plot_reciprocity_strength(reciprocity_agg: pd.DataFrame, output_dir: Path,
                              experiment_name: str):
    """
    Plot reciprocity strength (difference between conditional cooperation probabilities).
    
    Positive values indicate reciprocity (more likely to cooperate after opponent cooperates).
    Negative values indicate anti-reciprocity.
    Zero indicates no social learning (pure Nash).
    """
    game_name_map = {
        'Prisoners-Dilemma': "Prisoner's Dilemma",
        'prisoners-dilemma': "Prisoner's Dilemma",
        'Hawk-Dove': "Hawk-Dove",
        'hawk-dove': "Hawk-Dove",
        'Stag-Hunt': "Stag-Hunt",
        'stag-hunt': "Stag-Hunt"
    }
    
    reciprocity_agg['game_display'] = reciprocity_agg['game_name'].map(
        lambda x: game_name_map.get(x, x)
    )
    
    plt.figure(figsize=(10, 6))
    
    for game in sorted(reciprocity_agg['game_display'].unique()):
        game_data = reciprocity_agg[reciprocity_agg['game_display'] == game].sort_values('opponent_defect_prob')
        
        plt.errorbar(game_data['opponent_defect_prob'],
                    game_data['reciprocity_strength_mean'],
                    yerr=game_data['reciprocity_strength_std'],
                    marker='o', label=game, linewidth=2, markersize=8, capsize=5)
    
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, 
                label='No Reciprocity (Nash)')
    plt.xlabel('Opponent Defection Probability', fontsize=12, fontweight='bold')
    plt.ylabel('Reciprocity Strength\n[P(coop|opp_coop) - P(coop|opp_defect)]', 
               fontsize=12, fontweight='bold')
    plt.title(f'Reciprocity Strength: {experiment_name}\n' + 
              'Positive = Reciprocal, Negative = Anti-Reciprocal, Zero = No Social Learning',
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.0)
    
    plt.tight_layout()
    filename = f'reciprocity_strength_{experiment_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_dir / filename}")


def main():
    parser = argparse.ArgumentParser(description='Analyze reciprocity in test results')
    parser.add_argument('--whole-pop-test', type=str, 
                       help='Job ID for whole population test (task-only)')
    parser.add_argument('--task-opp-test', type=str,
                       help='Job ID for task-opponent test (with opponent action input)')
    parser.add_argument('--output-dir', type=str, default='experiments/analysis_scripts/output',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not args.whole_pop_test and not args.task_opp_test:
        parser.error("Must provide at least one test job ID")
    
    base_dir = Path.cwd()
    output_base = base_dir / args.output_dir
    
    print("="*80)
    print("RECIPROCITY ANALYSIS")
    print("="*80)
    print()
    
    # Analyze whole population (task-only)
    if args.whole_pop_test:
        print(f"Analyzing Whole Population Test (Job {args.whole_pop_test})...")
        test_dir = base_dir / 'experiments' / f'whole_population_test_{args.whole_pop_test}' / 'testing'
        output_dir = output_base / f'reciprocity_whole_population_{args.whole_pop_test}'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Loading detailed logs from {test_dir}...")
        df = load_detailed_test_logs(test_dir, 'whole_population')
        print(f"  ✓ Loaded {len(df):,} episodes from {df['test_session'].nunique()} sessions")
        
        print("  Computing reciprocity metrics...")
        reciprocity_df = compute_reciprocity_metrics(df)
        print(f"  ✓ Computed metrics for {len(reciprocity_df):,} games")
        
        # Save raw reciprocity data
        reciprocity_df.to_csv(output_dir / 'reciprocity_per_game.csv', index=False)
        print(f"  ✓ Saved raw data: {output_dir / 'reciprocity_per_game.csv'}")
        
        # Aggregate by condition
        agg = aggregate_reciprocity_by_condition(reciprocity_df, df, 'whole_population')
        agg.to_csv(output_dir / 'reciprocity_aggregated.csv', index=False)
        print(f"  ✓ Saved aggregated data: {output_dir / 'reciprocity_aggregated.csv'}")
        
        # Generate plots
        print("  Generating plots...")
        plot_reciprocity_by_game(agg, output_dir, 'Whole Population (Task-Only)')
        plot_reciprocity_strength(agg, output_dir, 'Whole Population (Task-Only)')
        
        print()
    
    # Analyze task-opponent
    if args.task_opp_test:
        print(f"Analyzing Task-Opponent Test (Job {args.task_opp_test})...")
        test_dir = base_dir / 'experiments' / f'generalization_matrix_test_{args.task_opp_test}' / 'testing'
        output_dir = output_base / f'reciprocity_task_opponent_{args.task_opp_test}'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Loading detailed logs from {test_dir}...")
        df = load_detailed_test_logs(test_dir, 'generalization_matrix')
        print(f"  ✓ Loaded {len(df):,} episodes from {df['test_session'].nunique()} sessions")
        
        print("  Computing reciprocity metrics...")
        reciprocity_df = compute_reciprocity_metrics(df)
        print(f"  ✓ Computed metrics for {len(reciprocity_df):,} games")
        
        # Save raw reciprocity data
        reciprocity_df.to_csv(output_dir / 'reciprocity_per_game.csv', index=False)
        print(f"  ✓ Saved raw data: {output_dir / 'reciprocity_per_game.csv'}")
        
        # Aggregate by condition
        agg = aggregate_reciprocity_by_condition(reciprocity_df, df, 'generalization_matrix')
        agg.to_csv(output_dir / 'reciprocity_aggregated.csv', index=False)
        print(f"  ✓ Saved aggregated data: {output_dir / 'reciprocity_aggregated.csv'}")
        
        # Generate plots
        print("  Generating plots...")
        plot_reciprocity_by_game(agg, output_dir, 'Task-Opponent (With Opponent Action Input)')
        plot_reciprocity_strength(agg, output_dir, 'Task-Opponent (With Opponent Action Input)')
        
        print()
    
    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print()
    print("Interpretation Guide:")
    print("  - Reciprocity > 0: Agent more likely to cooperate after opponent cooperates (social learning)")
    print("  - Reciprocity ≈ 0: Agent ignores opponent's previous action (pure Nash equilibrium)")
    print("  - Reciprocity < 0: Agent anti-reciprocates (defects more after cooperation)")
    print()


if __name__ == '__main__':
    main()
