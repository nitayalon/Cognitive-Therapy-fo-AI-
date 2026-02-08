#!/usr/bin/env python3
"""
Aggregate results from generalization matrix experiment.

This script combines results from all 16 SLURM array tasks into a single
comprehensive report with comparison tables and visualizations.

Usage:
    python aggregate_generalization_results.py --input-dir experiments/generalization_matrix_JOBID
"""

import argparse
import json
import os
import glob
from datetime import datetime
from typing import Dict, Any, List
import numpy as np


def load_task_results(input_dir: str) -> Dict[int, Dict[str, Any]]:
    """Load all task results from the experiment directory."""
    results = {}
    
    # Find all task directories
    task_dirs = glob.glob(os.path.join(input_dir, "generalization_matrix_task_*"))
    
    for task_dir in task_dirs:
        # Extract task ID from directory name
        dir_name = os.path.basename(task_dir)
        try:
            task_id = int(dir_name.split('_')[3])
        except (IndexError, ValueError):
            print(f"Warning: Could not parse task ID from {dir_name}")
            continue
        
        # Load the report JSON
        report_files = glob.glob(os.path.join(task_dir, "results", "task_*_report.json"))
        if report_files:
            with open(report_files[0], 'r') as f:
                results[task_id] = json.load(f)
            print(f"Loaded task {task_id}: {results[task_id].get('training_condition', {})}")
    
    return results


def create_summary_tables(results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Create summary tables for the experiment."""
    
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt', 'battle-of-sexes']
    opponent_ranges = ['low', 'mid_low', 'mid_high', 'high']
    
    # Initialize tables
    baseline_reward_table = {g: {r: None for r in opponent_ranges} for g in games}
    baseline_coop_table = {g: {r: None for r in opponent_ranges} for g in games}
    
    # Generalization error tables
    same_game_gen_error = {g: {r: {} for r in opponent_ranges} for g in games}
    cross_game_gen_error = {g: {r: {} for r in opponent_ranges} for g in games}
    
    for task_id, result in results.items():
        train_cond = result.get('training_condition', {})
        game = train_cond.get('game')
        opp_range = train_cond.get('opponent_range')
        
        if game is None or opp_range is None:
            continue
        
        summaries = result.get('evaluation_summaries', {})
        gen_errors = result.get('generalization_errors', {})
        
        # Baseline metrics
        baseline = summaries.get('baseline', {})
        baseline_reward_table[game][opp_range] = baseline.get('mean_reward', 0)
        baseline_coop_table[game][opp_range] = baseline.get('mean_cooperation_rate', 0)
        
        # Same game generalization errors
        for key, error in gen_errors.items():
            if key.startswith('same_game_'):
                test_range = key.replace('same_game_', '')
                same_game_gen_error[game][opp_range][test_range] = error.get('reward_delta', 0)
        
        # Cross-game generalization errors
        for key, error in gen_errors.items():
            if '_same_opponents' in key or ('_low' in key or '_mid' in key or '_high' in key):
                if not key.startswith('same_game_'):
                    cross_game_gen_error[game][opp_range][key] = error.get('reward_delta', 0)
    
    return {
        'baseline_reward': baseline_reward_table,
        'baseline_cooperation': baseline_coop_table,
        'same_game_generalization_error': same_game_gen_error,
        'cross_game_generalization_error': cross_game_gen_error
    }


def format_table(table: Dict[str, Dict[str, Any]], title: str) -> str:
    """Format a table as a string for printing."""
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt', 'battle-of-sexes']
    opponent_ranges = ['low', 'mid_low', 'mid_high', 'high']
    
    lines = [f"\n{title}", "=" * 80]
    
    # Header
    header = f"{'Game':<20} | " + " | ".join(f"{r:>10}" for r in opponent_ranges)
    lines.append(header)
    lines.append("-" * 80)
    
    # Data rows
    for game in games:
        row_data = []
        for opp_range in opponent_ranges:
            val = table.get(game, {}).get(opp_range)
            if val is None:
                row_data.append(f"{'N/A':>10}")
            elif isinstance(val, float):
                row_data.append(f"{val:>10.4f}")
            else:
                row_data.append(f"{str(val):>10}")
        
        row = f"{game:<20} | " + " | ".join(row_data)
        lines.append(row)
    
    return "\n".join(lines)


def compute_aggregate_metrics(results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics across all experiments."""
    
    all_baseline_rewards = []
    all_baseline_coop = []
    all_same_game_errors = []
    all_cross_game_errors = []
    
    for task_id, result in results.items():
        summaries = result.get('evaluation_summaries', {})
        gen_errors = result.get('generalization_errors', {})
        
        baseline = summaries.get('baseline', {})
        all_baseline_rewards.append(baseline.get('mean_reward', 0))
        all_baseline_coop.append(baseline.get('mean_cooperation_rate', 0))
        
        for key, error in gen_errors.items():
            if key.startswith('same_game_'):
                all_same_game_errors.append(abs(error.get('reward_delta', 0)))
            else:
                all_cross_game_errors.append(abs(error.get('reward_delta', 0)))
    
    return {
        'num_tasks_completed': len(results),
        'avg_baseline_reward': np.mean(all_baseline_rewards) if all_baseline_rewards else 0,
        'std_baseline_reward': np.std(all_baseline_rewards) if all_baseline_rewards else 0,
        'avg_baseline_coop': np.mean(all_baseline_coop) if all_baseline_coop else 0,
        'std_baseline_coop': np.std(all_baseline_coop) if all_baseline_coop else 0,
        'avg_same_game_gen_error': np.mean(all_same_game_errors) if all_same_game_errors else 0,
        'avg_cross_game_gen_error': np.mean(all_cross_game_errors) if all_cross_game_errors else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate generalization matrix results")
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing experiment results')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file for aggregated report (JSON)')
    args = parser.parse_args()
    
    print(f"Loading results from: {args.input_dir}")
    results = load_task_results(args.input_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"\nLoaded {len(results)} task results")
    
    # Create summary tables
    tables = create_summary_tables(results)
    
    # Compute aggregate metrics
    aggregates = compute_aggregate_metrics(results)
    
    # Print results
    print("\n" + "=" * 80)
    print("GENERALIZATION MATRIX EXPERIMENT - AGGREGATED RESULTS")
    print("=" * 80)
    
    print(f"\nTasks completed: {aggregates['num_tasks_completed']}/16")
    print(f"\nOverall Baseline Performance:")
    print(f"  Mean Reward: {aggregates['avg_baseline_reward']:.4f} ± {aggregates['std_baseline_reward']:.4f}")
    print(f"  Mean Cooperation: {aggregates['avg_baseline_coop']:.4f} ± {aggregates['std_baseline_coop']:.4f}")
    print(f"\nGeneralization Error (absolute reward delta):")
    print(f"  Same Game, New Opponents: {aggregates['avg_same_game_gen_error']:.4f}")
    print(f"  Cross-Game: {aggregates['avg_cross_game_gen_error']:.4f}")
    
    # Print tables
    print(format_table(tables['baseline_reward'], "Baseline Reward by Training Condition"))
    print(format_table(tables['baseline_cooperation'], "Baseline Cooperation Rate by Training Condition"))
    
    # Save report
    output_file = args.output_file or os.path.join(args.input_dir, 'aggregated_report.json')
    report = {
        'timestamp': datetime.now().isoformat(),
        'num_tasks': len(results),
        'aggregate_metrics': aggregates,
        'tables': tables,
        'per_task_summaries': {
            task_id: {
                'training_condition': result.get('training_condition', {}),
                'evaluation_summaries': result.get('evaluation_summaries', {}),
                'generalization_errors': result.get('generalization_errors', {})
            }
            for task_id, result in results.items()
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nAggregated report saved to: {output_file}")


if __name__ == "__main__":
    main()
