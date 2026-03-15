#!/usr/bin/env python3
"""
Seed Manifest Decoder and Validator

This utility helps decode nested array task IDs and validate seed configurations
for the generalization matrix experiments.

Usage:
    # Decode a specific task ID
    python decode_seed_manifest.py --task-id 37
    
    # Decode all tasks
    python decode_seed_manifest.py --all
    
    # Validate a master registry
    python decode_seed_manifest.py --validate experiments/generalization_matrix_12345/seed_manifests/MASTER_SEED_REGISTRY.csv
    
    # Generate submission script for missing tasks
    python decode_seed_manifest.py --check-missing experiments/generalization_matrix_12345/seed_manifests/MASTER_SEED_REGISTRY.csv
"""

import argparse
import pandas as pd
from typing import Dict, List, Tuple


# Configuration
NUM_SEEDS = 5
SEED_BASE = 42
SEED_GAP = 10
NUM_CONDITIONS = 15

GAMES = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
OPPONENT_RANGES = ['very_low', 'low', 'mid', 'high', 'very_high']


def decode_task_id(task_id: int) -> Dict[str, any]:
    """
    Decode a task ID into its constituent parts.
    
    Args:
        task_id: Array task ID (0-74)
        
    Returns:
        Dictionary with decoded information
    """
    if not 0 <= task_id < (NUM_CONDITIONS * NUM_SEEDS):
        raise ValueError(f"Task ID must be between 0 and {NUM_CONDITIONS * NUM_SEEDS - 1}")
    
    condition_id = task_id // NUM_SEEDS
    seed_id = task_id % NUM_SEEDS
    seed = SEED_BASE + seed_id * SEED_GAP
    
    # Decode condition into game and opponent range
    game_id = condition_id // len(OPPONENT_RANGES)
    opponent_id = condition_id % len(OPPONENT_RANGES)
    
    game = GAMES[game_id]
    opponent_range = OPPONENT_RANGES[opponent_id]
    
    return {
        'task_id': task_id,
        'condition_id': condition_id,
        'seed_id': seed_id,
        'seed': seed,
        'game_id': game_id,
        'game': game,
        'opponent_id': opponent_id,
        'opponent_range': opponent_range
    }


def encode_task_id(condition_id: int, seed_id: int) -> int:
    """
    Encode condition and seed IDs into a task ID.
    
    Args:
        condition_id: Condition ID (0-14)
        seed_id: Seed ID (0-4)
        
    Returns:
        Task ID (0-74)
    """
    return condition_id * NUM_SEEDS + seed_id


def print_task_info(task_id: int):
    """Pretty print information for a task ID."""
    info = decode_task_id(task_id)
    print(f"\n{'='*60}")
    print(f"TASK ID: {info['task_id']}")
    print(f"{'='*60}")
    print(f"  Condition ID:    {info['condition_id']} (of {NUM_CONDITIONS})")
    print(f"  Seed ID:         {info['seed_id']} (of {NUM_SEEDS})")
    print(f"  Random Seed:     {info['seed']}")
    print(f"{'-'*60}")
    print(f"  Game:            {info['game']}")
    print(f"  Opponent Range:  {info['opponent_range']}")
    print(f"{'='*60}")


def print_all_tasks():
    """Print mapping for all tasks."""
    print(f"\n{'='*80}")
    print(f"COMPLETE TASK MAPPING (Total: {NUM_CONDITIONS * NUM_SEEDS} tasks)")
    print(f"{'='*80}")
    print(f"{'Task':>4} | {'Cond':>4} | {'Seed':>4} | {'RandSeed':>8} | {'Game':<20} | {'Opponent Range':<15}")
    print(f"{'-'*80}")
    
    for task_id in range(NUM_CONDITIONS * NUM_SEEDS):
        info = decode_task_id(task_id)
        print(f"{info['task_id']:>4} | {info['condition_id']:>4} | {info['seed_id']:>4} | "
              f"{info['seed']:>8} | {info['game']:<20} | {info['opponent_range']:<15}")


def validate_registry(registry_path: str):
    """
    Validate a master seed registry file.
    
    Checks:
    - All expected tasks are present
    - No duplicate tasks
    - Seeds are correct
    - Condition IDs are valid
    """
    print(f"\n{'='*60}")
    print(f"VALIDATING REGISTRY: {registry_path}")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(registry_path)
    except FileNotFoundError:
        print(f"ERROR: Registry file not found: {registry_path}")
        return
    
    expected_tasks = set(range(NUM_CONDITIONS * NUM_SEEDS))
    actual_tasks = set(df['array_task_id'].values)
    
    missing_tasks = expected_tasks - actual_tasks
    duplicate_tasks = df[df.duplicated(subset=['array_task_id'], keep=False)]
    
    print(f"\nTotal entries: {len(df)}")
    print(f"Expected tasks: {len(expected_tasks)}")
    print(f"Unique tasks: {len(actual_tasks)}")
    
    if missing_tasks:
        print(f"\n⚠️  MISSING TASKS ({len(missing_tasks)}):")
        for task_id in sorted(missing_tasks):
            info = decode_task_id(task_id)
            print(f"  Task {task_id:>3}: {info['game']:<20} {info['opponent_range']:<15} seed={info['seed']}")
    else:
        print("\n✅ All tasks present")
    
    if len(duplicate_tasks) > 0:
        print(f"\n⚠️  DUPLICATE TASKS ({len(duplicate_tasks)}):")
        print(duplicate_tasks[['array_task_id', 'condition_id', 'seed']])
    else:
        print("✅ No duplicates")
    
    # Validate seed mappings
    invalid_seeds = []
    for _, row in df.iterrows():
        expected_info = decode_task_id(row['array_task_id'])
        if row['seed'] != expected_info['seed']:
            invalid_seeds.append({
                'task_id': row['array_task_id'],
                'expected_seed': expected_info['seed'],
                'actual_seed': row['seed']
            })
    
    if invalid_seeds:
        print(f"\n⚠️  INVALID SEEDS ({len(invalid_seeds)}):")
        for item in invalid_seeds:
            print(f"  Task {item['task_id']}: expected {item['expected_seed']}, got {item['actual_seed']}")
    else:
        print("✅ All seeds correct")
    
    print(f"\n{'='*60}")
    
    # Summary statistics
    if len(df) > 0:
        print("\nREGISTRY SUMMARY:")
        print(f"  Conditions covered: {df['condition_id'].nunique()} of {NUM_CONDITIONS}")
        print(f"  Seeds per condition: {df.groupby('condition_id')['seed_id'].nunique().describe()}")
        print(f"  Date range: {df['start_time'].min()} to {df['start_time'].max()}")


def generate_missing_tasks_script(registry_path: str, output_file: str = "resubmit_missing.sh"):
    """Generate a script to resubmit missing tasks."""
    print(f"\n{'='*60}")
    print(f"GENERATING RESUBMISSION SCRIPT")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(registry_path)
    except FileNotFoundError:
        print(f"ERROR: Registry file not found: {registry_path}")
        return
    
    expected_tasks = set(range(NUM_CONDITIONS * NUM_SEEDS))
    actual_tasks = set(df['array_task_id'].values)
    missing_tasks = sorted(expected_tasks - actual_tasks)
    
    if not missing_tasks:
        print("✅ No missing tasks - all complete!")
        return
    
    # Generate resubmission script
    with open(output_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated script to resubmit missing tasks\n")
        f.write(f"# Generated for registry: {registry_path}\n")
        f.write(f"# Missing tasks: {len(missing_tasks)}\n\n")
        
        task_list = ','.join(map(str, missing_tasks))
        f.write(f"sbatch --array={task_list} run_generalization_matrix.sh\n")
    
    print(f"\n✅ Generated: {output_file}")
    print(f"   Missing tasks: {len(missing_tasks)}")
    print(f"   Task IDs: {task_list}")
    print(f"\nTo resubmit: bash {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Decode and validate nested array task IDs for generalization matrix experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--task-id', type=int, help='Decode a specific task ID (0-74)')
    parser.add_argument('--all', action='store_true', help='Print all task mappings')
    parser.add_argument('--validate', type=str, metavar='REGISTRY_FILE',
                       help='Validate a master seed registry CSV file')
    parser.add_argument('--check-missing', type=str, metavar='REGISTRY_FILE',
                       help='Generate resubmission script for missing tasks')
    parser.add_argument('--output', type=str, default='resubmit_missing.sh',
                       help='Output filename for resubmission script (default: resubmit_missing.sh)')
    
    args = parser.parse_args()
    
    if args.task_id is not None:
        print_task_info(args.task_id)
    
    if args.all:
        print_all_tasks()
    
    if args.validate:
        validate_registry(args.validate)
    
    if args.check_missing:
        generate_missing_tasks_script(args.check_missing, args.output)
    
    if not any([args.task_id is not None, args.all, args.validate, args.check_missing]):
        parser.print_help()


if __name__ == '__main__':
    main()
