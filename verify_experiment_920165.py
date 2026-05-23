#!/usr/bin/env python3
"""
Verify completeness and architecture of experiment 920165.

Checks:
1. Network architecture (small vs standard)
2. Training data completeness
3. Test data availability
4. Expected vs actual task counts
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Paths
PROJECT_ROOT = Path(__file__).parent
TRAIN_DIR = PROJECT_ROOT / 'experiments' / 'whole_population_train_920165' / 'training'
TEST_DIR = PROJECT_ROOT / 'experiments' / 'whole_population_test_920165' / 'testing'
REGISTRY_PATH = PROJECT_ROOT / 'experiments' / 'whole_population_train_920165' / 'seed_manifests' / 'MASTER_TRAINING_REGISTRY.csv'

# Expected configuration for small network (from whole_population_config.json)
EXPECTED_SMALL_NETWORK = {
    'hidden_size': 32,
    'num_layers': 1,
    'dropout': 0.05,
    'input_size': 9
}

# Expected configuration for standard network (from generalization_matrix_config.json)
EXPECTED_STANDARD_NETWORK = {
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.1,
    'input_size': 9
}

# Expected setup
EXPECTED_GAMES = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
EXPECTED_CONDITIONS = 3  # 3 games
EXPECTED_SEEDS = 5  # 5 seeds per condition
EXPECTED_TOTAL_TASKS = EXPECTED_CONDITIONS * EXPECTED_SEEDS  # 15 total training runs

def load_network_config(task_dir):
    """Load network configuration from experiment config."""
    config_path = task_dir / 'experiment_config.json'
    if not config_path.exists():
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config.get('network_config', {})

def check_network_architecture():
    """Verify network architecture across all tasks."""
    print("=" * 80)
    print("NETWORK ARCHITECTURE VERIFICATION")
    print("=" * 80)
    
    task_dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    
    if not task_dirs:
        print("❌ ERROR: No training task directories found!")
        return False
    
    # Check first task for architecture
    first_config = load_network_config(task_dirs[0])
    if not first_config:
        print("❌ ERROR: Could not load network config from first task!")
        return False
    
    print(f"\nNetwork configuration from {task_dirs[0].name}:")
    print(f"  hidden_size: {first_config.get('hidden_size')}")
    print(f"  num_layers: {first_config.get('num_layers')}")
    print(f"  dropout: {first_config.get('dropout')}")
    print(f"  input_size: {first_config.get('input_size')}")
    
    # Determine architecture type
    is_small = (
        first_config.get('hidden_size') == EXPECTED_SMALL_NETWORK['hidden_size'] and
        first_config.get('num_layers') == EXPECTED_SMALL_NETWORK['num_layers'] and
        first_config.get('dropout') == EXPECTED_SMALL_NETWORK['dropout']
    )
    
    is_standard = (
        first_config.get('hidden_size') == EXPECTED_STANDARD_NETWORK['hidden_size'] and
        first_config.get('num_layers') == EXPECTED_STANDARD_NETWORK['num_layers'] and
        first_config.get('dropout') == EXPECTED_STANDARD_NETWORK['dropout']
    )
    
    print("\nArchitecture type:")
    if is_small:
        print("  ✅ SMALL NETWORK (32 hidden units, 1 layer, 0.05 dropout)")
        architecture_type = "SMALL"
    elif is_standard:
        print("  ⚠️  STANDARD NETWORK (128 hidden units, 2 layers, 0.1 dropout)")
        print("      WARNING: This is NOT the small network architecture!")
        architecture_type = "STANDARD"
    else:
        print("  ❓ UNKNOWN NETWORK (does not match expected patterns)")
        architecture_type = "UNKNOWN"
    
    # Verify consistency across all tasks
    print(f"\nVerifying consistency across {len(task_dirs)} tasks...")
    all_consistent = True
    for task_dir in task_dirs:
        config = load_network_config(task_dir)
        if config != first_config:
            print(f"  ❌ INCONSISTENT: {task_dir.name}")
            all_consistent = False
    
    if all_consistent:
        print(f"  ✅ All {len(task_dirs)} tasks use identical network architecture")
    
    return architecture_type, all_consistent

def check_training_data():
    """Check training data completeness."""
    print("\n" + "=" * 80)
    print("TRAINING DATA VERIFICATION")
    print("=" * 80)
    
    # Load registry
    if not REGISTRY_PATH.exists():
        print(f"❌ ERROR: Registry not found at {REGISTRY_PATH}")
        return False
    
    df_registry = pd.read_csv(REGISTRY_PATH)
    print(f"\nMaster training registry: {len(df_registry)} records")
    print(f"  Array job ID: {df_registry['array_job_id'].unique()}")
    print(f"  Conditions: {sorted(df_registry['condition_id'].unique())}")
    print(f"  Seeds: {sorted(df_registry['seed_id'].unique())}")
    print(f"  Tasks: {sorted(df_registry['array_task_id'].unique())}")
    
    # Expected vs actual
    expected_tasks = EXPECTED_TOTAL_TASKS
    actual_tasks = len(df_registry)
    
    print(f"\nExpected tasks: {expected_tasks} (3 conditions × 5 seeds)")
    print(f"Actual tasks: {actual_tasks}")
    
    if actual_tasks == expected_tasks:
        print("  ✅ Complete training data")
    else:
        print(f"  ❌ INCOMPLETE: Missing {expected_tasks - actual_tasks} tasks")
    
    # Check task directories
    task_dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    print(f"\nTask directories found: {len(task_dirs)}")
    
    # Check each task for required files
    print("\nChecking individual tasks:")
    complete_count = 0
    for task_dir in task_dirs:
        config_exists = (task_dir / 'experiment_config.json').exists()
        checkpoint_dir = task_dir / 'checkpoints'
        logs_dir = task_dir / 'logs'
        
        has_checkpoints = checkpoint_dir.exists() and any(checkpoint_dir.iterdir())
        has_logs = logs_dir.exists() and any(logs_dir.iterdir())
        
        is_complete = config_exists and has_checkpoints and has_logs
        
        if is_complete:
            complete_count += 1
        else:
            status_parts = []
            if not config_exists:
                status_parts.append("no config")
            if not has_checkpoints:
                status_parts.append("no checkpoints")
            if not has_logs:
                status_parts.append("no logs")
            
            print(f"  ❌ {task_dir.name}: {', '.join(status_parts)}")
    
    print(f"\nComplete tasks: {complete_count}/{len(task_dirs)}")
    
    # Analyze by condition
    print("\nBreakdown by condition (from registry):")
    for condition_id in sorted(df_registry['condition_id'].unique()):
        condition_data = df_registry[df_registry['condition_id'] == condition_id]
        num_seeds = len(condition_data)
        print(f"  Condition {condition_id}: {num_seeds} seeds")
    
    return complete_count == expected_tasks

def check_test_data():
    """Check for test data."""
    print("\n" + "=" * 80)
    print("TEST DATA VERIFICATION")
    print("=" * 80)
    
    if not TEST_DIR.exists():
        print(f"❌ No test directory found at:")
        print(f"   {TEST_DIR}")
        print("\n   Expected location: experiments/whole_population_test_920165/testing/")
        return False
    
    test_dirs = sorted([d for d in TEST_DIR.iterdir() if d.is_dir()])
    print(f"\nTest directories found: {len(test_dirs)}")
    
    if len(test_dirs) == 0:
        print("❌ Test directory exists but is empty")
        return False
    
    # Expected test structure: 15 training conditions × 15 test conditions = 225 test runs
    expected_test_runs = EXPECTED_TOTAL_TASKS * len(EXPECTED_GAMES) * 5  # 15 train × (3 games × 5 opponents)
    
    print(f"\nExpected test runs: ~{expected_test_runs}")
    print(f"Actual test directories: {len(test_dirs)}")
    
    return True

def main():
    """Run all verification checks."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "EXPERIMENT 920165 VERIFICATION" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    results = {}
    
    # Check 1: Network architecture
    arch_result = check_network_architecture()
    if arch_result:
        architecture_type, consistency = arch_result
        results['architecture'] = architecture_type
        results['architecture_consistent'] = consistency
    else:
        results['architecture'] = 'ERROR'
        results['architecture_consistent'] = False
    
    # Check 2: Training data
    results['training_complete'] = check_training_data()
    
    # Check 3: Test data
    results['test_exists'] = check_test_data()
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"\nArchitecture type: {results['architecture']}")
    
    if results['architecture'] == 'SMALL':
        print("  ✅ Confirmed: Small network (32 hidden, 1 layer)")
    elif results['architecture'] == 'STANDARD':
        print("  ⚠️  WARNING: Using standard network (128 hidden, 2 layers)")
        print("      This is NOT the small/reduced network!")
    
    print(f"\nArchitecture consistency: {'✅ Yes' if results['architecture_consistent'] else '❌ No'}")
    print(f"Training data complete: {'✅ Yes' if results['training_complete'] else '❌ No'}")
    print(f"Test data available: {'✅ Yes' if results['test_exists'] else '❌ No'}")
    
    # Overall status
    print("\n" + "-" * 80)
    if (results['architecture'] == 'SMALL' and 
        results['architecture_consistent'] and 
        results['training_complete'] and 
        results['test_exists']):
        print("✅ EXPERIMENT 920165: FULLY VERIFIED (Small Network)")
    elif (results['architecture'] == 'STANDARD' and 
          results['architecture_consistent'] and 
          results['training_complete']):
        print("⚠️  EXPERIMENT 920165: COMPLETE BUT USING STANDARD NETWORK")
        print("    Expected small network (32 hidden), found standard network (128 hidden)")
    else:
        print("❌ EXPERIMENT 920165: ISSUES DETECTED")
        if not results['training_complete']:
            print("    - Training data incomplete")
        if not results['test_exists']:
            print("    - Test data not found")
        if not results['architecture_consistent']:
            print("    - Network architecture inconsistent across tasks")
    
    print("=" * 80)
    print()

if __name__ == '__main__':
    main()
