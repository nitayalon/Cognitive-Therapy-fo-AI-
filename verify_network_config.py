#!/usr/bin/env python3
"""
Verify network configuration matches expected config file settings.

This script prevents the bug where NetworkConfig() defaults override JSON settings.
Use this to validate experiments before and after running.

Usage:
    python verify_network_config.py experiments/generalization_matrix_train_918988/training/
    python verify_network_config.py experiments/whole_population_train_920165/training/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Expected configurations
EXPECTED_CONFIGS = {
    'small_network': {
        'hidden_size': 32,
        'num_layers': 1,
        'dropout': 0.05,
        'input_size': 9
    },
    'standard_network': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.1,
        'input_size': 9
    }
}

def load_task_config(task_dir: Path) -> Dict[str, Any]:
    """Load network configuration from a task directory."""
    config_path = task_dir / 'experiment_config.json'
    if not config_path.exists():
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config.get('network_config', {})

def identify_network_type(network_config: Dict[str, Any]) -> str:
    """Identify which network type this config matches."""
    for name, expected in EXPECTED_CONFIGS.items():
        if all(network_config.get(k) == v for k, v in expected.items()):
            return name
    return 'unknown'

def verify_experiment_directory(exp_dir: Path, expected_type: str = None) -> Dict[str, Any]:
    """
    Verify all tasks in an experiment directory have consistent network configs.
    
    Args:
        exp_dir: Path to experiment directory (training or testing subdirectory)
        expected_type: Expected network type ('small_network' or 'standard_network')
    
    Returns:
        Dictionary with verification results
    """
    print(f"\nVerifying: {exp_dir}")
    print("=" * 80)
    
    # Find all task directories
    if not exp_dir.exists():
        print(f"❌ ERROR: Directory not found: {exp_dir}")
        return {'status': 'error', 'message': 'Directory not found'}
    
    task_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
    
    if not task_dirs:
        print(f"❌ ERROR: No task directories found")
        return {'status': 'error', 'message': 'No task directories'}
    
    print(f"Found {len(task_dirs)} task directories")
    
    # Check first task for network config
    first_config = load_task_config(task_dirs[0])
    if not first_config:
        print(f"❌ ERROR: Could not load config from {task_dirs[0].name}")
        return {'status': 'error', 'message': 'Could not load config'}
    
    network_type = identify_network_type(first_config)
    
    print(f"\nNetwork configuration (from {task_dirs[0].name}):")
    print(f"  hidden_size:  {first_config.get('hidden_size')}")
    print(f"  num_layers:   {first_config.get('num_layers')}")
    print(f"  dropout:      {first_config.get('dropout')}")
    print(f"  input_size:   {first_config.get('input_size')}")
    print(f"\nIdentified as: {network_type}")
    
    # Check if matches expected type
    if expected_type:
        if network_type == expected_type:
            print(f"✅ MATCHES expected type: {expected_type}")
        else:
            print(f"❌ MISMATCH: Expected {expected_type}, found {network_type}")
            print(f"\nExpected configuration:")
            for key, value in EXPECTED_CONFIGS[expected_type].items():
                expected_val = value
                actual_val = first_config.get(key)
                match = "✓" if expected_val == actual_val else "✗"
                print(f"  {match} {key}: {actual_val} (expected {expected_val})")
    
    # Verify consistency across all tasks
    print(f"\nChecking consistency across {len(task_dirs)} tasks...")
    inconsistent_tasks = []
    for task_dir in task_dirs:
        config = load_task_config(task_dir)
        if config and config != first_config:
            inconsistent_tasks.append(task_dir.name)
    
    if inconsistent_tasks:
        print(f"❌ INCONSISTENT: {len(inconsistent_tasks)} tasks have different configs")
        print(f"   Inconsistent tasks: {', '.join(inconsistent_tasks[:5])}")
        if len(inconsistent_tasks) > 5:
            print(f"   ... and {len(inconsistent_tasks) - 5} more")
    else:
        print(f"✅ CONSISTENT: All {len(task_dirs)} tasks use identical network config")
    
    return {
        'status': 'success',
        'network_type': network_type,
        'num_tasks': len(task_dirs),
        'consistent': len(inconsistent_tasks) == 0,
        'inconsistent_tasks': inconsistent_tasks,
        'config': first_config,
        'matches_expected': network_type == expected_type if expected_type else None
    }

def verify_against_config_file(exp_dir: Path, config_file: Path) -> bool:
    """
    Verify experiment matches its config file.
    
    Args:
        exp_dir: Path to experiment directory
        config_file: Path to JSON config file
    
    Returns:
        True if matches, False otherwise
    """
    print(f"\n{'=' * 80}")
    print(f"Verifying against config file: {config_file.name}")
    print(f"{'=' * 80}")
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return False
    
    # Load expected config
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    expected_config = config_data.get('network_config', {})
    print("\nExpected from config file:")
    for key in ['hidden_size', 'num_layers', 'dropout', 'input_size']:
        print(f"  {key}: {expected_config.get(key)}")
    
    # Check experiment
    result = verify_experiment_directory(exp_dir)
    
    if result['status'] != 'success':
        return False
    
    actual_config = result['config']
    
    # Compare
    print("\n" + "-" * 80)
    print("COMPARISON:")
    print("-" * 80)
    
    all_match = True
    for key in ['hidden_size', 'num_layers', 'dropout', 'input_size']:
        expected = expected_config.get(key)
        actual = actual_config.get(key)
        match = expected == actual
        all_match = all_match and match
        symbol = "✅" if match else "❌"
        print(f"{symbol} {key:15} Expected: {expected:6}  Actual: {actual:6}")
    
    print("-" * 80)
    if all_match:
        print("✅ VERIFICATION PASSED: Experiment matches config file")
    else:
        print("❌ VERIFICATION FAILED: Experiment does NOT match config file")
    print("-" * 80)
    
    return all_match

def main():
    parser = argparse.ArgumentParser(description="Verify network configuration in experiment directories")
    parser.add_argument('exp_dir', type=str, help='Path to experiment directory')
    parser.add_argument('--config', type=str, help='Path to config JSON file to verify against')
    parser.add_argument('--expected', type=str, choices=['small_network', 'standard_network'],
                       help='Expected network type')
    
    args = parser.parse_args()
    
    exp_dir = Path(args.exp_dir)
    
    if args.config:
        config_file = Path(args.config)
        verify_against_config_file(exp_dir, config_file)
    else:
        verify_experiment_directory(exp_dir, args.expected)
    
    print()

if __name__ == '__main__':
    main()
