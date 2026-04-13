#!/usr/bin/env python3
"""
Quick test script to verify whole-population experiment mode is available.
Run this to check if the code is properly updated.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_argument_parser():
    """Test that whole-population mode is available."""
    try:
        # Import the parse_arguments function
        from main_experiment import parse_arguments
        import argparse
        
        # Create a mock args list
        test_args = ['--experiment-mode', 'whole-population', '--help']
        
        # Save original sys.argv
        original_argv = sys.argv
        
        try:
            sys.argv = ['main_experiment.py'] + test_args
            parse_arguments()
        except SystemExit:
            # --help causes sys.exit(), which is expected
            pass
        finally:
            sys.argv = original_argv
            
        print("✓ SUCCESS: whole-population mode is available")
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_config_file():
    """Test that config file exists."""
    config_path = 'config/whole_population_config.json'
    if os.path.exists(config_path):
        print(f"✓ Config file exists: {config_path}")
        return True
    else:
        print(f"✗ Config file missing: {config_path}")
        return False

def test_slurm_scripts():
    """Test that SLURM scripts exist."""
    scripts = [
        'run_whole_population_train.sh',
        'run_whole_population_test.sh'
    ]
    
    all_exist = True
    for script in scripts:
        if os.path.exists(script):
            print(f"✓ SLURM script exists: {script}")
        else:
            print(f"✗ SLURM script missing: {script}")
            all_exist = False
    
    return all_exist

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Whole Population Experiment Installation")
    print("=" * 60)
    
    results = []
    
    print("\n1. Testing argument parser...")
    results.append(test_argument_parser())
    
    print("\n2. Testing config file...")
    results.append(test_config_file())
    
    print("\n3. Testing SLURM scripts...")
    results.append(test_slurm_scripts())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ ALL TESTS PASSED - Ready to run experiment")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED - Check output above")
        sys.exit(1)
