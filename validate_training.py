#!/usr/bin/env python3
"""Validate training outputs for generalization matrix experiment."""

import os
import json
import torch
from pathlib import Path

def validate_training_outputs(training_dir="experiments/training"):
    """Validate training outputs and checkpoints."""
    
    print("=" * 70)
    print("TRAINING OUTPUT VALIDATION REPORT")
    print("=" * 70)
    
    training_path = Path(training_dir)
    if not training_path.exists():
        print(f"❌ Training directory not found: {training_dir}")
        return False
    
    # Get all condition directories
    condition_dirs = sorted([d for d in training_path.iterdir() if d.is_dir()])
    
    print(f"\n📊 Found {len(condition_dirs)} condition directories")
    print("-" * 70)
    
    valid_checkpoints = 0
    invalid_checkpoints = 0
    empty_csvs = 0
    
    for cond_dir in condition_dirs:
        print(f"\n🔍 Checking: {cond_dir.name}")
        
        # Find experiment subdirectory
        exp_dirs = list(cond_dir.glob("generalization_matrix_*"))
        if not exp_dirs:
            print(f"   ❌ No experiment directory found")
            invalid_checkpoints += 1
            continue
        
        exp_dir = exp_dirs[0]
        
        # Check checkpoint
        checkpoint_dir = exp_dir / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("*_final_checkpoint.pth"))
        
        if not checkpoints:
            print(f"   ❌ No final checkpoint found")
            invalid_checkpoints += 1
            continue
        
        checkpoint_path = checkpoints[0]
        print(f"   ✓ Checkpoint: {checkpoint_path.name} ({checkpoint_path.stat().st_size / 1024:.1f} KB)")
        
        # Validate checkpoint can be loaded
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"   ✓ Checkpoint loads successfully")
            print(f"   ✓ Game: {checkpoint.get('game_name', 'NOT FOUND')}")
            print(f"   ✓ Epoch: {checkpoint.get('epoch', 'NOT FOUND')}")
            print(f"   ✓ Model params: {len(checkpoint['model_state_dict'])} keys")
            valid_checkpoints += 1
        except Exception as e:
            print(f"   ❌ Failed to load checkpoint: {e}")
            invalid_checkpoints += 1
            continue
        
        # Check results
        results_dir = exp_dir / "results"
        report_file = results_dir / f"training_task_0_report.json"
        csv_file = results_dir / f"training_task_0_metrics.csv"
        
        if report_file.exists():
            with open(report_file) as f:
                report = json.load(f)
            print(f"   ✓ Report: Task {report['task_id']}, {report['epochs']} epochs, "
                  f"converged={report['converged']}")
            print(f"      Game: {report['training_condition']['game']}, "
                  f"Opponents: {report['training_condition']['opponent_range']}")
        else:
            print(f"   ⚠️  No training report found")
        
        if csv_file.exists():
            csv_size = csv_file.stat().st_size
            if csv_size == 0:
                print(f"   ⚠️  Training CSV is empty (0 bytes)")
                empty_csvs += 1
            else:
                print(f"   ✓ Training CSV: {csv_size} bytes")
        else:
            print(f"   ⚠️  No training CSV found")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"✓ Valid checkpoints:    {valid_checkpoints}")
    print(f"❌ Invalid checkpoints:  {invalid_checkpoints}")
    print(f"⚠️  Empty CSVs:          {empty_csvs}")
    print()
    
    # Check if ready for testing
    if valid_checkpoints > 0 and invalid_checkpoints == 0:
        print("✅ STATUS: READY FOR TESTING PHASE")
        print()
        print("You have trained:", valid_checkpoints, "models")
        print("Expected:", "75 models (15 conditions × 5 seeds)")
        print()
        if valid_checkpoints < 75:
            print(f"⚠️  WARNING: Only {valid_checkpoints}/75 models trained.")
            print("   You can proceed with testing these models, but the full")
            print("   experiment requires all 75 models.")
        print()
        print("To run testing phase:")
        print("  1. Check your SLURM job ID from training logs")
        print("  2. Run: TRAINING_JOB_ID=<job_id> sbatch run_generalization_matrix_test.sh")
        return True
    else:
        print("❌ STATUS: NOT READY - Issues found")
        print(f"   Fix {invalid_checkpoints} invalid checkpoint(s) before proceeding")
        return False

if __name__ == "__main__":
    validate_training_outputs()
