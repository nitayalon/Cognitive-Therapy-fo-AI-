#!/usr/bin/env python3
"""Verify checkpoint path discovery logic before running full test array."""

import os
import sys
from pathlib import Path
from glob import glob

def verify_checkpoint_discovery(training_job_id="888509"):
    """Test checkpoint discovery logic on sample conditions."""
    
    training_dir = Path(f"experiments/generalization_matrix_train_{training_job_id}")
    
    print("=" * 80)
    print("CHECKPOINT PATH VERIFICATION")
    print(f"Training Job ID: {training_job_id}")
    print(f"Training Directory: {training_dir}")
    print("=" * 80)
    
    if not training_dir.exists():
        print(f"❌ ERROR: Training directory not found: {training_dir}")
        return False
    
    print("✓ Training directory found")
    print()
    
    # Test sample conditions (matching the bash script)
    test_conditions = [
        (0, 0),   # condition 0, seed 0
        (0, 1),   # condition 0, seed 1
        (7, 2),   # condition 7, seed 2
        (14, 4),  # condition 14, seed 4 (last condition, last seed)
    ]
    
    success_count = 0
    total_tests = len(test_conditions)
    
    for training_condition_id, seed_id in test_conditions:
        print(f"Testing: Condition {training_condition_id}, Seed {seed_id}")
        print("-" * 40)
        
        # Step 1: Find condition directory
        condition_dir = training_dir / "training" / f"condition_{training_condition_id}_seed_{seed_id}"
        if not condition_dir.exists():
            print(f"  ❌ Condition directory not found: {condition_dir}")
            print()
            continue
        print(f"  ✓ Condition directory: {condition_dir}")
        
        # Step 2: Find experiment directory (generalization_matrix_*)
        exp_dirs = list(condition_dir.glob("generalization_matrix_*"))
        if not exp_dirs:
            print(f"  ❌ Experiment directory not found in {condition_dir}")
            print()
            continue
        exp_dir = exp_dirs[0]
        print(f"  ✓ Experiment directory: {exp_dir.name}")
        
        # Step 3: Find checkpoint file
        checkpoint_dir = exp_dir / "checkpoints"
        if not checkpoint_dir.exists():
            print(f"  ❌ Checkpoints directory not found: {checkpoint_dir}")
            print()
            continue
        
        checkpoint_files = list(checkpoint_dir.glob("*_final_checkpoint.pth"))
        if not checkpoint_files:
            print(f"  ❌ Checkpoint not found in {checkpoint_dir}")
            print()
            continue
        
        checkpoint_path = checkpoint_files[0]
        checkpoint_size = checkpoint_path.stat().st_size / (1024 * 1024)  # MB
        
        print(f"  ✓ Checkpoint found: {checkpoint_path.name}")
        print(f"    Path: {checkpoint_path}")
        print(f"    Size: {checkpoint_size:.2f} MB")
        
        success_count += 1
        print()
    
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Successful: {success_count}/{total_tests}")
    print()
    
    if success_count == total_tests:
        print("✅ All test cases passed!")
        print("✅ Checkpoint discovery logic is working correctly")
        print()
        print("You can now submit the full test jobs on HPC:")
        print(f"  TRAINING_JOB_ID={training_job_id} sbatch run_generalization_matrix_test.sh")
        print(f"  TRAINING_JOB_ID={training_job_id} sbatch run_generalization_matrix_test_part2.sh")
        return True
    else:
        print("❌ Some test cases failed")
        print("❌ Please fix issues before submitting full test array")
        return False

if __name__ == "__main__":
    training_job_id = sys.argv[1] if len(sys.argv) > 1 else "888509"
    success = verify_checkpoint_discovery(training_job_id)
    sys.exit(0 if success else 1)
