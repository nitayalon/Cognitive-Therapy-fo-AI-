#!/usr/bin/env python3
"""Regenerate training CSV files from existing pickle results."""

import os
import pickle
import csv
from pathlib import Path

def save_training_metrics_to_csv(training_results: dict, csv_path: str):
    """Save training metrics to CSV for analysis."""
    
    # Extract epoch-level metrics from training results
    epoch_results = training_results.get('epoch_results', [])
    
    if not epoch_results:
        print(f"   ⚠️  No epoch results to save for {csv_path}")
        return
    
    rows = []
    for epoch_idx, epoch_data in enumerate(epoch_results):
        row = {
            'epoch': epoch_idx,
            'total_loss': epoch_data.get('total_loss', None),
            'rl_loss': epoch_data.get('rl_loss', None),
            'opponent_policy_loss': epoch_data.get('opponent_policy_loss', None),
            'opponent_reward_loss': epoch_data.get('opponent_reward_loss', None),
            'epoch_cumulative_reward': epoch_data.get('epoch_cumulative_reward', None),
            'epoch_average_cooperation_rate': epoch_data.get('epoch_average_cooperation_rate', None),
            'num_sessions': epoch_data.get('num_sessions', None),
        }
        rows.append(row)
    
    with open(csv_path, 'w', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    print(f"   ✓ Generated CSV with {len(rows)} rows: {Path(csv_path).name}")

def regenerate_training_csvs(training_dir="experiments/training"):
    """Regenerate all training CSV files from pickle results."""
    
    print("=" * 70)
    print("REGENERATING TRAINING CSV FILES")
    print("=" * 70)
    
    training_path = Path(training_dir)
    if not training_path.exists():
        print(f"❌ Training directory not found: {training_dir}")
        return
    
    # Get all condition directories
    condition_dirs = sorted([d for d in training_path.iterdir() if d.is_dir()])
    
    print(f"\n📊 Found {len(condition_dirs)} condition directories")
    print("-" * 70)
    
    regenerated = 0
    failed = 0
    
    for cond_dir in condition_dirs:
        print(f"\n🔄 Processing: {cond_dir.name}")
        
        # Find experiment subdirectory
        exp_dirs = list(cond_dir.glob("generalization_matrix_*"))
        if not exp_dirs:
            print(f"   ❌ No experiment directory found")
            failed += 1
            continue
        
        exp_dir = exp_dirs[0]
        
        # Find pickle file
        results_dir = exp_dir / "results"
        pkl_file = results_dir / "training_task_0_results.pkl"
        csv_file = results_dir / "training_task_0_metrics.csv"
        
        if not pkl_file.exists():
            print(f"   ❌ Pickle file not found")
            failed += 1
            continue
        
        # Load pickle and regenerate CSV
        try:
            with open(pkl_file, 'rb') as f:
                results = pickle.load(f)
            
            training_results = results.get('training_results', {})
            save_training_metrics_to_csv(training_results, str(csv_file))
            regenerated += 1
            
        except Exception as e:
            print(f"   ❌ Failed to process: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("REGENERATION SUMMARY")
    print("=" * 70)
    print(f"✓ Regenerated:  {regenerated}")
    print(f"❌ Failed:       {failed}")
    print()

if __name__ == "__main__":
    regenerate_training_csvs()
