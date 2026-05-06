"""Debug script to check grid alignment"""
import pickle
import pandas as pd
from pathlib import Path

# Load one sample test result to check the data structure
test_dir = Path("experiments/generalization_matrix_test_893509/testing")
sample_dir = test_dir / "model_5_test_cond_1"  # Model 5 = condition 1 (SH, very_low)

# Find the experiment directory
exp_dirs = list(sample_dir.glob("generalization_matrix_task_*"))
if exp_dirs:
    exp_dir = exp_dirs[0]
    pkl_files = list((exp_dir / "results").glob("*.pkl"))
    if pkl_files:
        with open(pkl_files[0], 'rb') as f:
            data = pickle.load(f)
        
        print("Sample model: model_5 (should be condition_id=1, seed=0)")
        print("Expected: Stag Hunt, Very Low")
        print()
        
        # Check all test conditions for this model
        eval_results = data.get('evaluation_results', {})
        print(f"Number of test conditions: {len(eval_results)}")
        print()
        
        for cond_key in sorted(eval_results.keys()):
            cond_data = eval_results[cond_key]
            test_cond = cond_data.get('test_condition', {})
            test_game = test_cond.get('game')
            test_opp_range = test_cond.get('opponent_range')
            print(f"{cond_key}: {test_game}, {test_opp_range}")

# Now check the games list order
games = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
opp_ranges = ['very_low', 'low', 'mid', 'high', 'very_high']

print("\n" + "="*60)
print("Expected grid structure:")
print("="*60)
print("Training conditions (rows in overall grid):")
for game_idx, game in enumerate(games):
    for opp_idx, opp_range in enumerate(opp_ranges):
        condition_id = game_idx * 5 + opp_idx
        print(f"  Row {game_idx}, Col {opp_idx}: {game}, {opp_range} (condition {condition_id})")
    print()

print("\nTest games order (rows in mini-heatmaps):")
for idx, game in enumerate(games):
    print(f"  Row {idx}: {game}")
