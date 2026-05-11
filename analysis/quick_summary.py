"""Quick summary of experiment results"""
from pathlib import Path
import pickle

exp_path = Path("..") / "generalization_matrix_871188"

games = ["prisoners-dilemma", "hawk-dove", "stag-hunt"]
opp_ranges = ["very_low", "low", "mid", "high", "very_high"]

print("\n" + "="*100)
print("VANILLA RL EXPERIMENT SUMMARY - generalization_matrix_871188")
print("="*100)

for game_idx, game in enumerate(games):
    print(f"\n{'='*100}")
    print(f"{game.upper().replace('-', ' ')}")
    print(f"{'='*100}")
    
    for opp_idx, opp_range in enumerate(opp_ranges):
        task_id = game_idx * 5 + opp_idx
        
        # Find all runs for this task
        task_dirs = sorted(exp_path.glob(f"generalization_matrix_task_{task_id}_*"))
        n_runs = len(task_dirs)
        
        if not task_dirs:
            print(f"  Task {task_id:2d} ({opp_range:9s}): NO DATA")
            continue
        
        # Load test results from most recent run
        test_pkl = task_dirs[-1] / "results" / f"task_{task_id}_results.pkl"
        if test_pkl.exists():
            with open(test_pkl, 'rb') as f:
                data = pickle.load(f)
            
            # Extract evaluation summaries
            eval_summaries = data.get('evaluation_summaries', {})
            
            # Calculate average metrics across all test conditions
            if eval_summaries:
                avg_coop = sum(v['mean_cooperation_rate'] for v in eval_summaries.values()) / len(eval_summaries)
                avg_reward = sum(v['mean_reward'] for v in eval_summaries.values()) / len(eval_summaries)
                
                status = "✓✓✓" if n_runs >= 5 else "✓✓" if n_runs >= 3 else "⚠" if n_runs >= 2 else "⚠⚠"
                print(f"  Task {task_id:2d} ({opp_range:9s}): {n_runs} run(s) {status} | Coop: {avg_coop:.3f} | Reward: {avg_reward:.3f}")
            else:
                print(f"  Task {task_id:2d} ({opp_range:9s}): {n_runs} run(s) | NO EVALUATION SUMMARIES")
        else:
            print(f"  Task {task_id:2d} ({opp_range:9s}): {n_runs} run(s) | NO TEST RESULTS")

print("\n" + "="*100)
print("STATISTICAL POWER ASSESSMENT")
print("="*100)
print("  ✓✓✓  = 5+ runs (Strong statistical power)")
print("  ✓✓   = 3-4 runs (Moderate statistical power)")
print("  ⚠    = 2 runs (Weak statistical power)")
print("  ⚠⚠   = 1 run (No statistical power - single point)")
print("\n" + "="*100)
print("RECOMMENDATIONS FOR ADDITIONAL RUNS")
print("="*100)
print("\nHIGH PRIORITY (need 4+ more runs to reach 5 total):")
tasks_1_run = []
for i in range(15):
    task_dirs = sorted(exp_path.glob(f"generalization_matrix_task_{i}_*"))
    if len(task_dirs) == 1:
        game = games[i // 5]
        opp_range = opp_ranges[i % 5]
        tasks_1_run.append(f"  Task {i:2d}: {game} + {opp_range}")
        
if tasks_1_run:
    for task in tasks_1_run:
        print(task)
else:
    print("  None")

print("\nMEDIUM PRIORITY (need 2-3 more runs to reach 5 total):")
tasks_2_3_runs = []
for i in range(15):
    task_dirs = sorted(exp_path.glob(f"generalization_matrix_task_{i}_*"))
    if 2 <= len(task_dirs) <= 3:
        game = games[i // 5]
        opp_range = opp_ranges[i % 5]
        tasks_2_3_runs.append(f"  Task {i:2d}: {game} + {opp_range} ({len(task_dirs)} runs)")
        
if tasks_2_3_runs:
    for task in tasks_2_3_runs:
        print(task)
else:
    print("  None")

print("\n" + "="*100)
