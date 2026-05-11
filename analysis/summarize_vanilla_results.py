"""
Quick summary of vanilla RL fast analysis results.
"""

import pickle
import pandas as pd
from pathlib import Path

# Load results
results_path = Path(__file__).parent / "experiments" / "results" / "vanilla_baseline_analysis" / "all_results.pkl"

with open(results_path, 'rb') as f:
    all_results = pickle.load(f)

print("="*80)
print("VANILLA RL FAST ANALYSIS - KEY FINDINGS")
print("="*80)
print(f"\nLoaded results for {len(all_results)} tasks\n")

# Analyze by game
games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt', 'battle-of-sexes']
opp_ranges = ['low', 'mid_low', 'mid_high', 'high']

print("\n1. FINAL TRAINING PERFORMANCE (by game and opponent range)")
print("-" * 80)
for game in games:
    print(f"\n{game.upper()}:")
    for opp_range in opp_ranges:
        # Find matching task
        for task_id, res in all_results.items():
            if res['training_game'] == game and res['training_opp_range'] == opp_range:
                if 'training' in res:
                    coop = res['training']['final_coop_rate']
                    reward_norm = res['training']['final_reward_norm']
                    entropy = res['training']['final_entropy']
                    print(f"  {opp_range:12s}: Reward={reward_norm:.4f}, Coop={coop:.4f}, Entropy={entropy:.4f}")
                break

print("\n\n2. BASELINE TEST PERFORMANCE (same game, same opponents)")
print("-" * 80)
for game in games:
    print(f"\n{game.upper()}:")
    for opp_range in opp_ranges:
        for task_id, res in all_results.items():
            if res['training_game'] == game and res['training_opp_range'] == opp_range:
                if 'test_summaries' in res and 'baseline' in res['test_summaries']:
                    baseline = res['test_summaries']['baseline']
                    print(f"  {opp_range:12s}: Reward={baseline['reward_norm']:.4f}, Coop={baseline['coop_rate']:.4f}")
                break

print("\n\n3. GENERALIZATION TO NEW OPPONENTS (same game)")
print("-" * 80)
# For each task, show average performance on other opponent ranges
for game in games:
    print(f"\n{game.upper()}:")
    for train_opp in opp_ranges:
        for task_id, res in all_results.items():
            if res['training_game'] == game and res['training_opp_range'] == train_opp:
                if 'test_summaries' not in res:
                    continue
                    
                # Calculate average reward on other opponent ranges
                other_rewards = []
                for cond_name, summary in res['test_summaries'].items():
                    if cond_name.startswith('same_game_'):
                        other_rewards.append(summary['reward_norm'])
                
                if other_rewards:
                    avg_gen = sum(other_rewards) / len(other_rewards)
                    baseline_reward = res['test_summaries']['baseline']['reward_norm']
                    gen_drop = baseline_reward - avg_gen
                    print(f"  Train {train_opp:12s}: Baseline={baseline_reward:.4f}, Avg_Gen={avg_gen:.4f}, Drop={gen_drop:.4f}")
                break

print("\n\n4. CROSS-GAME GENERALIZATION")
print("-" * 80)
# For each task, show average performance on different games
for game in games:
    print(f"\n{game.upper()} agents:")
    for train_opp in opp_ranges:
        for task_id, res in all_results.items():
            if res['training_game'] == game and res['training_opp_range'] == train_opp:
                if 'test_summaries' not in res:
                    continue
                    
                # Calculate average reward on different games
                cross_rewards = []
                for cond_name, summary in res['test_summaries'].items():
                    # Include both "_same_opponents" and cross-gen conditions
                    if cond_name not in ['baseline'] and not cond_name.startswith('same_game_'):
                        cross_rewards.append(summary['reward_norm'])
                
                if cross_rewards:
                    avg_cross = sum(cross_rewards) / len(cross_rewards)
                    baseline_reward = res['test_summaries']['baseline']['reward_norm']
                    cross_drop = baseline_reward - avg_cross
                    print(f"  Train {train_opp:12s}: Baseline={baseline_reward:.4f}, Avg_Cross={avg_cross:.4f}, Drop={cross_drop:.4f}")
                break

print("\n\n5. OVERALL STATISTICS")
print("-" * 80)
# Gather all baseline rewards
all_baseline_rewards = []
all_same_game_gen = []
all_cross_game_gen = []

for task_id, res in all_results.items():
    if 'test_summaries' not in res:
        continue
    
    baseline_reward = res['test_summaries']['baseline']['reward_norm']
    all_baseline_rewards.append(baseline_reward)
    
    # Same game generalization
    for cond_name, summary in res['test_summaries'].items():
        if cond_name.startswith('same_game_'):
            all_same_game_gen.append(summary['reward_norm'])
    
    # Cross-game generalization
    for cond_name, summary in res['test_summaries'].items():
        if cond_name not in ['baseline'] and not cond_name.startswith('same_game_'):
            all_cross_game_gen.append(summary['reward_norm'])

print(f"Baseline performance:         Mean={sum(all_baseline_rewards)/len(all_baseline_rewards):.4f}, Std={pd.Series(all_baseline_rewards).std():.4f}")
print(f"Same-game generalization:     Mean={sum(all_same_game_gen)/len(all_same_game_gen):.4f}, Std={pd.Series(all_same_game_gen).std():.4f}")
print(f"Cross-game generalization:    Mean={sum(all_cross_game_gen)/len(all_cross_game_gen):.4f}, Std={pd.Series(all_cross_game_gen).std():.4f}")

gen_drop = sum(all_baseline_rewards)/len(all_baseline_rewards) - sum(all_same_game_gen)/len(all_same_game_gen)
cross_drop = sum(all_baseline_rewards)/len(all_baseline_rewards) - sum(all_cross_game_gen)/len(all_cross_game_gen)
print(f"\nGeneralization drops:")
print(f"  Same-game:  {gen_drop:.4f} ({gen_drop/sum(all_baseline_rewards)*len(all_baseline_rewards)*100:.1f}%)")
print(f"  Cross-game: {cross_drop:.4f} ({cross_drop/sum(all_baseline_rewards)*len(all_baseline_rewards)*100:.1f}%)")

print("\n" + "="*80)
print("See plots in experiments/results/vanilla_baseline_analysis/")
print("="*80)
