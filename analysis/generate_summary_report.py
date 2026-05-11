"""Generate a summary report from the comprehensive comparison."""

import pandas as pd
import numpy as np
from pathlib import Path

# Load data
df = pd.read_csv("experiments/results/vanilla_vs_proto_tom_analysis/comprehensive_comparison.csv")

print("="*80)
print("VANILLA RL VS PROTO-TOM: MAIN FINDINGS")
print("="*80)
print()

print(f"Total comparisons: {len(df)} (16 tasks × 10 test conditions)")
print()

# Overall statistics
print("="*80)
print("1. OVERALL PERFORMANCE")
print("="*80)
print()

vanilla_mean = df['vanilla_reward'].mean()
proto_tom_mean = df['proto_tom_reward'].mean()
advantage = proto_tom_mean - vanilla_mean
percent_improvement = (advantage / abs(vanilla_mean)) * 100

print(f"Average Reward (across all conditions):")
print(f"  • Vanilla RL:       {vanilla_mean:.4f}")
print(f"  • Proto-ToM:        {proto_tom_mean:.4f}")
print(f"  • Advantage:        {advantage:+.4f} ({percent_improvement:+.2f}%)")
print(f"  • Effect:           {'✓ Proto-ToM BETTER' if advantage > 0 else '✗ Vanilla BETTER'}")
print()

# Cooperation
vanilla_coop = df['vanilla_cooperation'].mean()
proto_tom_coop = df['proto_tom_cooperation'].mean()
coop_diff = proto_tom_coop - vanilla_coop

print(f"Cooperation Rate:")
print(f"  • Vanilla RL:       {vanilla_coop:.4f} ({vanilla_coop*100:.1f}%)")
print(f"  • Proto-ToM:        {proto_tom_coop:.4f} ({proto_tom_coop*100:.1f}%)")
print(f"  • Difference:       {coop_diff:+.4f} ({coop_diff*100:+.1f} percentage points)")
print()

# Baseline vs Generalization
print("="*80)
print("2. BASELINE VS GENERALIZATION")
print("="*80)
print()

baseline_df = df[df['test_condition'] == 'baseline']
gen_df = df[df['test_condition'] != 'baseline']

print("In-Distribution (Baseline) Performance:")
print(f"  • Vanilla RL:       {baseline_df['vanilla_reward'].mean():.4f}")
print(f"  • Proto-ToM:        {baseline_df['proto_tom_reward'].mean():.4f}")
print(f"  • Advantage:        {baseline_df['reward_advantage'].mean():+.4f}")
print()

print("Out-of-Distribution (Generalization) Performance:")
print(f"  • Vanilla RL:       {gen_df['vanilla_reward'].mean():.4f}")
print(f"  • Proto-ToM:        {gen_df['proto_tom_reward'].mean():.4f}")
print(f"  • Advantage:        {gen_df['reward_advantage'].mean():+.4f}")
print()

# Generalization gap
vanilla_gen_gap = baseline_df['vanilla_reward'].mean() - gen_df['vanilla_reward'].mean()
proto_tom_gen_gap = baseline_df['proto_tom_reward'].mean() - gen_df['proto_tom_reward'].mean()

print("Generalization Gap (Baseline - OOD):")
print(f"  • Vanilla RL:       {vanilla_gen_gap:+.4f} (larger = worse generalization)")
print(f"  • Proto-ToM:        {proto_tom_gen_gap:+.4f}")
print(f"  • ToM Benefit:      {vanilla_gen_gap - proto_tom_gen_gap:+.4f} (smaller gap due to ToM)")
print()

# By generalization type
print("="*80)
print("3. PERFORMANCE BY GENERALIZATION TYPE")
print("="*80)
print()

same_game_df = df[df['test_condition'].str.contains('same_game', na=False)]
same_opp_df = df[df['test_condition'].str.contains('same_opponents', na=False)]
cross_df = df[(~df['test_condition'].str.contains('same', na=False)) & (df['test_condition'] != 'baseline')]

print("Same Game (New Opponents):")
if len(same_game_df) > 0:
    print(f"  • Vanilla RL:       {same_game_df['vanilla_reward'].mean():.4f}")
    print(f"  • Proto-ToM:        {same_game_df['proto_tom_reward'].mean():.4f}")
    print(f"  • Advantage:        {same_game_df['reward_advantage'].mean():+.4f}")
print()

print("New Game (Same Opponents):")
if len(same_opp_df) > 0:
    print(f"  • Vanilla RL:       {same_opp_df['vanilla_reward'].mean():.4f}")
    print(f"  • Proto-ToM:        {same_opp_df['proto_tom_reward'].mean():.4f}")
    print(f"  • Advantage:        {same_opp_df['reward_advantage'].mean():+.4f}")
print()

print("Cross-Generalization (New Game + New Opponents):")
if len(cross_df) > 0:
    print(f"  • Vanilla RL:       {cross_df['vanilla_reward'].mean():.4f}")
    print(f"  • Proto-ToM:        {cross_df['proto_tom_reward'].mean():.4f}")
    print(f"  • Advantage:        {cross_df['reward_advantage'].mean():+.4f}")
print()

# By training game
print("="*80)
print("4. PERFORMANCE BY TRAINING GAME")
print("="*80)
print()

games = df['training_game'].unique()
for game in sorted(games):
    game_df = df[df['training_game'] == game]
    print(f"{game.replace('-', ' ').title()}:")
    print(f"  • Vanilla RL:       {game_df['vanilla_reward'].mean():.4f}")
    print(f"  • Proto-ToM:        {game_df['proto_tom_reward'].mean():.4f}")
    print(f"  • Advantage:        {game_df['reward_advantage'].mean():+.4f}")
    print()

# Summary
print("="*80)
print("5. KEY FINDINGS")
print("="*80)
print()

findings = []

if advantage > 0:
    findings.append(f"✓ Proto-ToM shows {percent_improvement:.1f}% better average reward overall")
else:
    findings.append(f"✗ Vanilla RL shows {abs(percent_improvement):.1f}% better average reward overall")

baseline_advantage = baseline_df['reward_advantage'].mean()
gen_advantage = gen_df['reward_advantage'].mean()

if baseline_advantage > 0 and gen_advantage > 0:
    findings.append("✓ Proto-ToM outperforms vanilla in both baseline AND generalization")
elif baseline_advantage > 0:
    findings.append("⚠  Proto-ToM better at baseline but not generalization")
elif gen_advantage > 0:
    findings.append("⚠  Proto-ToM better at generalization but not baseline")
else:
    findings.append("✗ Vanilla RL outperforms Proto-ToM in most conditions")

if proto_tom_gen_gap < vanilla_gen_gap:
    findings.append(f"✓ Proto-ToM has {abs(vanilla_gen_gap - proto_tom_gen_gap):.3f} smaller generalization gap")
else:
    findings.append(f"✗ Proto-ToM has {abs(vanilla_gen_gap - proto_tom_gen_gap):.3f} larger generalization gap")

if same_game_df['reward_advantage'].mean() > same_opp_df['reward_advantage'].mean():
    findings.append("✓ Proto-ToM advantage is largest for opponent generalization")
else:
    findings.append("• Proto-ToM advantage is largest for game generalization")

for i, finding in enumerate(findings, 1):
    print(f"{i}. {finding}")

print()
print("="*80)

# Save summary
summary_text = f"""
VANILLA RL VS PROTO-TOM: EXECUTIVE SUMMARY

Overall Performance:
- Vanilla RL:  {vanilla_mean:.4f} reward, {vanilla_coop*100:.1f}% cooperation
- Proto-ToM:   {proto_tom_mean:.4f} reward, {proto_tom_coop*100:.1f}% cooperation  
- Advantage:   {advantage:+.4f} reward ({percent_improvement:+.1f}%)

Baseline vs Generalization:
- Baseline: Proto-ToM advantage = {baseline_advantage:+.4f}
- OOD:      Proto-ToM advantage = {gen_advantage:+.4f}

Generalization Gap:
- Vanilla RL:  {vanilla_gen_gap:.4f}
- Proto-ToM:   {proto_tom_gen_gap:.4f}
- Difference:  {vanilla_gen_gap - proto_tom_gen_gap:+.4f}

Key Findings:
"""
for finding in findings:
    summary_text += f"  {finding}\n"

with open("experiments/results/vanilla_vs_proto_tom_analysis/EXECUTIVE_SUMMARY.txt", "w") as f:
    f.write(summary_text)

print("✅ Summary saved to: experiments/results/vanilla_vs_proto_tom_analysis/EXECUTIVE_SUMMARY.txt")
