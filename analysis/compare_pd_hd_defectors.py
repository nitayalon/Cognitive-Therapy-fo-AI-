"""Compare PD and HD defectors when tested on SH"""
import pandas as pd
import numpy as np

df = pd.read_csv('Results/task_opponent_setup/unified_data/task_opponent_test_results.csv')

print("="*70)
print("Comparing DEFECTING agents (cooperation = 0%) on SH test")
print("="*70)

# Filter for defectors on SH
pd_to_sh = df[(df['train_game']=='prisoners-dilemma') & (df['test_game']=='stag-hunt')]
hd_to_sh_defectors = df[
    (df['train_game']=='hawk-dove') & 
    (df['test_game']=='stag-hunt') &
    (df['train_opponent'].isin([0.1, 0.3, 0.5]))  # These are the defectors
]

print("\nPD → SH (all defect):")
print(f"  Rows: {len(pd_to_sh)}")
print(f"  Mean cooperation: {pd_to_sh['cooperation_rate'].mean():.4f}")
print(f"  Mean reward (raw): {pd_to_sh['mean_reward'].mean():.4f}")
print(f"  Mean normalized reward: {pd_to_sh['normalized_reward'].mean():.4f}")

print("\nHD (trained on 0.1-0.5) → SH (all defect):")
print(f"  Rows: {len(hd_to_sh_defectors)}")
print(f"  Mean cooperation: {hd_to_sh_defectors['cooperation_rate'].mean():.4f}")
print(f"  Mean reward (raw): {hd_to_sh_defectors['mean_reward'].mean():.4f}")
print(f"  Mean normalized reward: {hd_to_sh_defectors['normalized_reward'].mean():.4f}")

print("\n" + "="*70)
print("Breakdown by test opponent:")
print("="*70)

print("\nPD → SH by test opponent:")
pd_by_opp = pd_to_sh.groupby('test_opponent').agg({
    'cooperation_rate': 'mean',
    'mean_reward': 'mean', 
    'normalized_reward': 'mean'
})
print(pd_by_opp)

print("\nHD (0.1-0.5) → SH by test opponent:")
hd_by_opp = hd_to_sh_defectors.groupby('test_opponent').agg({
    'cooperation_rate': 'mean',
    'mean_reward': 'mean',
    'normalized_reward': 'mean'
})
print(hd_by_opp)

print("\n" + "="*70)
print("Sample raw data (first 10 rows each):")
print("="*70)
print("\nPD → SH:")
print(pd_to_sh[['train_opponent', 'test_opponent', 'cooperation_rate', 'mean_reward', 'normalized_reward']].head(10))

print("\nHD (0.1-0.5) → SH:")
print(hd_to_sh_defectors[['train_opponent', 'test_opponent', 'cooperation_rate', 'mean_reward', 'normalized_reward']].head(10))

# Check if normalized_reward values are consistent
print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)
if pd_to_sh['mean_reward'].mean() == hd_to_sh_defectors['mean_reward'].mean():
    print("✓ Raw rewards are IDENTICAL (both ~2.0 as expected)")
    if pd_to_sh['normalized_reward'].mean() != hd_to_sh_defectors['normalized_reward'].mean():
        print("✗ BUT normalized rewards DIFFER!")
        print("  → This suggests normalization was done with WRONG payoff matrices")
        print("  → The normalized_reward column was computed during testing, not analysis")
else:
    print("✗ Raw rewards DIFFER (unexpected!)")
