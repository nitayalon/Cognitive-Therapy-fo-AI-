"""Investigate HD→SH cooperation distribution"""
import pandas as pd
import numpy as np

df = pd.read_csv('Results/task_opponent_setup/unified_data/task_opponent_test_results.csv')

hd_to_sh = df[(df['train_game']=='hawk-dove') & (df['test_game']=='stag-hunt')]

print("="*70)
print("HD → SH Cooperation Distribution")
print("="*70)
print(f"Total rows: {len(hd_to_sh)}")
print(f"Overall mean cooperation: {hd_to_sh['cooperation_rate'].mean():.4f}")
print()

# Group by train_opponent and test_opponent
pivot = hd_to_sh.pivot_table(
    index='train_opponent',
    columns='test_opponent',
    values='cooperation_rate',
    aggfunc='mean'
)

print("Pivot table (train_opponent × test_opponent):")
print(pivot)
print()

# Check if there are multiple seeds per combination
print("Rows per (train_opponent, test_opponent) combination:")
counts = hd_to_sh.groupby(['train_opponent', 'test_opponent']).size()
print(counts)
print()

# Show all unique combinations with cooperation > 0
coop_data = hd_to_sh[hd_to_sh['cooperation_rate'] > 0]
print(f"Rows with cooperation > 0: {len(coop_data)}")
if len(coop_data) > 0:
    print(coop_data[['train_opponent', 'test_opponent', 'cooperation_rate', 'seed', 'model_id']])
