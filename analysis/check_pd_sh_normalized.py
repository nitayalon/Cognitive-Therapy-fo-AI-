"""Check PD->SH normalized rewards"""
import pandas as pd

df = pd.read_csv('Results/task_opponent_setup/unified_data/task_opponent_test_results.csv')

print("PD → SH normalized rewards:")
pd_sh = df[(df['train_game']=='prisoners-dilemma') & (df['test_game']=='stag-hunt')]
print(pd_sh[['train_opponent', 'test_opponent', 'cooperation_rate', 'mean_reward', 'normalized_reward']].head(20))
print(f"\nUnique normalized_reward values: {sorted(pd_sh['normalized_reward'].unique())}")
print(f"Mean: {pd_sh['normalized_reward'].mean():.4f}")
