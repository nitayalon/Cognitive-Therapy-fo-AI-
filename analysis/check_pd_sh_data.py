"""Quick check for PD->SH test data availability"""
import pandas as pd

df = pd.read_csv('Results/task_opponent_setup/unified_data/task_opponent_test_results.csv')

print("Available columns:")
print(df.columns.tolist())
print()

print("="*60)
print("PD → SH Testing Data")
print("="*60)
pd_to_sh = df[(df['train_game']=='prisoners-dilemma') & (df['test_game']=='stag-hunt')]
print(f"Rows: {len(pd_to_sh)}")
if len(pd_to_sh) > 0:
    print(f"Mean reward: {pd_to_sh['mean_reward'].mean():.3f}")
    print(f"Mean cooperation: {pd_to_sh['cooperation_rate'].mean():.3f}")
    print(f"Sample rows:")
    print(pd_to_sh[['train_game', 'train_opponent', 'test_game', 'test_opponent', 'mean_reward', 'cooperation_rate']].head(10))
else:
    print("NO DATA FOUND!")

print("\n" + "="*60)
print("HD → SH Testing Data")
print("="*60)
hd_to_sh = df[(df['train_game']=='hawk-dove') & (df['test_game']=='stag-hunt')]
print(f"Rows: {len(hd_to_sh)}")
if len(hd_to_sh) > 0:
    print(f"Mean reward: {hd_to_sh['mean_reward'].mean():.3f}")
    print(f"Mean cooperation: {hd_to_sh['cooperation_rate'].mean():.3f}")
    print(f"Sample rows:")
    print(hd_to_sh[['train_game', 'train_opponent', 'test_game', 'test_opponent', 'mean_reward', 'cooperation_rate']].head(10))

print("\n" + "="*60)
print("All cross-game combinations:")
print("="*60)
cross_game = df[df['train_game'] != df['test_game']]
summary = cross_game.groupby(['train_game', 'test_game']).agg({'mean_reward': 'mean', 'cooperation_rate': 'mean'})
print(summary)
