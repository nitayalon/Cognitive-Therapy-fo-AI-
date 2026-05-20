"""Directly plot from CSV to diagnose heatmap issue"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('Results/task_opponent_setup/unified_data/task_opponent_test_results.csv')

games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
game_names = {'prisoners-dilemma': 'PD', 'hawk-dove': 'HD', 'stag-hunt': 'SH'}
opponents = [0.1, 0.3, 0.5, 0.7, 0.9]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Normalized Reward: Testing on SH (Direct from CSV)', fontsize=16, fontweight='bold')

for j, train_game in enumerate(['prisoners-dilemma', 'hawk-dove', 'stag-hunt']):
    ax = axes[j]
    
    subset = df[
        (df['train_game'] == train_game) & 
        (df['test_game'] == 'stag-hunt')
    ]
    
    if len(subset) == 0:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Train: {game_names[train_game]}', fontsize=12, fontweight='bold')
        continue
    
    # Create pivot table
    pivot = subset.pivot_table(
        index='train_opponent',
        columns='test_opponent',
        values='normalized_reward',
        aggfunc='mean'
    )
    
    # Reindex
    pivot = pivot.reindex(index=opponents, columns=opponents)
    
    print(f"\n{game_names[train_game]} → SH pivot table:")
    print(pivot)
    
    # Plot heatmap
    sns.heatmap(pivot, ax=ax, cmap='RdYlGn', 
               vmin=0, vmax=1, cbar=True,
               annot=True, fmt='.2f', 
               cbar_kws={'shrink': 0.8})
    
    ax.set_title(f'Train: {game_names[train_game]} | Test: SH', fontsize=12, fontweight='bold')
    ax.set_xlabel('Test Opponent', fontsize=10)
    ax.set_ylabel('Train Opponent', fontsize=10)

plt.tight_layout()
plt.savefig('Results/task_opponent_setup/plots/DEBUG_all_to_sh_normalized.png', dpi=300, bbox_inches='tight')
print('\nSaved: DEBUG_all_to_sh_normalized.png')
plt.close()
