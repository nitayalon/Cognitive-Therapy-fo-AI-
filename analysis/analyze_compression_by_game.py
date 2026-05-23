#!/usr/bin/env python3
"""
Analyze Information Compression by Game - Experiment 918988
==========================================================

Shows how compression and information flow vary across opponents within each game.

Expected pattern:
- Games with fixed optimal strategies → high compression (deterministic)
- Games with opponent-dependent strategies → low compression (adaptive)

Usage:
    python analysis/analyze_compression_by_game.py

Author: Research Team
Date: May 20, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
results_dir = project_root / 'Results' / 'task_opponent_918988_analysis' / 'network_representation'
output_dir = project_root / 'Results' / 'task_opponent_918988_analysis' / 'compression_by_game'
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
info_flow = pd.read_csv(results_dir / 'information_flow.csv')
linear_probing = pd.read_csv(results_dir / 'linear_probing_results.csv')

print("="*80)
print("INFORMATION COMPRESSION ANALYSIS BY GAME")
print("="*80)

# ============================================================================
# ANALYSIS 1: Information Flow Metrics by Game
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

games = ['PD', 'HD', 'SH']
colors = {'PD': 'steelblue', 'HD': 'coral', 'SH': 'forestgreen'}

# Plot 1: Compression Ratio by Game
ax = axes[0, 0]
for game in games:
    game_data = info_flow[info_flow['game'] == game].sort_values('opponent')
    ax.plot(game_data['opponent'], game_data['compression_ratio'], 
            marker='o', label=game, color=colors[game], linewidth=2, markersize=8)
ax.set_xlabel('Opponent Defection Probability', fontsize=12, fontweight='bold')
ax.set_ylabel('Compression Ratio (Input/Output bits)', fontsize=12, fontweight='bold')
ax.set_title('Network Compression by Game', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_ylim([0, None])

# Plot 2: Action Entropy by Game
ax = axes[0, 1]
for game in games:
    game_data = info_flow[info_flow['game'] == game].sort_values('opponent')
    ax.plot(game_data['opponent'], game_data['action_entropy'], 
            marker='s', label=game, color=colors[game], linewidth=2, markersize=8)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Max entropy')
ax.set_xlabel('Opponent Defection Probability', fontsize=12, fontweight='bold')
ax.set_ylabel('Action Entropy (bits)', fontsize=12, fontweight='bold')
ax.set_title('Output Complexity by Game', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_ylim([0, 1.1])

# Plot 3: Information Retention by Game
ax = axes[1, 0]
for game in games:
    game_data = info_flow[info_flow['game'] == game].sort_values('opponent')
    # Cap retention at 100% for visualization (values >100% are MI estimation artifacts)
    retention_capped = game_data['info_retention'].clip(upper=1.0) * 100
    ax.plot(game_data['opponent'], retention_capped, 
            marker='^', label=game, color=colors[game], linewidth=2, markersize=8)
ax.set_xlabel('Opponent Defection Probability', fontsize=12, fontweight='bold')
ax.set_ylabel('Information Retention (%)', fontsize=12, fontweight='bold')
ax.set_title('Input → Hidden Information Flow', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_ylim([0, 100])

# Plot 4: Active Hidden Dimensions by Game
ax = axes[1, 1]
for game in games:
    game_data = info_flow[info_flow['game'] == game].sort_values('opponent')
    ax.plot(game_data['opponent'], game_data['active_dims_pct'] * 100, 
            marker='D', label=game, color=colors[game], linewidth=2, markersize=8)
ax.set_xlabel('Opponent Defection Probability', fontsize=12, fontweight='bold')
ax.set_ylabel('Active Hidden Dimensions (%)', fontsize=12, fontweight='bold')
ax.set_title('Information-Carrying Capacity', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_ylim([0, None])

plt.tight_layout()
plt.savefig(output_dir / 'compression_by_game_overview.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: compression_by_game_overview.png")
plt.close()

# ============================================================================
# ANALYSIS 2: Game-Specific Statistics
# ============================================================================

print("\n" + "="*80)
print("COMPRESSION STATISTICS BY GAME")
print("="*80)

for game in games:
    game_data = info_flow[info_flow['game'] == game]
    
    print(f"\n{game} (Prisoner's Dilemma)" if game == 'PD' else 
          f"\n{game} (Hawk-Dove)" if game == 'HD' else
          f"\n{game} (Stag Hunt)")
    print("-" * 40)
    print(f"Mean compression ratio:    {game_data['compression_ratio'].mean():>8.1f}x")
    print(f"Std compression ratio:     {game_data['compression_ratio'].std():>8.1f}x")
    print(f"Mean action entropy:       {game_data['action_entropy'].mean():>8.3f} bits")
    print(f"Std action entropy:        {game_data['action_entropy'].std():>8.3f} bits")
    print(f"Mean active dimensions:    {game_data['active_dims_pct'].mean()*100:>8.1f}%")
    print(f"Max active dimensions:     {game_data['active_dims_pct'].max()*100:>8.1f}%")
    
    # Find most and least compressed opponents
    most_compressed_idx = game_data['compression_ratio'].idxmax()
    least_compressed_idx = game_data['compression_ratio'].idxmin()
    
    most_compressed = game_data.loc[most_compressed_idx]
    least_compressed = game_data.loc[least_compressed_idx]
    
    print(f"\nMost compressed:  p={most_compressed['opponent']} ({most_compressed['compression_ratio']:.0f}x)")
    print(f"Least compressed: p={least_compressed['opponent']} ({least_compressed['compression_ratio']:.0f}x)")

# ============================================================================
# ANALYSIS 3: Heatmap of Compression Metrics
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

opponents = [0.1, 0.3, 0.5, 0.7, 0.9]

# Create matrices for heatmaps
compression_matrix = np.zeros((3, 5))
entropy_matrix = np.zeros((3, 5))
active_dims_matrix = np.zeros((3, 5))

for i, game in enumerate(games):
    for j, opp in enumerate(opponents):
        row = info_flow[(info_flow['game'] == game) & (info_flow['opponent'] == opp)]
        if len(row) > 0:
            compression_matrix[i, j] = row['compression_ratio'].values[0]
            entropy_matrix[i, j] = row['action_entropy'].values[0]
            active_dims_matrix[i, j] = row['active_dims_pct'].values[0] * 100

# Heatmap 1: Compression Ratio
sns.heatmap(compression_matrix, annot=True, fmt='.0f', cmap='YlOrRd', 
            xticklabels=[f'p={o}' for o in opponents],
            yticklabels=games, ax=axes[0], cbar_kws={'label': 'Compression (x)'})
axes[0].set_title('Compression Ratio by Game × Opponent', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Opponent Defection Probability', fontsize=11)
axes[0].set_ylabel('Game', fontsize=11)

# Heatmap 2: Action Entropy
sns.heatmap(entropy_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
            xticklabels=[f'p={o}' for o in opponents],
            yticklabels=games, ax=axes[1], cbar_kws={'label': 'Entropy (bits)'}, vmin=0, vmax=1)
axes[1].set_title('Action Entropy by Game × Opponent', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Opponent Defection Probability', fontsize=11)
axes[1].set_ylabel('Game', fontsize=11)

# Heatmap 3: Active Dimensions
sns.heatmap(active_dims_matrix, annot=True, fmt='.1f', cmap='Blues', 
            xticklabels=[f'p={o}' for o in opponents],
            yticklabels=games, ax=axes[2], cbar_kws={'label': 'Active Dims (%)'})
axes[2].set_title('Active Dimensions by Game × Opponent', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Opponent Defection Probability', fontsize=11)
axes[2].set_ylabel('Game', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'compression_heatmaps.png', dpi=150, bbox_inches='tight')
print(f"Saved: compression_heatmaps.png")
plt.close()

# ============================================================================
# ANALYSIS 4: Linear Probing Accuracy by Layer
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

# Prepare data
layers = linear_probing['layer'].unique()
tasks = ['game', 'opponent', 'condition']
task_labels = {'game': 'Game (3-way)', 'opponent': 'Opponent (5-way)', 'condition': 'Condition (15-way)'}

x = np.arange(len(layers))
width = 0.25

for i, task in enumerate(tasks):
    accuracies = []
    for layer in layers:
        acc = linear_probing[(linear_probing['layer'] == layer) & 
                            (linear_probing['task'] == task)]['accuracy'].values[0]
        accuracies.append(acc * 100)
    
    ax.bar(x + i*width, accuracies, width, label=task_labels[task], alpha=0.8)

ax.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Linear Probing: Information Preservation Across Network Layers', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(layers, fontsize=11)
ax.set_xlabel('Network Layer', fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='lower left')
ax.set_ylim([0, 105])
ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, task in enumerate(tasks):
    accuracies = []
    for layer in layers:
        acc = linear_probing[(linear_probing['layer'] == layer) & 
                            (linear_probing['task'] == task)]['accuracy'].values[0]
        accuracies.append(acc * 100)
    
    for j, (layer_x, acc) in enumerate(zip(x + i*width, accuracies)):
        ax.text(layer_x, acc + 2, f'{acc:.0f}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'layer_accuracy_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved: layer_accuracy_comparison.png")
plt.close()

# ============================================================================
# ANALYSIS 5: Compression Variability Summary
# ============================================================================

print("\n" + "="*80)
print("COMPRESSION VARIABILITY ANALYSIS")
print("="*80)

for game in games:
    game_data = info_flow[info_flow['game'] == game]
    
    compression_cv = game_data['compression_ratio'].std() / game_data['compression_ratio'].mean()
    entropy_cv = game_data['action_entropy'].std() / (game_data['action_entropy'].mean() + 1e-6)
    
    print(f"\n{game}:")
    print(f"  Compression variability (CV): {compression_cv:.2%}")
    print(f"  Entropy variability (CV):     {entropy_cv:.2%}")
    
    if game_data['action_entropy'].max() > 0.5:
        print(f"  ⚠️  Contains adaptive conditions (entropy > 0.5 bits)")
    else:
        print(f"  ✓  Fully deterministic across all opponents")

# Summary table
summary = []
for game in games:
    game_data = info_flow[info_flow['game'] == game]
    summary.append({
        'Game': game,
        'Mean_Compression': game_data['compression_ratio'].mean(),
        'Std_Compression': game_data['compression_ratio'].std(),
        'Mean_Entropy': game_data['action_entropy'].mean(),
        'Max_Entropy': game_data['action_entropy'].max(),
        'Deterministic_Conditions': (game_data['action_entropy'] < 0.01).sum(),
        'Adaptive_Conditions': (game_data['action_entropy'] > 0.5).sum()
    })

df_summary = pd.DataFrame(summary)
df_summary.to_csv(output_dir / 'compression_summary_by_game.csv', index=False)
print(f"\nSaved: compression_summary_by_game.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nOutput directory: {output_dir}")
print("\nGenerated files:")
print("  - compression_by_game_overview.png")
print("  - compression_heatmaps.png")
print("  - layer_accuracy_comparison.png")
print("  - compression_summary_by_game.csv")
