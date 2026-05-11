"""
Test if t-SNE clusters are driven by training seed
==================================================

Creates t-SNE visualization colored by seed to determine if the observed
clustering is due to random initialization rather than game structure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
data_dir = Path("experiments/representation_coupling_complete/data")
tsne_embedding = np.load(data_dir / "tsne_weights.npy")
metadata = pd.read_csv(data_dir / "all_weight_metadata.csv")

# Create figure with 3 subplots: by seed, by game, by opponent
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

games = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
game_abbrev = {'prisoners-dilemma': 'PD', 'stag-hunt': 'SH', 'hawk-dove': 'HD'}
opp_ranges = ['very_low', 'low', 'mid', 'high', 'very_high']

# Plot 1: Colored by SEED
ax = axes[0]
seed_colors = {0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a', 3: '#984ea3', 4: '#ff7f00'}
for i, row in metadata.iterrows():
    ax.scatter(tsne_embedding[i, 0], tsne_embedding[i, 1],
              c=seed_colors[row['seed']], alpha=0.8, s=120,
              edgecolors='black', linewidth=0.8)

for seed, color in seed_colors.items():
    ax.scatter([], [], c=color, label=f'Seed {seed}', s=150, alpha=0.9)
ax.legend(fontsize=13, loc='best', title='Training Seed', framealpha=0.9)
ax.set_title('t-SNE: Weight Space Colored by Training Seed\n(Testing seed-driven clustering hypothesis)',
            fontsize=14, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax.grid(True, alpha=0.3)

# Plot 2: Colored by GAME (reference)
ax = axes[1]
game_colors = {'prisoners-dilemma': 'red', 'stag-hunt': 'blue', 'hawk-dove': 'green'}
for i, row in metadata.iterrows():
    ax.scatter(tsne_embedding[i, 0], tsne_embedding[i, 1],
              c=game_colors[row['game']], alpha=0.8, s=120,
              edgecolors='black', linewidth=0.8)

for game, color in game_colors.items():
    ax.scatter([], [], c=color, label=game_abbrev[game], s=150, alpha=0.9)
ax.legend(fontsize=13, loc='best', title='Training Game', framealpha=0.9)
ax.set_title('t-SNE: Weight Space Colored by Training Game\n(Original visualization)',
            fontsize=14, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax.grid(True, alpha=0.3)

# Plot 3: Colored by OPPONENT
ax = axes[2]
opp_colors = {'very_low': '#440154', 'low': '#31688e', 'mid': '#35b779',
             'high': '#fde724', 'very_high': '#ff6e3a'}
opp_labels = ['VL', 'L', 'M', 'H', 'VH']
for i, row in metadata.iterrows():
    ax.scatter(tsne_embedding[i, 0], tsne_embedding[i, 1],
              c=opp_colors[row['opponent']], alpha=0.8, s=120,
              edgecolors='black', linewidth=0.8)

for opp, color in opp_colors.items():
    ax.scatter([], [], c=color, label=opp_labels[opp_ranges.index(opp)], s=150, alpha=0.9)
ax.legend(fontsize=13, loc='best', title='Opponent Range', ncol=2, framealpha=0.9)
ax.set_title('t-SNE: Weight Space Colored by Opponent Range\n(Secondary factor)',
            fontsize=14, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("experiments/representation_coupling_complete/figures/tsne_seed_analysis.png", 
           dpi=150, bbox_inches='tight')
print("✓ Saved: tsne_seed_analysis.png")
plt.close()

# Quantitative analysis: Within-seed vs between-seed distances
weight_l2_dist = np.load(data_dir / "weight_l2_dist.npy")

within_seed_dists = []
between_seed_dists = []
within_game_dists = []
between_game_dists = []

n_models = len(metadata)
for i in range(n_models):
    for j in range(i+1, n_models):
        dist = weight_l2_dist[i, j]
        
        # Seed analysis
        if metadata.iloc[i]['seed'] == metadata.iloc[j]['seed']:
            within_seed_dists.append(dist)
        else:
            between_seed_dists.append(dist)
        
        # Game analysis
        if metadata.iloc[i]['game'] == metadata.iloc[j]['game']:
            within_game_dists.append(dist)
        else:
            between_game_dists.append(dist)

print("\n" + "="*80)
print("QUANTITATIVE CLUSTERING ANALYSIS")
print("="*80)

print(f"\n📊 SEED-DRIVEN CLUSTERING:")
print(f"  Within-seed distance:  {np.mean(within_seed_dists):.2f} ± {np.std(within_seed_dists):.2f}")
print(f"  Between-seed distance: {np.mean(between_seed_dists):.2f} ± {np.std(between_seed_dists):.2f}")
print(f"  Ratio (between/within): {np.mean(between_seed_dists)/np.mean(within_seed_dists):.3f}x")

print(f"\n📊 GAME-DRIVEN CLUSTERING:")
print(f"  Within-game distance:  {np.mean(within_game_dists):.2f} ± {np.std(within_game_dists):.2f}")
print(f"  Between-game distance: {np.mean(between_game_dists):.2f} ± {np.std(between_game_dists):.2f}")
print(f"  Ratio (between/within): {np.mean(between_game_dists)/np.mean(within_game_dists):.3f}x")

print("\n💡 INTERPRETATION:")
if np.mean(between_seed_dists) / np.mean(within_seed_dists) > 1.5:
    print("  ✓ SEED-DRIVEN: Clustering is primarily driven by random initialization!")
    print("    Models from the same seed are much more similar than different seeds.")
elif np.mean(between_game_dists) / np.mean(within_game_dists) > 1.5:
    print("  ✓ GAME-DRIVEN: Clustering is primarily driven by training game!")
    print("    Models from the same game are much more similar than different games.")
else:
    print("  ✓ MIXED: Both seed and game contribute to clustering structure.")

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Box plot comparison
ax = axes[0]
bp = ax.boxplot([within_seed_dists, between_seed_dists, within_game_dists, between_game_dists],
                labels=['Within\nSeed', 'Between\nSeeds', 'Within\nGame', 'Between\nGames'],
                patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
bp['boxes'][2].set_facecolor('lightgreen')
bp['boxes'][3].set_facecolor('lightyellow')
ax.set_ylabel('Weight Distance (L2)', fontsize=13, fontweight='bold')
ax.set_title('Clustering Factors: Seed vs Game',
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Histogram overlay
ax = axes[1]
ax.hist(within_seed_dists, bins=30, alpha=0.5, color='blue', label='Within Seed', density=True)
ax.hist(between_seed_dists, bins=30, alpha=0.5, color='red', label='Between Seeds', density=True)
ax.hist(within_game_dists, bins=30, alpha=0.5, color='green', label='Within Game', density=True, histtype='step', linewidth=2)
ax.hist(between_game_dists, bins=30, alpha=0.5, color='orange', label='Between Games', density=True, histtype='step', linewidth=2)
ax.set_xlabel('Weight Distance (L2)', fontsize=13, fontweight='bold')
ax.set_ylabel('Density', fontsize=13, fontweight='bold')
ax.set_title('Distribution of Weight Distances',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("experiments/representation_coupling_complete/figures/seed_vs_game_clustering.png",
           dpi=150, bbox_inches='tight')
print("✓ Saved: seed_vs_game_clustering.png")
plt.close()

# Save quantitative results
results = pd.DataFrame([{
    'within_seed_mean': np.mean(within_seed_dists),
    'within_seed_std': np.std(within_seed_dists),
    'between_seed_mean': np.mean(between_seed_dists),
    'between_seed_std': np.std(between_seed_dists),
    'seed_ratio': np.mean(between_seed_dists) / np.mean(within_seed_dists),
    'within_game_mean': np.mean(within_game_dists),
    'within_game_std': np.std(within_game_dists),
    'between_game_mean': np.mean(between_game_dists),
    'between_game_std': np.std(between_game_dists),
    'game_ratio': np.mean(between_game_dists) / np.mean(within_game_dists)
}])
results.to_csv("experiments/representation_coupling_complete/data/seed_vs_game_analysis.csv", index=False)
print("✓ Saved: seed_vs_game_analysis.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
