#!/usr/bin/env python3
"""
Generate standalone Network Compression plot from information flow analysis.

Usage:
    python analysis/plot_network_compression_standalone.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'results' / 'task_opponent_918988_analysis' / 'network_representation'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'task_opponent_918988_analysis' / 'plots'

# Load information flow data
df_info = pd.read_csv(DATA_DIR / 'information_flow.csv')

# Create standalone plot
fig, ax = plt.subplots(figsize=(8, 6))

games_order = ['PD', 'HD', 'SH']
colors = {'PD': '#1f77b4', 'HD': '#ff7f0e', 'SH': '#2ca02c'}

for game_abbrev in games_order:
    game_data = df_info[df_info['game'] == game_abbrev]
    ax.plot(game_data['opponent'], game_data['compression_ratio'], 
            marker='s', label=game_abbrev, linewidth=2.5, markersize=8,
            color=colors[game_abbrev])

ax.set_xlabel('Opponent Defection Probability', fontsize=12, fontweight='bold')
ax.set_ylabel('Compression Ratio (Input/Output bits)', fontsize=12, fontweight='bold')
ax.set_title('Network Compression: Input → Output', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(alpha=0.3)
ax.set_ylim([0, None])

plt.tight_layout()
output_file = OUTPUT_DIR / 'network_compression_standalone.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()
