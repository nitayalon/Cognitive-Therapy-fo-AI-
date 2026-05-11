"""
Analyze vanilla RL baseline performance across all 16 training conditions.

This script analyzes the in-distribution (baseline) performance data
from the vanilla RL experiments and compares it with proto-ToM baselines.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Directories
EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
VANILLA_DIRS = [
    "vanilla_rl_array_835338_20260223_065612",  # tasks 0-7
    "vanilla_rl_array_835338_20260223_152600",  # task 8
    "vanilla_rl_array_835338_20260223_152654",  # task 9
    "vanilla_rl_array_835338_20260223_152715",  # task 10
    "vanilla_rl_array_835338_20260223_153050",  # task 11
    "vanilla_rl_array_835338_20260223_153053",  # task 12
    "vanilla_rl_array_835338_20260223_153234",  # task 13
    "vanilla_rl_array_835338_20260223_153306",  # task 14
    "vanilla_rl_array_835338_20260223_153933",  # task 15
]
PROTO_TOM_DIR = "generalization_matrix_834222"

# Training conditions mapping (from config)
GAMES = ["prisoners-dilemma", "hawk-dove", "stag-hunt", "battle-of-sexes"]
OPPONENT_RANGES = ["low", "mid_low", "mid_high", "high"]

TRAINING_CONDITIONS = {
    0: ("prisoners-dilemma", "low", [0.1, 0.3]),
    1: ("prisoners-dilemma", "mid_low", [0.3, 0.5]),
    2: ("prisoners-dilemma", "mid_high", [0.5, 0.7]),
    3: ("prisoners-dilemma", "high", [0.7, 0.9]),
    4: ("hawk-dove", "low", [0.1, 0.3]),
    5: ("hawk-dove", "mid_low", [0.3, 0.5]),
    6: ("hawk-dove", "mid_high", [0.5, 0.7]),
    7: ("hawk-dove", "high", [0.7, 0.9]),
    8: ("stag-hunt", "low", [0.1, 0.3]),
    9: ("stag-hunt", "mid_low", [0.3, 0.5]),
    10: ("stag-hunt", "mid_high", [0.5, 0.7]),
    11: ("stag-hunt", "high", [0.7, 0.9]),
    12: ("battle-of-sexes", "low", [0.1, 0.3]),
    13: ("battle-of-sexes", "mid_low", [0.3, 0.5]),
    14: ("battle-of-sexes", "mid_high", [0.5, 0.7]),
    15: ("battle-of-sexes", "high", [0.7, 0.9]),
}

def find_vanilla_task_dirs():
    """Find all vanilla RL task directories."""
    task_paths = {}
    for dir_name in VANILLA_DIRS:
        dir_path = EXPERIMENTS_DIR / dir_name
        if not dir_path.exists():
            continue
        for task_dir in dir_path.glob("vanilla_matrix_task*"):
            task_num = int(task_dir.name.replace("vanilla_matrix_task", ""))
            task_paths[task_num] = task_dir
    return task_paths

def find_proto_tom_task_dirs():
    """Find all proto-ToM task directories."""
    task_paths = {}
    proto_dir = EXPERIMENTS_DIR / PROTO_TOM_DIR
    if not proto_dir.exists():
        return task_paths
    
    for task_dir in proto_dir.glob("generalization_matrix_task_*"):
        # Extract task number
        name_parts = task_dir.name.split("_")
        task_num = int(name_parts[3])  # task_X_timestamp
        task_paths[task_num] = task_dir
    return task_paths

def load_task_results(task_path):
    """Load results for a task."""
    # Try different possible result file names
    possible_names = [
        f"task_{task_path.name.split('task')[-1].split('_')[0]}_results.pkl",
        "task_results.pkl",
        "matrix_results.pkl",
        "results.pkl"
    ]
    
    results_dir = task_path / "results"
    if not results_dir.exists():
        return None
    
    for name in possible_names:
        result_file = results_dir / name
        if result_file.exists():
            try:
                with open(result_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
    return None

def extract_baseline_metrics(results):
    """Extract baseline performance metrics from results."""
    if not results:
        return None
    
    # Look for baseline evaluation in different possible keys
    baseline_keys = ['eval_baseline', 'baseline', 'training_results']
    baseline_data = None
    
    for key in baseline_keys:
        if key in results:
            baseline_data = results[key]
            break
    
    if not baseline_data:
        # Try to find it in nested structure
        if 'evaluation_results' in results:
            eval_results = results['evaluation_results']
            if 'eval_baseline' in eval_results:
                baseline_data = eval_results['eval_baseline']
    
    if not baseline_data:
        return None
    
    # Extract metrics
    metrics = {}
    
    # Try different possible keys for each metric
    reward_keys = ['average_reward', 'avg_reward', 'mean_reward', 'reward']
    coop_keys = ['cooperation_rate', 'coop_rate', 'agent_cooperation_rate']
    
    for key in reward_keys:
        if key in baseline_data:
            metrics['reward'] = baseline_data[key]
            break
    
    for key in coop_keys:
        if key in baseline_data:
            metrics['cooperation_rate'] = baseline_data[key]
            break
    
    # Additional metrics if available
    if 'policy_entropy' in baseline_data:
        metrics['policy_entropy'] = baseline_data['policy_entropy']
    if 'opponent_prediction_accuracy' in baseline_data:
        metrics['opponent_prediction_accuracy'] = baseline_data['opponent_prediction_accuracy']
    
    return metrics if metrics else None

def create_performance_dataframe(vanilla_metrics, proto_tom_metrics):
    """Create comparative performance dataframe."""
    rows = []
    
    for task_id in range(16):
        game, opp_range, opp_probs = TRAINING_CONDITIONS[task_id]
        
        row = {
            'task_id': task_id,
            'game': game,
            'opponent_range': opp_range,
            'opponent_probs': f"{opp_probs[0]}-{opp_probs[1]}",
        }
        
        # Vanilla metrics
        if task_id in vanilla_metrics and vanilla_metrics[task_id]:
            for key, value in vanilla_metrics[task_id].items():
                row[f'vanilla_{key}'] = value
        
        # Proto-ToM metrics
        if task_id in proto_tom_metrics and proto_tom_metrics[task_id]:
            for key, value in proto_tom_metrics[task_id].items():
                row[f'proto_tom_{key}'] = value
        
        # Calculate difference (proto-ToM advantage)
        if f'vanilla_reward' in row and f'proto_tom_reward' in row:
            row['reward_advantage'] = row['proto_tom_reward'] - row['vanilla_reward']
        
        if f'vanilla_cooperation_rate' in row and f'proto_tom_cooperation_rate' in row:
            row['coop_advantage'] = row['proto_tom_cooperation_rate'] - row['vanilla_cooperation_rate']
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def plot_performance_comparison(df, output_dir):
    """Create visualizations comparing vanilla vs proto-ToM."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # 1. Reward comparison by game
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Baseline Performance: Vanilla RL vs Proto-ToM', fontsize=16, fontweight='bold')
    
    for idx, game in enumerate(GAMES):
        ax = axes[idx // 2, idx % 2]
        game_data = df[df['game'] == game]
        
        x = np.arange(len(game_data))
        width = 0.35
        
        if 'vanilla_reward' in game_data.columns and 'proto_tom_reward' in game_data.columns:
            vanilla_rewards = game_data['vanilla_reward'].values
            proto_tom_rewards = game_data['proto_tom_reward'].values
            
            ax.bar(x - width/2, vanilla_rewards, width, label='Vanilla RL', alpha=0.8)
            ax.bar(x + width/2, proto_tom_rewards, width, label='Proto-ToM', alpha=0.8)
            
            ax.set_xlabel('Opponent Range', fontweight='bold')
            ax.set_ylabel('Average Reward', fontweight='bold')
            ax.set_title(game.replace('-', ' ').title(), fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(game_data['opponent_range'].values)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_reward_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Cooperation rate comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cooperation Rates: Vanilla RL vs Proto-ToM', fontsize=16, fontweight='bold')
    
    for idx, game in enumerate(GAMES):
        ax = axes[idx // 2, idx % 2]
        game_data = df[df['game'] == game]
        
        x = np.arange(len(game_data))
        width = 0.35
        
        if 'vanilla_cooperation_rate' in game_data.columns and 'proto_tom_cooperation_rate' in game_data.columns:
            vanilla_coop = game_data['vanilla_cooperation_rate'].values
            proto_tom_coop = game_data['proto_tom_cooperation_rate'].values
            
            ax.bar(x - width/2, vanilla_coop, width, label='Vanilla RL', alpha=0.8)
            ax.bar(x + width/2, proto_tom_coop, width, label='Proto-ToM', alpha=0.8)
            
            ax.set_xlabel('Opponent Range', fontweight='bold')
            ax.set_ylabel('Cooperation Rate', fontweight='bold')
            ax.set_title(game.replace('-', ' ').title(), fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(game_data['opponent_range'].values)
            ax.legend()
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_cooperation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Proto-ToM advantage heatmap
    if 'reward_advantage' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        pivot_df = df.pivot(index='game', columns='opponent_range', values='reward_advantage')
        pivot_df = pivot_df[['low', 'mid_low', 'mid_high', 'high']]  # Ensure correct order
        
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                    cbar_kws={'label': 'Reward Advantage (Proto-ToM - Vanilla)'},
                    linewidths=0.5, ax=ax)
        
        ax.set_title('Proto-ToM Advantage Over Vanilla RL (Baseline Performance)', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Opponent Range', fontweight='bold')
        ax.set_ylabel('Game', fontweight='bold')
        
        # Format y-axis labels
        yticklabels = [label.get_text().replace('-', ' ').title() for label in ax.get_yticklabels()]
        ax.set_yticklabels(yticklabels, rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'proto_tom_advantage_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\n✅ Plots saved to {output_dir}/")

def print_summary_statistics(df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("BASELINE PERFORMANCE SUMMARY STATISTICS")
    print("="*80 + "\n")
    
    # Overall statistics
    if 'vanilla_reward' in df.columns and 'proto_tom_reward' in df.columns:
        vanilla_mean = df['vanilla_reward'].mean()
        proto_tom_mean = df['proto_tom_reward'].mean()
        
        print("AVERAGE REWARD:")
        print(f"  Vanilla RL:  {vanilla_mean:.4f} (±{df['vanilla_reward'].std():.4f})")
        print(f"  Proto-ToM:   {proto_tom_mean:.4f} (±{df['proto_tom_reward'].std():.4f})")
        print(f"  Difference:  {proto_tom_mean - vanilla_mean:.4f}")
        print(f"  % Change:    {((proto_tom_mean - vanilla_mean) / abs(vanilla_mean)) * 100:.2f}%")
        print()
    
    if 'vanilla_cooperation_rate' in df.columns and 'proto_tom_cooperation_rate' in df.columns:
        vanilla_coop_mean = df['vanilla_cooperation_rate'].mean()
        proto_tom_coop_mean = df['proto_tom_cooperation_rate'].mean()
        
        print("COOPERATION RATE:")
        print(f"  Vanilla RL:  {vanilla_coop_mean:.4f} (±{df['vanilla_cooperation_rate'].std():.4f})")
        print(f"  Proto-ToM:   {proto_tom_coop_mean:.4f} (±{df['proto_tom_cooperation_rate'].std():.4f})")
        print(f"  Difference:  {proto_tom_coop_mean - vanilla_coop_mean:.4f}")
        print()
    
    # By game
    print("PERFORMANCE BY GAME:")
    print("-" * 80)
    for game in GAMES:
        game_data = df[df['game'] == game]
        print(f"\n{game.replace('-', ' ').title()}:")
        
        if 'vanilla_reward' in game_data.columns and 'proto_tom_reward' in game_data.columns:
            vanilla_game_mean = game_data['vanilla_reward'].mean()
            proto_tom_game_mean = game_data['proto_tom_reward'].mean()
            advantage = proto_tom_game_mean - vanilla_game_mean
            
            print(f"  Vanilla RL reward:  {vanilla_game_mean:.4f}")
            print(f"  Proto-ToM reward:   {proto_tom_game_mean:.4f}")
            print(f"  Proto-ToM advantage: {advantage:.4f} ({'↑' if advantage > 0 else '↓'})")
    
    print("\n" + "="*80)

def main():
    print("="*80)
    print("VANILLA RL BASELINE PERFORMANCE ANALYSIS")
    print("="*80)
    print()
    
    # Find task directories
    print("Loading data...")
    vanilla_paths = find_vanilla_task_dirs()
    proto_tom_paths = find_proto_tom_task_dirs()
    
    print(f"  Found {len(vanilla_paths)} vanilla RL tasks")
    print(f"  Found {len(proto_tom_paths)} proto-ToM tasks")
    print()
    
    # Load results
    vanilla_metrics = {}
    proto_tom_metrics = {}
    
    for task_id in range(16):
        if task_id in vanilla_paths:
            results = load_task_results(vanilla_paths[task_id])
            metrics = extract_baseline_metrics(results)
            if metrics:
                vanilla_metrics[task_id] = metrics
        
        if task_id in proto_tom_paths:
            results = load_task_results(proto_tom_paths[task_id])
            metrics = extract_baseline_metrics(results)
            if metrics:
                proto_tom_metrics[task_id] = metrics
    
    print(f"Successfully loaded:")
    print(f"  Vanilla RL: {len(vanilla_metrics)}/16 tasks")
    print(f"  Proto-ToM:  {len(proto_tom_metrics)}/16 tasks")
    print()
    
    # Create dataframe
    df = create_performance_dataframe(vanilla_metrics, proto_tom_metrics)
    
    # Save dataframe
    output_dir = Path("experiments/results/vanilla_baseline_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / "baseline_performance_comparison.csv", index=False)
    print(f"✅ Saved comparison data to {output_dir}/baseline_performance_comparison.csv")
    print()
    
    # Print statistics
    print_summary_statistics(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_performance_comparison(df, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nFiles generated:")
    print("  - baseline_performance_comparison.csv")
    print("  - baseline_reward_comparison.png")
    print("  - baseline_cooperation_comparison.png")
    print("  - proto_tom_advantage_heatmap.png")
    print()

if __name__ == "__main__":
    main()
