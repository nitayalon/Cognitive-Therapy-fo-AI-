"""
Task Setup: Embedding Analysis (Metric 4)

Analyzes which embeddings (social vs environmental) are used by agents
trained on specific games (no specific opponent).

Research Question: Do general agents learn social strategies or Nash equilibrium?
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import pickle

# Import embedding analysis functions
from metric_4_embedding_analysis import analyze_model_embeddings

# Paths
BASE_DIR = Path(__file__).parent.parent
TRAIN_DIR = BASE_DIR / 'experiments' / 'whole_population_train_913310' / 'training'
OUTPUT_DIR = BASE_DIR / 'Results' / 'task_setup' / 'embedding_analysis'
PLOTS_DIR = OUTPUT_DIR / 'plots'
DATA_DIR = OUTPUT_DIR / 'unified_data'

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Constants
GAMES = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load training registry
REGISTRY_PATH = TRAIN_DIR.parent / 'seed_manifests' / 'MASTER_TRAINING_REGISTRY.csv'
REGISTRY = pd.read_csv(REGISTRY_PATH)

CONDITION_TO_GAME = {
    0: 'prisoners-dilemma',
    1: 'hawk-dove',
    2: 'stag-hunt'
}

TASK_TO_GAME = {}
for _, row in REGISTRY.iterrows():
    task_id = row['array_task_id']
    condition_id = row['condition_id']
    TASK_TO_GAME[task_id] = CONDITION_TO_GAME[condition_id]


def generate_test_states(n_samples=5000):
    """Generate diverse test states."""
    states = []
    
    payoffs = {
        'prisoners-dilemma': [3, 0, 5, 1],
        'hawk-dove': [2, 1, 3, 0],
        'stag-hunt': [4, 0, 3, 2]
    }
    
    for _ in range(n_samples):
        game = np.random.choice(GAMES)
        payoff = payoffs[game]
        round_num = np.random.randint(0, 100) / 100.0
        opp_action = np.random.choice([0.0, 1.0])
        agent_action = np.random.choice([0.0, 1.0])
        
        if agent_action == 0 and opp_action == 0:
            agent_reward = payoff[0]
            opp_reward = payoff[0]
        elif agent_action == 0 and opp_action == 1:
            agent_reward = payoff[1]
            opp_reward = payoff[2]
        elif agent_action == 1 and opp_action == 0:
            agent_reward = payoff[2]
            opp_reward = payoff[1]
        else:
            agent_reward = payoff[3]
            opp_reward = payoff[3]
        
        agent_reward = agent_reward / 5.0
        opp_reward = opp_reward / 5.0
        
        state = payoff + [round_num, opp_action, agent_action, agent_reward, opp_reward]
        states.append(state)
    
    return torch.tensor(states, dtype=torch.float32)


def collect_embedding_data():
    """Analyze embeddings for all task setup models."""
    print("="*70)
    print("COLLECTING EMBEDDING DATA")
    print("="*70)
    
    print("\nGenerating test states...")
    test_states = generate_test_states(n_samples=5000)
    print(f"  Created {len(test_states)} test states")
    
    all_results = []
    
    for _, row in REGISTRY.iterrows():
        task_id = row['array_task_id']
        condition_id = row['condition_id']
        seed = row['seed']
        
        train_game = CONDITION_TO_GAME[condition_id]
        
        # Find model directory
        model_dirs = list(TRAIN_DIR.glob(f'whole_population_task_{task_id}_*'))
        
        if not model_dirs:
            print(f"  WARNING: No directory for task {task_id}")
            continue
        
        model_dir = model_dirs[0]
        checkpoint_path = model_dir / 'checkpoints' / 'best_model.pth'
        
        if not checkpoint_path.exists():
            print(f"  WARNING: No checkpoint at {checkpoint_path}")
            continue
        
        metadata = {
            'model_id': task_id,
            'condition_id': condition_id,
            'train_game': train_game,
            'seed': seed
        }
        
        try:
            results = analyze_model_embeddings(
                checkpoint_path=checkpoint_path,
                test_states=test_states,
                model_metadata=metadata,
                device=DEVICE
            )
            all_results.append(results)
            
            if len(all_results) % 5 == 0:
                print(f"  Processed {len(all_results)} models...")
        
        except Exception as e:
            print(f"  ERROR analyzing {checkpoint_path}: {e}")
            continue
    
    df = pd.DataFrame(all_results)
    
    csv_path = DATA_DIR / 'embedding_analysis_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved embedding analysis data: {csv_path.name}")
    print(f"  Total models analyzed: {len(df)}")
    
    return df


def plot_weight_magnitude_comparison(df):
    """Metric 4.1.1: Compare weight magnitudes across games."""
    print("\nGENERATING PLOT: Metric 4.1.1 - Weight Magnitude Comparison")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    games_sorted = sorted(GAMES)
    x = np.arange(len(games_sorted))
    width = 0.35
    
    env_means = [df[df['train_game'] == g]['env_total'].mean() for g in games_sorted]
    soc_means = [df[df['train_game'] == g]['soc_total'].mean() for g in games_sorted]
    
    env_stds = [df[df['train_game'] == g]['env_total'].std() / np.sqrt(len(df[df['train_game'] == g])) 
                for g in games_sorted]
    soc_stds = [df[df['train_game'] == g]['soc_total'].std() / np.sqrt(len(df[df['train_game'] == g]))
                for g in games_sorted]
    
    ax.bar(x - width/2, env_means, width, yerr=env_stds, label='Environmental',
           color='#2CA02C', alpha=0.8, capsize=5)
    ax.bar(x + width/2, soc_means, width, yerr=soc_stds, label='Social',
           color='#1F77B4', alpha=0.8, capsize=5)
    
    ax.set_xlabel('Training Game', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight Magnitude (L2 Norm)', fontsize=12, fontweight='bold')
    ax.set_title('Task Setup: Embedding Weight Magnitudes', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(games_sorted)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = PLOTS_DIR / 'metric_4.1.1_weight_magnitude_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file.name}")
    plt.close()


def plot_social_ratio_by_game(df):
    """Metric 4.1.2: Social ratio distribution by game."""
    print("\nGENERATING PLOT: Metric 4.1.2 - Social Ratio by Game")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(data=df, x='train_game', y='social_ratio', ax=ax,
                palette='Set2', whis=1.5)
    
    ax.set_xlabel('Training Game', fontsize=12, fontweight='bold')
    ax.set_ylabel('Social Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Task Setup: Social Embedding Usage Ratio', fontsize=14, fontweight='bold')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Equal usage')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = PLOTS_DIR / 'metric_4.1.2_social_ratio_by_game.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file.name}")
    plt.close()


def plot_ablation_comparison(df):
    """Metric 4.3: Ablation KLD comparison across games."""
    print("\nGENERATING PLOT: Metric 4.3 - Ablation KLD Comparison")
    
    ablation_cols = ['abl_env_payoff', 'abl_env_round', 'abl_soc_opp_action',
                     'abl_soc_agent_action', 'abl_soc_agent_reward', 'abl_soc_opp_reward']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, game in enumerate(sorted(GAMES)):
        ax = axes[idx]
        game_data = df[df['train_game'] == game]
        
        means = [game_data[col].mean() for col in ablation_cols]
        stds = [game_data[col].std() / np.sqrt(len(game_data)) for col in ablation_cols]
        
        labels = ['Payoff', 'Round', 'Opp Act', 'Agt Act', 'Agt Rew', 'Opp Rew']
        x = np.arange(len(labels))
        
        colors = ['#2CA02C', '#2CA02C', '#1F77B4', '#1F77B4', '#1F77B4', '#1F77B4']
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7,
               edgecolor='black', linewidth=1)
        
        ax.set_title(f'Training Game: {game}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('KL Divergence', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Task Setup: Embedding Importance (Ablation KLD)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_4.3_ablation_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file.name}")
    plt.close()


def main():
    print("\n" + "="*70)
    print("TASK SETUP: EMBEDDING ANALYSIS (METRIC 4)")
    print("="*70)
    
    # Collect data
    df = collect_embedding_data()
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Generate plots
    plot_weight_magnitude_comparison(df)
    plot_social_ratio_by_game(df)
    plot_ablation_comparison(df)
    
    print("\n" + "="*70)
    print("EMBEDDING ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Data saved to: {DATA_DIR}")


if __name__ == '__main__':
    main()
