"""
Task-Opponent Setup: Embedding Analysis (Metric 4)

Analyzes which embeddings (social vs environmental) are used by agents
trained on specific game-opponent combinations.

Research Question: Do specialized agents learn to integrate opponent behavior,
or do they only learn task structure (Nash equilibrium)?
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pickle

# Import embedding analysis functions
from metric_4_embedding_analysis import (
    analyze_model_embeddings,
    extract_embedding_activations,
    visualize_embedding_tsne,
    visualize_pca_variance
)

# Paths
BASE_DIR = Path(__file__).parent.parent
TRAIN_DIR = BASE_DIR / 'experiments' / 'generalization_matrix_train_913243' / 'training'
OUTPUT_DIR = BASE_DIR / 'Results' / 'task_opponent_setup' / 'embedding_analysis'
PLOTS_DIR = OUTPUT_DIR / 'plots'
DATA_DIR = OUTPUT_DIR / 'unified_data'

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Constants
GAMES = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
OPPONENTS = [0.1, 0.3, 0.5, 0.7, 0.9]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load training registry
REGISTRY_PATH = TRAIN_DIR.parent / 'seed_manifests' / 'MASTER_TRAINING_REGISTRY.csv'
REGISTRY = pd.read_csv(REGISTRY_PATH)

# Create condition mapping
CONDITION_TO_GAME_OPP = {}
for cond_id in range(15):
    game_idx = cond_id // 5
    opp_idx = cond_id % 5
    CONDITION_TO_GAME_OPP[cond_id] = (GAMES[game_idx], OPPONENTS[opp_idx])


#############################################################################
# Data Collection
#############################################################################

def generate_test_states(n_samples=5000):
    """
    Generate diverse test states covering all games and opponent behaviors.
    
    Returns:
        Tensor of shape (n_samples, 9)
    """
    states = []
    
    # Payoff matrices for each game
    payoffs = {
        'prisoners-dilemma': [3, 0, 5, 1],  # R, S, T, P
        'hawk-dove': [2, 1, 3, 0],
        'stag-hunt': [4, 0, 3, 2]
    }
    
    for _ in range(n_samples):
        # Random game
        game = np.random.choice(GAMES)
        payoff = payoffs[game]
        
        # Random round (0-99 normalized to 0-1)
        round_num = np.random.randint(0, 100) / 100.0
        
        # Random opponent action (0=cooperate, 1=defect)
        opp_action = np.random.choice([0.0, 1.0])
        
        # Random agent action
        agent_action = np.random.choice([0.0, 1.0])
        
        # Compute rewards based on actions
        if agent_action == 0 and opp_action == 0:  # Both cooperate
            agent_reward = payoff[0]  # R
            opp_reward = payoff[0]
        elif agent_action == 0 and opp_action == 1:  # Agent coop, opp defect
            agent_reward = payoff[1]  # S
            opp_reward = payoff[2]  # T
        elif agent_action == 1 and opp_action == 0:  # Agent defect, opp coop
            agent_reward = payoff[2]  # T
            opp_reward = payoff[1]  # S
        else:  # Both defect
            agent_reward = payoff[3]  # P
            opp_reward = payoff[3]
        
        # Normalize rewards to [0, 1]
        agent_reward = agent_reward / 5.0
        opp_reward = opp_reward / 5.0
        
        state = payoff + [round_num, opp_action, agent_action, agent_reward, opp_reward]
        states.append(state)
    
    return torch.tensor(states, dtype=torch.float32)


def collect_embedding_data():
    """
    Analyze embeddings for all trained models.
    
    Returns:
        DataFrame with embedding analysis results
    """
    print("="*70)
    print("COLLECTING EMBEDDING DATA")
    print("="*70)
    
    # Generate test states once
    print("\nGenerating test states...")
    test_states = generate_test_states(n_samples=5000)
    print(f"  Created {len(test_states)} test states")
    
    all_results = []
    
    # Iterate through all models
    for _, row in REGISTRY.iterrows():
        task_id = row['array_task_id']
        condition_id = row['condition_id']
        seed = row['seed']
        
        if condition_id not in CONDITION_TO_GAME_OPP:
            continue
        
        train_game, train_opponent = CONDITION_TO_GAME_OPP[condition_id]
        
        # Find model directory
        model_dirs = list(TRAIN_DIR.glob(f'condition_{condition_id}_seed_{seed}/generalization_matrix_task_{task_id}_*'))
        
        if not model_dirs:
            print(f"  WARNING: No directory for task {task_id}, condition {condition_id}, seed {seed}")
            continue
        
        model_dir = model_dirs[0]
        checkpoint_path = model_dir / 'checkpoints' / 'best_model.pth'
        
        if not checkpoint_path.exists():
            print(f"  WARNING: No checkpoint at {checkpoint_path}")
            continue
        
        # Analyze this model
        metadata = {
            'model_id': task_id,
            'condition_id': condition_id,
            'train_game': train_game,
            'train_opponent': train_opponent,
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
            
            if len(all_results) % 10 == 0:
                print(f"  Processed {len(all_results)} models...")
        
        except Exception as e:
            print(f"  ERROR analyzing {checkpoint_path}: {e}")
            continue
    
    df = pd.DataFrame(all_results)
    
    # Save
    csv_path = DATA_DIR / 'embedding_analysis_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved embedding analysis data: {csv_path.name}")
    print(f"  Total models analyzed: {len(df)}")
    
    return df


#############################################################################
# Visualization Functions
#############################################################################

def plot_weight_magnitude_bars(df):
    """
    Metric 4.1.1: Stacked bar chart of weight magnitudes.
    """
    print("\nGENERATING PLOT: Metric 4.1.1 - Weight Magnitude Bars")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    games_sorted = sorted(GAMES)
    
    for idx, game in enumerate(games_sorted):
        ax = axes[idx]
        game_data = df[df['train_game'] == game].copy()
        
        # Group by opponent, compute means
        grouped = game_data.groupby('train_opponent').agg({
            'env_total': 'mean',
            'soc_total': 'mean'
        }).reset_index()
        
        grouped = grouped.sort_values('train_opponent')
        
        # Plot stacked bars
        x = np.arange(len(grouped))
        width = 0.6
        
        ax.bar(x, grouped['env_total'], width, label='Environmental', 
               color='#2CA02C', alpha=0.8)
        ax.bar(x, grouped['soc_total'], width, bottom=grouped['env_total'],
               label='Social', color='#1F77B4', alpha=0.8)
        
        ax.set_title(f'Training Game: {game}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Weight Magnitude (L2 Norm)', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{o:.1f}' for o in grouped['train_opponent']])
        if idx == 2:
            ax.set_xlabel('Training Opponent Defection Probability', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Task-Opponent Setup: Embedding Weight Magnitudes', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_4.1.1_weight_magnitude_bars.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file.name}")
    plt.close()


def plot_social_ratio_heatmap(df):
    """
    Metric 4.1.2: Heatmap of social ratios by game and opponent.
    """
    print("\nGENERATING PLOT: Metric 4.1.2 - Social Ratio Heatmap")
    
    # Create pivot table
    pivot = df.groupby(['train_game', 'train_opponent'])['social_ratio'].mean().reset_index()
    pivot_matrix = pivot.pivot(index='train_game', columns='train_opponent', values='social_ratio')
    
    # Reorder rows
    pivot_matrix = pivot_matrix.reindex(['hawk-dove', 'prisoners-dilemma', 'stag-hunt'])
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Social Ratio'},
                vmin=0, vmax=1, linewidths=1, linecolor='gray')
    
    plt.title('Task-Opponent Setup: Social Embedding Usage Ratio', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Training Opponent Defection Probability', fontsize=11)
    plt.ylabel('Training Game', fontsize=11)
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_4.1.2_social_ratio_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file.name}")
    plt.close()


def plot_variance_boxplots(df):
    """
    Metric 4.2.1: Box plots of activation variance by embedding type.
    """
    print("\nGENERATING PLOT: Metric 4.2.1 - Activation Variance Boxplots")
    
    # Prepare data for plotting
    variance_cols = ['var_env_payoff', 'var_env_round', 'var_soc_opp_action',
                     'var_soc_agent_action', 'var_soc_agent_reward', 'var_soc_opp_reward']
    
    plot_data = []
    for col in variance_cols:
        for _, row in df.iterrows():
            plot_data.append({
                'embedding': col.replace('var_', '').replace('_', ' ').title(),
                'variance': row[col],
                'train_opponent': row['train_opponent'],
                'train_game': row['train_game']
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, game in enumerate(sorted(GAMES)):
        ax = axes[idx]
        game_data = plot_df[plot_df['train_game'] == game]
        
        sns.boxplot(data=game_data, x='embedding', y='variance', ax=ax,
                   palette='Set2', whis=1.5)
        
        ax.set_title(f'Training Game: {game}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Activation Variance', fontsize=11)
        ax.set_xlabel('Embedding Type', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Task-Opponent Setup: Embedding Activation Variance',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_4.2.1_variance_boxplots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file.name}")
    plt.close()


def plot_social_vs_env_variance_scatter(df):
    """
    Metric 4.2.2: Scatter plot of social vs environmental variance.
    """
    print("\nGENERATING PLOT: Metric 4.2.2 - Social vs Environmental Variance Scatter")
    
    opp_colors = {0.1: '#2E86AB', 0.3: '#54A8C7', 0.5: '#9E9E9E', 
                  0.7: '#E07A5F', 0.9: '#C1121F'}
    game_markers = {'prisoners-dilemma': 'o', 'hawk-dove': 's', 'stag-hunt': '^'}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for game in GAMES:
        for opp in OPPONENTS:
            game_opp_data = df[(df['train_game'] == game) & (df['train_opponent'] == opp)]
            
            if len(game_opp_data) == 0:
                continue
            
            ax.scatter(game_opp_data['var_env_total'], 
                      game_opp_data['var_soc_total'],
                      c=[opp_colors[opp]], marker=game_markers[game],
                      s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Diagonal line
    max_val = max(df['var_env_total'].max(), df['var_soc_total'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal usage')
    
    ax.set_xlabel('Environmental Embedding Variance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Social Embedding Variance', fontsize=12, fontweight='bold')
    ax.set_title('Task-Opponent Setup: Social vs Environmental Variance',
                 fontsize=14, fontweight='bold')
    
    # Create legends
    from matplotlib.lines import Line2D
    game_legend = [Line2D([0], [0], marker=game_markers[g], color='w',
                          markerfacecolor='gray', markersize=10, label=g)
                   for g in GAMES]
    opp_legend = [Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=opp_colors[o], markersize=10, label=f'opp={o:.1f}')
                  for o in OPPONENTS]
    
    first_legend = ax.legend(handles=game_legend, title='Game',
                            loc='upper left', fontsize=9)
    ax.add_artist(first_legend)
    ax.legend(handles=opp_legend + [Line2D([0], [0], color='k', linestyle='--', label='Equal usage')],
             title='Opponent', loc='lower right', fontsize=9)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_4.2.2_social_vs_env_variance_scatter.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file.name}")
    plt.close()


def plot_ablation_heatmap(df):
    """
    Metric 4.3.1: Heatmap of ablation KL divergences.
    """
    print("\nGENERATING PLOT: Metric 4.3.1 - Ablation KL Divergence Heatmap")
    
    ablation_cols = ['abl_env_payoff', 'abl_env_round', 'abl_soc_opp_action',
                     'abl_soc_agent_action', 'abl_soc_agent_reward', 'abl_soc_opp_reward']
    
    # Create matrix: rows = models (grouped), columns = embeddings
    plot_data = df[['train_game', 'train_opponent'] + ablation_cols].copy()
    plot_data = plot_data.sort_values(['train_game', 'train_opponent'])
    
    # Create row labels
    row_labels = [f"{row['train_game'][:2].upper()}-{row['train_opponent']:.1f}" 
                  for _, row in plot_data.iterrows()]
    
    # Get values matrix
    values = plot_data[ablation_cols].values
    
    # Rename columns for display
    col_labels = ['Payoff', 'Round', 'Opp Action', 'Agent Action', 
                  'Agent Reward', 'Opp Reward']
    
    # Plot
    plt.figure(figsize=(10, 20))
    sns.heatmap(values, cmap='Reds', cbar_kws={'label': 'KL Divergence'},
                xticklabels=col_labels, yticklabels=row_labels,
                linewidths=0.5, linecolor='gray', annot=False)
    
    plt.title('Task-Opponent Setup: Embedding Importance (Ablation KLD)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Embedding Type', fontsize=11)
    plt.ylabel('Training Condition', fontsize=11)
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_4.3.1_ablation_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file.name}")
    plt.close()


def plot_ablation_importance_by_condition(df):
    """
    Metric 4.3.2: Bar chart of average ablation KLD by condition.
    """
    print("\nGENERATING PLOT: Metric 4.3.2 - Ablation Importance by Condition")
    
    ablation_cols = ['abl_env_payoff', 'abl_env_round', 'abl_soc_opp_action',
                     'abl_soc_agent_action', 'abl_soc_agent_reward', 'abl_soc_opp_reward']
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    for game in sorted(GAMES):
        for opp in OPPONENTS:
            ax = axes[plot_idx]
            
            cond_data = df[(df['train_game'] == game) & (df['train_opponent'] == opp)]
            
            if len(cond_data) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                plot_idx += 1
                continue
            
            # Compute means and std errors
            means = [cond_data[col].mean() for col in ablation_cols]
            stds = [cond_data[col].std() / np.sqrt(len(cond_data)) for col in ablation_cols]
            
            labels = ['Payoff', 'Round', 'Opp Act', 'Agt Act', 'Agt Rew', 'Opp Rew']
            x = np.arange(len(labels))
            
            colors = ['#2CA02C', '#2CA02C', '#1F77B4', '#1F77B4', '#1F77B4', '#1F77B4']
            ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1)
            
            ax.set_title(f'{game[:2].upper()} | opp={opp:.1f}', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('KL Divergence', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            
            plot_idx += 1
    
    plt.suptitle('Task-Opponent Setup: Embedding Importance by Training Condition',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_4.3.2_importance_by_condition.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file.name}")
    plt.close()


#############################################################################
# Main Execution
#############################################################################

def main():
    print("\n" + "="*70)
    print("TASK-OPPONENT SETUP: EMBEDDING ANALYSIS (METRIC 4)")
    print("="*70)
    
    # Collect data
    df = collect_embedding_data()
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Generate all plots
    plot_weight_magnitude_bars(df)
    plot_social_ratio_heatmap(df)
    plot_variance_boxplots(df)
    plot_social_vs_env_variance_scatter(df)
    plot_ablation_heatmap(df)
    plot_ablation_importance_by_condition(df)
    
    print("\n" + "="*70)
    print("EMBEDDING ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Data saved to: {DATA_DIR}")


if __name__ == '__main__':
    main()
