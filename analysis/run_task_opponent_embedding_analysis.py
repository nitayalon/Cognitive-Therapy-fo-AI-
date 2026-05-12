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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Add src directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
# Import embedding analysis functions
from metric_4_embedding_analysis import (
    analyze_model_embeddings,
    extract_embedding_activations,
    visualize_embedding_tsne,
    visualize_pca_variance,
    analyze_full_network_weights,
    extract_hidden_state_representations,
    compute_representational_similarity
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

# Display name mapping
GAME_DISPLAY_NAMES = {
    'prisoners-dilemma': 'PD',
    'hawk-dove': 'HD',
    'stag-hunt': 'SH'
}

# Load training registry
REGISTRY_PATH = TRAIN_DIR.parent / 'seed_manifests' / 'MASTER_TRAINING_REGISTRY.csv'
REGISTRY = pd.read_csv(REGISTRY_PATH, on_bad_lines='skip').dropna()

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
    
    # Iterate through condition directories
    condition_dirs = sorted(TRAIN_DIR.glob("condition_*_seed_*"))
    print(f"\nFound {len(condition_dirs)} condition/seed directories")
    
    for cond_dir in condition_dirs:
        # Parse condition_id and seed_id from directory name
        # Format: condition_X_seed_Y
        dir_name = cond_dir.name
        parts = dir_name.split('_')
        
        try:
            condition_id = int(parts[1])
            seed_id = int(parts[3])
        except (IndexError, ValueError):
            print(f"  WARNING: Could not parse directory name: {dir_name}")
            continue
        
        # Get game and opponent from condition_id
        if condition_id not in CONDITION_TO_GAME_OPP:
            print(f"  WARNING: Unknown condition_id: {condition_id}")
            continue
        
        train_game, train_opponent = CONDITION_TO_GAME_OPP[condition_id]
        
        # Find task directories
        task_dirs = sorted(cond_dir.glob("generalization_matrix_task_*"))
        
        for task_dir in task_dirs:
            # Parse task_id from directory name
            task_name = task_dir.name
            try:
                task_id = int(task_name.split('_')[3])
            except (IndexError, ValueError):
                print(f"  WARNING: Could not parse task_id from: {task_name}")
                continue
            
            # Find checkpoint file (game-specific final checkpoint)
            checkpoint_path = task_dir / 'checkpoints' / f'{train_game}_final_checkpoint.pth'
            
            if not checkpoint_path.exists():
                print(f"  WARNING: No checkpoint at {checkpoint_path}")
                continue
            
            # Analyze this model
            metadata = {
                'model_id': task_id,
                'condition_id': condition_id,
                'train_game': train_game,
                'train_opponent': train_opponent,
                'seed': seed_id
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


def aggregate_across_seeds(df):
    """
    Aggregate the 75 models (3 games × 5 opponents × 5 seeds) into 15 conditions.
    For each (game, opponent) pair, compute mean and SEM across seeds.
    """
    print("\n" + "="*70)
    print("AGGREGATING ACROSS SEEDS")
    print("="*70)
    
    # Identify numeric columns to aggregate
    exclude_cols = ['model_id', 'condition_id', 'seed', 'train_game', 'train_opponent', 'cluster']
    numeric_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Group by (game, opponent) and compute statistics
    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = ['mean', 'sem', 'std']
    
    grouped = df.groupby(['train_game', 'train_opponent']).agg(agg_dict)
    
    # Flatten multi-level column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    # Add sample size
    sample_counts = df.groupby(['train_game', 'train_opponent']).size().reset_index(name='n_seeds')
    aggregated_df = grouped.merge(sample_counts, on=['train_game', 'train_opponent'])
    
    print(f"  Aggregated {len(df)} models into {len(aggregated_df)} conditions")
    print(f"  Each condition has {aggregated_df['n_seeds'].iloc[0]} replicate seeds")
    
    return aggregated_df


#############################################################################
# Visualization Functions
#############################################################################

def plot_weight_magnitude_bars(df_agg):
    """
    Metric 4.1.1: Stacked bar chart of weight magnitudes with error bars.
    df_agg: Aggregated dataframe with _mean and _sem columns
    """
    print("\nGENERATING PLOT: Metric 4.1.1 - Weight Magnitude Bars (Aggregated)")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    games_sorted = sorted(GAMES)
    
    for idx, game in enumerate(games_sorted):
        ax = axes[idx]
        game_data = df_agg[df_agg['train_game'] == game].copy()
        game_data = game_data.sort_values('train_opponent')
        
        # Plot stacked bars with error bars
        x = np.arange(len(game_data))
        width = 0.6
        
        # Environmental bars
        ax.bar(x, game_data['env_total_mean'], width, 
               yerr=game_data['env_total_sem'],
               label='Environmental', color='#2CA02C', alpha=0.8,
               capsize=5, error_kw={'linewidth': 2})
        
        # Social bars (stacked on top)
        ax.bar(x, game_data['soc_total_mean'], width, 
               bottom=game_data['env_total_mean'],
               yerr=game_data['soc_total_sem'],
               label='Social', color='#1F77B4', alpha=0.8,
               capsize=5, error_kw={'linewidth': 2})
        
        ax.set_title(f'Training Game: {GAME_DISPLAY_NAMES[game]} (n={int(game_data["n_seeds"].iloc[0])} seeds/condition)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Weight Magnitude (L2 Norm)', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{o:.1f}' for o in game_data['train_opponent']])
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


def plot_social_ratio_heatmap(df_agg):
    """
    Metric 4.1.2: Heatmap of social ratios by game and opponent (aggregated data).
    """
    print("\nGENERATING PLOT: Metric 4.1.2 - Social Ratio Heatmap (Aggregated)")
    
    # Create pivot table from aggregated data
    pivot_matrix = df_agg.pivot(index='train_game', columns='train_opponent', values='social_ratio_mean')
    
    # Reorder rows and rename with display names
    pivot_matrix = pivot_matrix.reindex(['hawk-dove', 'prisoners-dilemma', 'stag-hunt'])
    pivot_matrix.index = [GAME_DISPLAY_NAMES[g] for g in pivot_matrix.index]
    
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
    
    # Get global y-axis limits
    global_min = plot_df['variance'].min()
    global_max = plot_df['variance'].max()
    y_margin = (global_max - global_min) * 0.05
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, game in enumerate(sorted(GAMES)):
        ax = axes[idx]
        game_data = plot_df[plot_df['train_game'] == game]
        
        sns.boxplot(data=game_data, x='embedding', y='variance', ax=ax,
                   palette='Set2', whis=1.5)
        
        ax.set_title(f'Training Game: {GAME_DISPLAY_NAMES[game]}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Activation Variance', fontsize=11)
        ax.set_xlabel('Embedding Type', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(global_min - y_margin, global_max + y_margin)
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
                          markerfacecolor='gray', markersize=10, label=GAME_DISPLAY_NAMES[g])
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
    row_labels = [f"{GAME_DISPLAY_NAMES[row['train_game']]}-{row['train_opponent']:.1f}" 
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
    
    # Calculate global y-axis limits
    global_min = df[ablation_cols].min().min()
    global_max = df[ablation_cols].max().max()
    y_margin = (global_max - global_min) * 0.05
    
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
            
            ax.set_title(f'{GAME_DISPLAY_NAMES[game]} | opp={opp:.1f}', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('KL Divergence', fontsize=9)
            ax.set_ylim(global_min - y_margin, global_max + y_margin)
            ax.grid(True, alpha=0.3, axis='y')
            
            plot_idx += 1
    
    plt.suptitle('Task-Opponent Setup: Embedding Importance by Training Condition',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_4.3.2_importance_by_condition.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file.name}")
    plt.close()


def plot_activation_clustering(df):
    """
    Metric 4.4: Cluster agents based on embedding importance (ablation KLD values).
    """
    print("\nGENERATING PLOT: Metric 4.4 - Embedding Importance Clustering")
    
    # Select ablation KLD features for clustering (importance of each embedding)
    abl_cols = ['abl_env_payoff', 'abl_env_round', 'abl_soc_opp_action',
                'abl_soc_agent_action', 'abl_soc_agent_reward', 'abl_soc_opp_reward']
    
    # Prepare data
    X = df[abl_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering (try different numbers of clusters)
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]
    
    # Create visualization
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Clusters colored by cluster ID
    ax1 = plt.subplot(1, 3, 1)
    cluster_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']
    
    for cluster_id in range(n_clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        ax1.scatter(cluster_data['pca1'], cluster_data['pca2'],
                   c=cluster_colors[cluster_id], label=f'Cluster {cluster_id}',
                   s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                   fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                   fontsize=11, fontweight='bold')
    ax1.set_title('Agents Clustered by Embedding Importance', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Color by training game
    ax2 = plt.subplot(1, 3, 2)
    game_colors = {'prisoners-dilemma': '#D62728', 'hawk-dove': '#2CA02C', 'stag-hunt': '#1F77B4'}
    game_markers = {'prisoners-dilemma': 'o', 'hawk-dove': 's', 'stag-hunt': '^'}
    
    for game in GAMES:
        game_data = df[df['train_game'] == game]
        ax2.scatter(game_data['pca1'], game_data['pca2'],
                   c=game_colors[game], marker=game_markers[game],
                   label=GAME_DISPLAY_NAMES[game], s=100, alpha=0.7, 
                   edgecolors='black', linewidth=1)
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                   fontsize=11, fontweight='bold')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                   fontsize=11, fontweight='bold')
    ax2.set_title('Colored by Training Game', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Color by training opponent
    ax3 = plt.subplot(1, 3, 3)
    opp_colors = {0.1: '#2E86AB', 0.3: '#54A8C7', 0.5: '#9E9E9E', 
                  0.7: '#E07A5F', 0.9: '#C1121F'}
    
    for opp in OPPONENTS:
        opp_data = df[df['train_opponent'] == opp]
        ax3.scatter(opp_data['pca1'], opp_data['pca2'],
                   c=opp_colors[opp], s=100, alpha=0.7,
                   edgecolors='black', linewidth=1, label=f'{opp:.1f}')
    
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                   fontsize=11, fontweight='bold')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                   fontsize=11, fontweight='bold')
    ax3.set_title('Colored by Training Opponent', fontsize=12, fontweight='bold')
    ax3.legend(title='Defection Prob', fontsize=9, title_fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Task-Opponent Setup: Clustering Based on Embedding Importance (Ablation KLD)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_4.4_activation_clustering.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file.name}")
    
    # Print cluster composition
    print(f"\n  Cluster Composition:")
    for cluster_id in range(n_clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        print(f"    Cluster {cluster_id} (n={len(cluster_data)}):")
        
        # Game distribution
        game_counts = cluster_data['train_game'].value_counts()
        for game, count in game_counts.items():
            print(f"      {GAME_DISPLAY_NAMES[game]}: {count} ({count/len(cluster_data)*100:.1f}%)")
        
        # Opponent distribution
        opp_mean = cluster_data['train_opponent'].mean()
        opp_std = cluster_data['train_opponent'].std()
        print(f"      Avg opponent: {opp_mean:.2f} ± {opp_std:.2f}")
        
        # Social ratio
        social_ratio_mean = cluster_data['social_ratio'].mean()
        print(f"      Avg social ratio: {social_ratio_mean:.3f}")
    
    plt.close()
    
    return df


def plot_activation_clustering_aggregated(df_agg):
    """
    Metric 4.4: Cluster 15 aggregated conditions based on embedding importance.
    """
    print("\nGENERATING PLOT: Metric 4.4 - Embedding Importance Clustering (15 Conditions)")
    
    # Select ablation KLD features for clustering
    abl_cols = ['abl_env_payoff_mean', 'abl_env_round_mean', 'abl_soc_opp_action_mean',
                'abl_soc_agent_action_mean', 'abl_soc_agent_reward_mean', 'abl_soc_opp_reward_mean']
    
    # Prepare data
    X = df_agg[abl_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering (fewer clusters for 15 points)
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_agg['cluster'] = kmeans.fit_predict(X_scaled)
    
    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df_agg['pca1'] = X_pca[:, 0]
    df_agg['pca2'] = X_pca[:, 1]
    
    # Create visualization
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Clusters colored by cluster ID
    ax1 = plt.subplot(1, 3, 1)
    cluster_colors = ['#E41A1C', '#377EB8', '#4DAF4A']
    
    for cluster_id in range(n_clusters):
        cluster_data = df_agg[df_agg['cluster'] == cluster_id]
        ax1.scatter(cluster_data['pca1'], cluster_data['pca2'],
                   c=cluster_colors[cluster_id], label=f'Cluster {cluster_id}',
                   s=200, alpha=0.7, edgecolors='black', linewidth=2)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                   fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                   fontsize=11, fontweight='bold')
    ax1.set_title('15 Conditions Clustered by Embedding Importance', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Color by training game
    ax2 = plt.subplot(1, 3, 2)
    game_colors = {'prisoners-dilemma': '#D62728', 'hawk-dove': '#2CA02C', 'stag-hunt': '#1F77B4'}
    game_markers = {'prisoners-dilemma': 'o', 'hawk-dove': 's', 'stag-hunt': '^'}
    
    for game in GAMES:
        game_data = df_agg[df_agg['train_game'] == game]
        ax2.scatter(game_data['pca1'], game_data['pca2'],
                   c=game_colors[game], marker=game_markers[game],
                   label=GAME_DISPLAY_NAMES[game], s=200, alpha=0.7, 
                   edgecolors='black', linewidth=2)
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                   fontsize=11, fontweight='bold')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                   fontsize=11, fontweight='bold')
    ax2.set_title('Colored by Training Game', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Color by training opponent
    ax3 = plt.subplot(1, 3, 3)
    opp_colors = {0.1: '#2E86AB', 0.3: '#54A8C7', 0.5: '#9E9E9E', 
                  0.7: '#E07A5F', 0.9: '#C1121F'}
    
    for opp in OPPONENTS:
        opp_data = df_agg[df_agg['train_opponent'] == opp]
        ax3.scatter(opp_data['pca1'], opp_data['pca2'],
                   c=opp_colors[opp], s=200, alpha=0.7,
                   edgecolors='black', linewidth=2, label=f'{opp:.1f}')
    
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                   fontsize=11, fontweight='bold')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                   fontsize=11, fontweight='bold')
    ax3.set_title('Colored by Training Opponent', fontsize=12, fontweight='bold')
    ax3.legend(title='Defection Prob', fontsize=9, title_fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Task-Opponent: Clustering 15 Conditions by Embedding Importance (Ablation KLD)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_4.4_activation_clustering_aggregated.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file.name}")
    
    # Print cluster composition
    print(f"\n  Cluster Composition (15 conditions):")
    for cluster_id in range(n_clusters):
        cluster_data = df_agg[df_agg['cluster'] == cluster_id]
        print(f"    Cluster {cluster_id} (n={len(cluster_data)} conditions):")
        
        # Game distribution
        game_counts = cluster_data['train_game'].value_counts()
        for game, count in game_counts.items():
            print(f"      {GAME_DISPLAY_NAMES[game]}: {count} ({count/len(cluster_data)*100:.1f}%)")
        
        # Opponent distribution  
        opp_mean = cluster_data['train_opponent'].mean()
        opp_std = cluster_data['train_opponent'].std()
        print(f"      Avg opponent: {opp_mean:.2f} ± {opp_std:.2f}")
        
        # Social ratio
        social_ratio_mean = cluster_data['social_ratio_mean'].mean()
        print(f"      Avg social ratio: {social_ratio_mean:.3f}")
    
    plt.close()
    
    return df_agg


def statistical_analysis(df, df_agg):
    """
    Statistical tests to quantify differences in representations as a function of setup.
    Tests:
    1. Two-way ANOVA: Effect of game and opponent on social ratio
    2. Effect sizes (eta-squared)
    3. Post-hoc comparisons
    """
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS: Setup Effects on Learned Representations")
    print("="*70)
    
    from scipy import stats
    
    # Test 1: Effect of game on social ratio
    print("\n1. Effect of Training Game on Social Ratio:")
    games = [df[df['train_game'] == game]['social_ratio'].values for game in GAMES]
    f_stat, p_val = stats.f_oneway(*games)
    print(f"   One-way ANOVA: F={f_stat:.3f}, p={p_val:.4f}")
    
    # Effect size (eta-squared)
    grand_mean = df['social_ratio'].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in games)
    ss_total = sum((df['social_ratio'] - grand_mean)**2)
    eta_sq = ss_between / ss_total
    print(f"   Effect size (eta²): {eta_sq:.4f}")
    
    # Test 2: Effect of opponent on social ratio
    print("\n2. Effect of Training Opponent on Social Ratio:")
    opponents = [df[df['train_opponent'] == opp]['social_ratio'].values for opp in OPPONENTS]
    f_stat, p_val = stats.f_oneway(*opponents)
    print(f"   One-way ANOVA: F={f_stat:.3f}, p={p_val:.4f}")
    
    ss_between = sum(len(o) * (o.mean() - grand_mean)**2 for o in opponents)
    eta_sq = ss_between / ss_total
    print(f"   Effect size (eta²): {eta_sq:.4f}")
    
    # Test 3: Ablation importance differences
    print("\n3. Embedding Importance (Ablation KLD) by Game:")
    for embed in ['env_payoff', 'env_round', 'soc_opp_action', 
                  'soc_agent_action', 'soc_agent_reward', 'soc_opp_reward']:
        col = f'abl_{embed}'
        games_abl = [df[df['train_game'] == game][col].values for game in GAMES]
        f_stat, p_val = stats.f_oneway(*games_abl)
        
        # Only report significant differences
        if p_val < 0.05:
            means_str = ", ".join([f"{GAME_DISPLAY_NAMES[game]}={g.mean():.4f}" 
                                   for game, g in zip(GAMES, games_abl)])
            print(f"   {embed}: F={f_stat:.3f}, p={p_val:.4f} ({means_str})")
    
    # Test 4: Within-condition variance (reliability across seeds)
    print("\n4. Reliability Across Seeds (Coefficient of Variation):")
    for metric in ['social_ratio', 'env_total', 'soc_total']:
        cv_values = df_agg[f'{metric}_std'] / df_agg[f'{metric}_mean']
        mean_cv = cv_values.mean()
        print(f"   {metric}: CV = {mean_cv:.4f} (lower = more reliable)")


def full_network_representation_analysis(df_agg):
    """
    Analyze full network representations (beyond just input embeddings).
    Computes:
    1. Weight magnitudes for all network components (LSTM, output heads)
    2. Representational similarity between conditions (CKA on hidden states)
    """
    print("\n" + "="*70)
    print("FULL NETWORK REPRESENTATION ANALYSIS")
    print("="*70)
    
    # Generate test states
    test_states = generate_test_states(n_samples=1000)
    
    # Collect full network weights and hidden states for each condition
    network_data = []
    hidden_states_dict = {}
    
    print("\nAnalyzing full network for each condition...")
    for idx, row in df_agg.iterrows():
        game = row['train_game']
        opponent = row['train_opponent']
        
        # Find one checkpoint for this condition (use seed 1)
        condition_dirs = list(TRAIN_DIR.glob("condition_*_seed_1"))
        checkpoint_path = None
        
        for cond_dir in condition_dirs:
            dir_name = cond_dir.name
            parts = dir_name.split('_')
            condition_id = int(parts[1])
            
            if condition_id in CONDITION_TO_GAME_OPP:
                cond_game, cond_opp = CONDITION_TO_GAME_OPP[condition_id]
                if cond_game == game and cond_opp == opponent:
                    task_dirs = list(cond_dir.glob("generalization_matrix_task_*"))
                    if task_dirs:
                        checkpoint_path = task_dirs[0] / 'checkpoints' / f'{game}_final_checkpoint.pth'
                        break
        
        if checkpoint_path and checkpoint_path.exists():
            # Load model
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            from cognitive_therapy_ai.network import GameLSTM
            model = GameLSTM(input_size=9, hidden_size=128, num_layers=2, dropout=0.1)
            model.load_state_dict(state_dict)
            model = model.to(DEVICE)
            model.eval()
            
            # Analyze full network weights
            network_weights = analyze_full_network_weights(model)
            network_weights['train_game'] = game
            network_weights['train_opponent'] = opponent
            network_data.append(network_weights)
            
            # Extract hidden states
            hidden_states = extract_hidden_state_representations(model, test_states, DEVICE)
            hidden_states_dict[(game, opponent)] = hidden_states
    
    print(f"  Analyzed {len(network_data)} conditions")
    
    # 1. Compare weight magnitudes across network components
    print("\n1. Network Component Weight Magnitudes:")
    network_df = pd.DataFrame(network_data)
    
    components = ['input_embeddings_total', 'lstm_input_hidden', 'lstm_hidden_hidden',
                  'policy_head', 'opponent_pred_head', 'value_head']
    
    for comp in components:
        mean_val = network_df[comp].mean()
        std_val = network_df[comp].std()
        print(f"   {comp}: {mean_val:.4f} ± {std_val:.4f}")
    
    # 2. Compute representational similarity (CKA) between all pairs
    print("\n2. Representational Similarity (CKA on Hidden States):")
    
    conditions = list(hidden_states_dict.keys())
    n_cond = len(conditions)
    similarity_matrix = np.zeros((n_cond, n_cond))
    
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            if i <= j:
                sim = compute_representational_similarity(
                    hidden_states_dict[cond1],
                    hidden_states_dict[cond2],
                    method='cka'
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
    
    # Summary statistics
    # Within-game similarity
    within_game_sims = []
    between_game_sims = []
    
    for i, (game1, opp1) in enumerate(conditions):
        for j, (game2, opp2) in enumerate(conditions):
            if i < j:
                sim = similarity_matrix[i, j]
                if game1 == game2:
                    within_game_sims.append(sim)
                else:
                    between_game_sims.append(sim)
    
    print(f"   Within-game similarity: {np.mean(within_game_sims):.4f} ± {np.std(within_game_sims):.4f}")
    print(f"   Between-game similarity: {np.mean(between_game_sims):.4f} ± {np.std(between_game_sims):.4f}")
    
    # Test significance
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(within_game_sims, between_game_sims)
    print(f"   t-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    # 3. Visualize similarity matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create labels
    labels = [f"{GAME_DISPLAY_NAMES[g]}-{o:.1f}" for g, o in conditions]
    
    im = ax.imshow(similarity_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(np.arange(n_cond))
    ax.set_yticks(np.arange(n_cond))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('CKA Similarity', fontsize=11)
    
    # Add text annotations
    for i in range(n_cond):
        for j in range(n_cond):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=6)
    
    ax.set_title('Representational Similarity Matrix (CKA on LSTM Hidden States)\n' +
                 'Full Network Representations Across Conditions',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_4.5_network_similarity_matrix.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_file.name}")
    plt.close()
    
    return network_df, similarity_matrix


#############################################################################
# Main Execution
#############################################################################

def main():
    print("\n" + "="*70)
    print("TASK-OPPONENT SETUP: EMBEDDING ANALYSIS (METRIC 4)")
    print("="*70)
    
    # Collect data (75 models: 3 games × 5 opponents × 5 seeds)
    df = collect_embedding_data()
    
    # Aggregate across seeds (75 models → 15 conditions)
    df_agg = aggregate_across_seeds(df)
    
    # Statistical analysis
    statistical_analysis(df, df_agg)
    
    # Full network representation analysis (LSTM + output heads)
    network_df, similarity_matrix = full_network_representation_analysis(df_agg)
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS (15 Aggregated Conditions)")
    print("="*70)
    
    # Generate plots with aggregated data
    plot_weight_magnitude_bars(df_agg)
    plot_social_ratio_heatmap(df_agg)
    plot_variance_boxplots(df)  # Keep original for now to show variability
    plot_social_vs_env_variance_scatter(df)  # Keep original
    plot_ablation_heatmap(df)  # Keep original
    plot_ablation_importance_by_condition(df)  # Keep original
    df_agg = plot_activation_clustering_aggregated(df_agg)  # New: 15 conditions
    
    print("\n" + "="*70)
    print("EMBEDDING ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Data saved to: {DATA_DIR}")


if __name__ == '__main__':
    main()
