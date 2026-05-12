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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Import embedding analysis functions
from metric_4_embedding_analysis import (
    analyze_model_embeddings,
    analyze_full_network_weights,
    extract_hidden_state_representations,
    compute_representational_similarity
)

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

# Display name mapping
GAME_DISPLAY_NAMES = {
    'prisoners-dilemma': 'PD',
    'hawk-dove': 'HD',
    'stag-hunt': 'SH'
}

# Load training registry
REGISTRY_PATH = TRAIN_DIR.parent / 'seed_manifests' / 'MASTER_TRAINING_REGISTRY.csv'
REGISTRY = pd.read_csv(REGISTRY_PATH, on_bad_lines='skip').dropna()

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
        
        # Find checkpoint file (game-specific final checkpoint)
        checkpoint_path = model_dir / 'checkpoints' / f'{train_game}_final_checkpoint.pth'
        
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


def aggregate_across_seeds(df):
    """
    Aggregate the 15 models (3 games × 5 seeds) into 3 conditions.
    For each game, compute mean and SEM across seeds.
    """
    print("\n" + "="*70)
    print("AGGREGATING ACROSS SEEDS")
    print("="*70)
    
    # Identify numeric columns to aggregate
    exclude_cols = ['model_id', 'condition_id', 'seed', 'train_game']
    numeric_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Group by game and compute statistics
    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = ['mean', 'sem', 'std']
    
    grouped = df.groupby('train_game').agg(agg_dict)
    
    # Flatten multi-level column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    # Add sample size
    sample_counts = df.groupby('train_game').size().reset_index(name='n_seeds')
    aggregated_df = grouped.merge(sample_counts, on='train_game')
    
    print(f"  Aggregated {len(df)} models into {len(aggregated_df)} conditions")
    print(f"  Each condition has {aggregated_df['n_seeds'].iloc[0]} replicate seeds")
    
    return aggregated_df


def plot_weight_magnitude_comparison(df_agg):
    """Metric 4.1.1: Compare weight magnitudes across games (aggregated)."""
    print("\nGENERATING PLOT: Metric 4.1.1 - Weight Magnitude Comparison (Aggregated)")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    games_sorted = sorted(GAMES)
    df_agg_sorted = df_agg.set_index('train_game').loc[games_sorted].reset_index()
    
    x = np.arange(len(games_sorted))
    width = 0.35
    
    env_means = df_agg_sorted['env_total_mean'].values
    soc_means = df_agg_sorted['soc_total_mean'].values
    env_sems = df_agg_sorted['env_total_sem'].values
    soc_sems = df_agg_sorted['soc_total_sem'].values
    
    ax.bar(x - width/2, env_means, width, yerr=env_sems, label='Environmental',
           color='#2CA02C', alpha=0.8, capsize=5, error_kw={'linewidth': 2})
    ax.bar(x + width/2, soc_means, width, yerr=soc_sems, label='Social',
           color='#1F77B4', alpha=0.8, capsize=5, error_kw={'linewidth': 2})
    
    ax.set_xlabel('Training Game', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight Magnitude (L2 Norm)', fontsize=12, fontweight='bold')
    ax.set_title(f'Task Setup: Embedding Weight Magnitudes (n={int(df_agg_sorted["n_seeds"].iloc[0])} seeds/game)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([GAME_DISPLAY_NAMES[g] for g in games_sorted])
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


def plot_ablation_comparison(df_agg):
    """Metric 4.3: Ablation KLD comparison across games (aggregated)."""
    print("\nGENERATING PLOT: Metric 4.3 - Ablation KLD Comparison (Aggregated)")
    
    ablation_cols_mean = ['abl_env_payoff_mean', 'abl_env_round_mean', 'abl_soc_opp_action_mean',
                          'abl_soc_agent_action_mean', 'abl_soc_agent_reward_mean', 'abl_soc_opp_reward_mean']
    ablation_cols_sem = ['abl_env_payoff_sem', 'abl_env_round_sem', 'abl_soc_opp_action_sem',
                         'abl_soc_agent_action_sem', 'abl_soc_agent_reward_sem', 'abl_soc_opp_reward_sem']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Find global y-axis limits for synchronization
    all_vals = []
    for game in GAMES:
        game_data = df_agg[df_agg['train_game'] == game]
        all_vals.extend(game_data[ablation_cols_mean].values.flatten())
    global_max = max(all_vals) * 1.1
    
    for idx, game in enumerate(sorted(GAMES)):
        ax = axes[idx]
        game_data = df_agg[df_agg['train_game'] == game]
        
        means = game_data[ablation_cols_mean].values.flatten()
        sems = game_data[ablation_cols_sem].values.flatten()
        
        labels = ['Payoff', 'Round', 'Opp Act', 'Agt Act', 'Agt Rew', 'Opp Rew']
        x = np.arange(len(labels))
        
        colors = ['#2CA02C', '#2CA02C', '#1F77B4', '#1F77B4', '#1F77B4', '#1F77B4']
        ax.bar(x, means, yerr=sems, capsize=5, color=colors, alpha=0.7,
               edgecolor='black', linewidth=1, error_kw={'linewidth': 2})
        
        ax.set_title(f'Training Game: {GAME_DISPLAY_NAMES[game]}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('KL Divergence', fontsize=11)
        ax.set_ylim(0, global_max)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Task Setup: Embedding Importance (Ablation KLD)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_4.3_ablation_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file.name}")
    plt.close()


def statistical_analysis(df, df_agg):
    """
    Statistical tests to quantify differences in representations across games.
    """
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS: Game Effects on Learned Representations")
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
    
    # Test 2: Embedding importance differences
    print("\n2. Embedding Importance (Ablation KLD) by Game:")
    for embed in ['env_payoff', 'env_round', 'soc_opp_action',
                  'soc_agent_action', 'soc_agent_reward', 'soc_opp_reward']:
        col = f'abl_{embed}'
        games_abl = [df[df['train_game'] == game][col].values for game in GAMES]
        f_stat, p_val = stats.f_oneway(*games_abl)
        
        # Only report significant or near-significant differences
        if p_val < 0.1:
            means_str = ", ".join([f"{GAME_DISPLAY_NAMES[game]}={g.mean():.4f}"
                                   for game, g in zip(GAMES, games_abl)])
            print(f"   {embed}: F={f_stat:.3f}, p={p_val:.4f} ({means_str})")
    
    # Test 3: Within-game variance (reliability across seeds)
    print("\n3. Reliability Across Seeds (Coefficient of Variation):")
    for metric in ['social_ratio', 'env_total', 'soc_total']:
        cv_values = df_agg[f'{metric}_std'] / df_agg[f'{metric}_mean']
        mean_cv = cv_values.mean()
        print(f"   {metric}: CV = {mean_cv:.4f} (lower = more reliable)")


def full_network_representation_analysis(df_agg):
    """
    Analyze full network representations (LSTM + output heads) for the 3 game conditions.
    """
    print("\n" + "="*70)
    print("FULL NETWORK REPRESENTATION ANALYSIS")
    print("="*70)
    
    # Generate test states
    test_states = generate_test_states(n_samples=1000)
    
    # Collect full network weights and hidden states for each game
    network_data = []
    hidden_states_dict = {}
    
    print("\nAnalyzing full network for each game...")
    for game in GAMES:
        # Find one checkpoint for this game (use first available seed)
        game_rows = REGISTRY[REGISTRY['condition_id'] == list(CONDITION_TO_GAME.keys())[list(CONDITION_TO_GAME.values()).index(game)]]
        if len(game_rows) == 0:
            continue
        
        first_task = game_rows.iloc[0]['array_task_id']
        model_dirs = list(TRAIN_DIR.glob(f'whole_population_task_{first_task}_*'))
        
        if model_dirs:
            checkpoint_path = model_dirs[0] / 'checkpoints' / f'{game}_final_checkpoint.pth'
            
            if checkpoint_path.exists():
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
                network_data.append(network_weights)
                
                # Extract hidden states
                hidden_states = extract_hidden_state_representations(model, test_states, DEVICE)
                hidden_states_dict[game] = hidden_states
    
    print(f"  Analyzed {len(network_data)} games")
    
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
    
    games_list = list(hidden_states_dict.keys())
    n_games = len(games_list)
    similarity_matrix = np.zeros((n_games, n_games))
    
    for i, game1 in enumerate(games_list):
        for j, game2 in enumerate(games_list):
            if i <= j:
                sim = compute_representational_similarity(
                    hidden_states_dict[game1],
                    hidden_states_dict[game2],
                    method='cka'
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
    
    # Print pairwise similarities
    for i, game1 in enumerate(games_list):
        for j, game2 in enumerate(games_list):
            if i < j:
                print(f"   {GAME_DISPLAY_NAMES[game1]} vs {GAME_DISPLAY_NAMES[game2]}: {similarity_matrix[i, j]:.4f}")
    
    # 3. Visualize similarity matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    
    labels = [GAME_DISPLAY_NAMES[g] for g in games_list]
    
    im = ax.imshow(similarity_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(np.arange(n_games))
    ax.set_yticks(np.arange(n_games))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('CKA Similarity', fontsize=11)
    
    # Add text annotations
    for i in range(n_games):
        for j in range(n_games):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    ax.set_title('Representational Similarity Matrix (CKA on LSTM Hidden States)\n' +
                 'Task Setup: Full Network Representations Across Games',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_4.5_network_similarity_matrix.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_file.name}")
    plt.close()
    
    return network_df, similarity_matrix


def main():
    print("\n" + "="*70)
    print("TASK SETUP: EMBEDDING ANALYSIS (METRIC 4)")
    print("="*70)
    
    # Collect data (15 models: 3 games × 5 seeds)
    df = collect_embedding_data()
    
    # Aggregate across seeds (15 models → 3 conditions)
    df_agg = aggregate_across_seeds(df)
    
    # Statistical analysis
    statistical_analysis(df, df_agg)
    
    # Full network representation analysis (LSTM + output heads)
    network_df, similarity_matrix = full_network_representation_analysis(df_agg)
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS (3 Aggregated Conditions)")
    print("="*70)
    
    # Generate plots with aggregated data
    plot_weight_magnitude_comparison(df_agg)
    plot_social_ratio_by_game(df)  # Keep original to show distribution
    plot_ablation_comparison(df_agg)
    
    print("\n" + "="*70)
    print("EMBEDDING ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Data saved to: {DATA_DIR}")


if __name__ == '__main__':
    main()
