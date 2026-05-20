"""
Task-Opponent Setup Analysis Script - Experiment 918988/918989
===============================================================
Analyzes generalization matrix experiments with 3 games × 5 opponents (15 conditions).
Updated for reduced network architecture (128 hidden units, 2 layers).

ETL Pipeline:
1. Load master training registry  
2. Create TASK_TO_CONDITION mapping (condition_id → (game, opponent))
3. Extract training metrics from condition directories
4. Extract test metrics from test experiment directory
5. Generate visualizations (3×5 subplots for games × opponents)

Output Structure:
- unified_data/task_opponent_training_cooperation.csv
- unified_data/task_opponent_test_results.csv
- plots/cooperation_vs_epoch_3x5.png
- plots/normalized_reward_heatmap_3x3.png
- plots/cooperation_heatmap_3x3.png
- plots/metric_3_4_kld_from_optimal.png
- plots/metric_3_5_cluster_analysis.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent

# Experiment directories - NEW DATA with reduced network (128 hidden, 2 layers)
TRAIN_EXPERIMENT = BASE_DIR / "experiments" / "generalization_matrix_train_918988"
TEST_EXPERIMENT = BASE_DIR / "experiments" / "generalization_matrix_test_918989"

# Output directory
OUTPUT_DIR = BASE_DIR / "results" / "task_opponent_918988_analysis"
UNIFIED_DATA_DIR = OUTPUT_DIR / "unified_data"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Create output directories
UNIFIED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Payoff matrices for normalized reward calculation
PAYOFF_MATRICES = {
    'prisoners-dilemma': {
        'R': 3, 'S': 0, 'T': 5, 'P': 1,
        'min': 0, 'max': 5
    },
    'hawk-dove': {
        'R': 3, 'S': 0, 'T': 6, 'P': -2,
        'min': -2, 'max': 6
    },
    'stag-hunt': {
        'R': 4, 'S': 0, 'T': 2, 'P': 2,
        'min': 0, 'max': 4
    }
}

# Display names for plots
GAME_DISPLAY_NAMES = {
    'prisoners-dilemma': 'PD',
    'hawk-dove': 'HD',
    'stag-hunt': 'SH'
}

# ============================================================================
# TASK-TO-CONDITION MAPPING
# ============================================================================

def create_condition_mapping() -> Dict[int, Tuple[str, float]]:
    """
    Create mapping from condition_id to (game, opponent).
    
    Returns:
        Dict mapping condition_id (0-14) to (game_name, opponent_prob)
    """
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    opponents = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    mapping = {}
    condition_id = 0
    
    for game in games:
        for opponent in opponents:
            mapping[condition_id] = (game, opponent)
            condition_id += 1
    
    return mapping

def create_task_to_condition_mapping() -> Dict[int, Tuple[str, float]]:
    """
    Load master training registry and create mapping from task_id to (game, opponent).
    
    Returns:
        Dict mapping task_id to (game_name, opponent_prob)
    """
    condition_to_game_opp = create_condition_mapping()
    registry_path = TRAIN_EXPERIMENT / "seed_manifests" / "MASTER_TRAINING_REGISTRY.csv"
    
    try:
        df = pd.read_csv(registry_path, on_bad_lines='skip').dropna()
        
        task_to_condition = {}
        for _, row in df.iterrows():
            task_id = int(row['array_task_id'])
            cond_id = int(row['condition_id'])
            
            if cond_id in condition_to_game_opp:
                task_to_condition[task_id] = condition_to_game_opp[cond_id]
        
        return task_to_condition
    
    except Exception as e:
        print(f"WARNING: Could not load training registry: {e}")
        return condition_to_game_opp

CONDITION_TO_GAME_OPP = create_condition_mapping()
TASK_TO_CONDITION = create_task_to_condition_mapping()

print("=" * 80)
print("TASK-OPPONENT SETUP ANALYSIS - Experiment 918988/918989")
print("=" * 80)
print(f"\nCondition Mapping: {len(CONDITION_TO_GAME_OPP)} conditions")
print(f"Task Mapping: {len(TASK_TO_CONDITION)} tasks")
print("\nCondition breakdown:")
condition_summary = {}
for cond_id, (game, opp) in sorted(CONDITION_TO_GAME_OPP.items()):
    key = (game, opp)
    if key not in condition_summary:
        condition_summary[key] = []
    condition_summary[key].append(cond_id)

for (game, opp), cond_ids in sorted(condition_summary.items()):
    print(f"  {game:20s} | Opp: {opp:.1f} | Conditions: {cond_ids}")

# ============================================================================
# ETL: TRAINING DATA
# ============================================================================

def extract_training_data() -> pd.DataFrame:
    """Extract training metrics from condition directories."""
    print("\n" + "=" * 80)
    print("EXTRACTING TRAINING DATA")
    print("=" * 80)
    
    training_dir = TRAIN_EXPERIMENT / "training"
    
    if not training_dir.exists():
        print(f"ERROR: Training directory not found: {training_dir}")
        return pd.DataFrame()
    
    all_training_data = []
    condition_dirs = sorted(training_dir.glob("condition_*_seed_*"))
    print(f"\nFound {len(condition_dirs)} condition/seed directories")
    
    for cond_dir in condition_dirs:
        dir_name = cond_dir.name
        parts = dir_name.split('_')
        
        try:
            cond_id = int(parts[1])
            seed = int(parts[3])
        except (IndexError, ValueError):
            print(f"  WARNING: Could not parse directory name: {dir_name}")
            continue
        
        if cond_id not in CONDITION_TO_GAME_OPP:
            print(f"  WARNING: Unknown condition_id: {cond_id}")
            continue
        
        train_game, train_opponent = CONDITION_TO_GAME_OPP[cond_id]
        unique_task_id = cond_id * 5 + seed
        
        task_dirs = sorted(cond_dir.glob("generalization_matrix_task_*"))
        
        for task_dir in task_dirs:
            task_name = task_dir.name
            try:
                original_task_id = int(task_name.split('_')[3])
            except (IndexError, ValueError):
                continue
            
            results_dir = task_dir / "results"
            metrics_file = results_dir / f"training_task_{original_task_id}_metrics.csv"
            
            if not metrics_file.exists():
                continue
            
            try:
                df = pd.read_csv(metrics_file)
                
                for _, row in df.iterrows():
                    all_training_data.append({
                        'task_id': unique_task_id,
                        'train_game': train_game,
                        'train_opponent': train_opponent,
                        'seed': seed,
                        'epoch': row['epoch'],
                        'cooperation_rate': row.get('epoch_average_cooperation_rate', np.nan),
                        'cumulative_reward': row.get('epoch_cumulative_reward', np.nan),
                        'loss': row.get('total_loss', np.nan)
                    })
            
            except Exception as e:
                print(f"  ERROR loading {metrics_file}: {e}")
                continue
    
    training_df = pd.DataFrame(all_training_data)
    
    print(f"\nExtracted {len(training_df)} training records")
    if len(training_df) > 0:
        print(f"  Tasks: {len(training_df['task_id'].unique())} unique")
        print(f"  Seeds: {sorted(training_df['seed'].unique())}")
        print(f"  Epochs: {training_df['epoch'].min()}-{training_df['epoch'].max()}")
    
    return training_df

# ============================================================================
# ETL: TEST DATA
# ============================================================================

def calculate_normalized_reward(game: str, mean_reward: float) -> float:
    """Calculate normalized reward using payoff matrix bounds."""
    if game not in PAYOFF_MATRICES:
        return np.nan
    
    payoff = PAYOFF_MATRICES[game]
    min_reward = payoff['min']
    max_reward = payoff['max']
    
    if max_reward == min_reward:
        return 0.0
    
    return (mean_reward - min_reward) / (max_reward - min_reward)

def extract_test_data() -> pd.DataFrame:
    """Extract test metrics from test experiment directory."""
    print("\n" + "=" * 80)
    print("EXTRACTING TEST DATA")
    print("=" * 80)
    
    all_test_data = []
    testing_dir = TEST_EXPERIMENT / "testing"
    
    if not testing_dir.exists():
        print(f"ERROR: Test directory not found: {testing_dir}")
        return pd.DataFrame()
    
    test_dirs = sorted(testing_dir.glob("model_*_test_cond_*"))
    print(f"\nFound {len(test_dirs)} test directories")
    
    for test_dir in test_dirs:
        dir_name = test_dir.name
        parts = dir_name.split('_')
        
        try:
            model_id = int(parts[1])
            test_cond_id = int(parts[4])
        except (IndexError, ValueError):
            continue
        
        # Get training info from model_id (model_id = condition_id * 5 + seed)
        train_cond_id = model_id // 5
        seed = model_id % 5
        
        if train_cond_id not in CONDITION_TO_GAME_OPP:
            continue
        
        train_game, train_opponent = CONDITION_TO_GAME_OPP[train_cond_id]
        
        # Get test info
        if test_cond_id not in CONDITION_TO_GAME_OPP:
            continue
        
        test_game, test_opponent = CONDITION_TO_GAME_OPP[test_cond_id]
        
        # Find task directories
        task_dirs = sorted(test_dir.glob("generalization_matrix_task_*"))
        
        for task_dir in task_dirs:
            results_dir = task_dir / "results"
            
            # Find CSV file (pattern may vary)
            csv_files = list(results_dir.glob("*.csv"))
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Aggregate metrics across rows (different opponents within same test condition)
                    mean_reward = df['mean_reward'].mean() if 'mean_reward' in df.columns else np.nan
                    cooperation_rate = df['mean_cooperation_rate'].mean() if 'mean_cooperation_rate' in df.columns else np.nan
                    
                    # Calculate normalized reward using test game's payoff matrix
                    normalized_reward = calculate_normalized_reward(test_game, mean_reward)
                    
                    all_test_data.append({
                        'model_id': model_id,
                        'train_cond_id': train_cond_id,
                        'test_cond_id': test_cond_id,
                        'train_game': train_game,
                        'train_opponent': train_opponent,
                        'test_game': test_game,
                        'test_opponent': test_opponent,
                        'seed': seed,
                        'cooperation_rate': cooperation_rate,
                        'mean_reward': mean_reward,
                        'normalized_reward': normalized_reward
                    })
                    
                    break  # Only process first CSV file
                    
                except Exception as e:
                    continue
    
    test_df = pd.DataFrame(all_test_data)
    
    print(f"\nExtracted {len(test_df)} test records")
    if len(test_df) > 0:
        print(f"  Models tested: {len(test_df['model_id'].unique())}")
        print(f"  Test conditions: {len(test_df['test_cond_id'].unique())}")
    
    return test_df

# ============================================================================
# VISUALIZATION: COOPERATION VS EPOCH (3×5 GRID)
# ============================================================================

def plot_cooperation_vs_epoch(training_df: pd.DataFrame):
    """Plot cooperation rate vs epoch with 3×5 grid (games × opponents)."""
    print("\n" + "=" * 80)
    print("GENERATING PLOT: Cooperation vs Epoch (3×5)")
    print("=" * 80)
    
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    opponents = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle('Training Cooperation Rate vs Epoch (Task-Opponent Setup)', 
                 fontsize=16, fontweight='bold')
    
    for i, game in enumerate(games):
        for j, opponent in enumerate(opponents):
            ax = axes[i, j]
            
            # Filter data
            subset = training_df[
                (training_df['train_game'] == game) & 
                (training_df['train_opponent'] == opponent)
            ]
            
            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                ax.set_xlim(0, 500)
                ax.set_ylim(0, 1)
            else:
                # Compute mean and SEM across seeds
                grouped = subset.groupby('epoch')['cooperation_rate'].agg(['mean', 'sem'])
                
                # Plot mean with shaded SEM region
                ax.plot(grouped.index, grouped['mean'], 
                       linewidth=2.5, color='blue', alpha=0.8)
                ax.fill_between(grouped.index, 
                               grouped['mean'] - grouped['sem'],
                               grouped['mean'] + grouped['sem'],
                               alpha=0.3, color='blue')
                
                ax.set_xlim(0, 500)
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
            
            # Labels
            if i == 2:  # Bottom row
                ax.set_xlabel('Epoch', fontsize=9)
            if j == 0:  # Left column
                ax.set_ylabel('Cooperation Rate', fontsize=9)
            
            # Title
            game_abbrev = GAME_DISPLAY_NAMES[game]
            ax.set_title(f'{game_abbrev} | Opp p={opponent:.1f}', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_file = PLOTS_DIR / "cooperation_vs_epoch_3x5.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

# ============================================================================
# VISUALIZATION: NORMALIZED REWARD HEATMAP (3×3 GRID)
# ============================================================================

def plot_normalized_reward_heatmap(test_df: pd.DataFrame):
    """Plot normalized reward heatmap with 3×3 grid (train game × test game)."""
    print("\n" + "=" * 80)
    print("GENERATING PLOT: Normalized Reward Heatmap (3×3)")
    print("=" * 80)
    
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    game_names = {g: GAME_DISPLAY_NAMES[g] for g in games}
    opponents = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle('Normalized Reward by Train/Test Opponent (Task-Opponent Setup)', 
                 fontsize=16, fontweight='bold')
    
    for i, train_game in enumerate(games):
        for j, test_game in enumerate(games):
            ax = axes[i, j]
            
            subset = test_df[
                (test_df['train_game'] == train_game) & 
                (test_df['test_game'] == test_game)
            ]
            
            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                pivot = subset.pivot_table(
                    index='train_opponent',
                    columns='test_opponent',
                    values='normalized_reward',
                    aggfunc='mean'
                )
                
                pivot_normalized = pivot.reindex(index=opponents, columns=opponents)
                
                sns.heatmap(pivot_normalized, ax=ax, cmap='RdYlGn', center=0.5, 
                           vmin=0, vmax=1, cbar=(j == 2),
                           annot=True, fmt='.2f', 
                           cbar_kws={'shrink': 0.8, 'label': 'Normalized Reward'},
                           annot_kws={'fontsize': 9})
                
                ax.set_xlabel('Test Opponent' if i == 2 else '', fontsize=10)
                ax.set_ylabel('Train Opponent' if j == 0 else '', fontsize=10)
                ax.tick_params(labelsize=9)
            
            ax.set_title(f'Train: {game_names[train_game]} | Test: {game_names[test_game]}', 
                       fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_file = PLOTS_DIR / "normalized_reward_heatmap_3x3.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

# ============================================================================
# VISUALIZATION: COOPERATION HEATMAP (3×3 GRID)
# ============================================================================

def plot_cooperation_heatmap(test_df: pd.DataFrame):
    """Plot cooperation rate heatmap with 3×3 grid (train game × test game)."""
    print("\n" + "=" * 80)
    print("GENERATING PLOT: Cooperation Rate Heatmap (3×3)")
    print("=" * 80)
    
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    game_names = {g: GAME_DISPLAY_NAMES[g] for g in games}
    opponents = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle('Cooperation Rate by Train/Test Opponent (Task-Opponent Setup)', 
                 fontsize=16, fontweight='bold')
    
    for i, train_game in enumerate(games):
        for j, test_game in enumerate(games):
            ax = axes[i, j]
            
            subset = test_df[
                (test_df['train_game'] == train_game) & 
                (test_df['test_game'] == test_game)
            ]
            
            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                pivot = subset.pivot_table(
                    index='train_opponent',
                    columns='test_opponent',
                    values='cooperation_rate',
                    aggfunc='mean'
                )
                
                pivot = pivot.reindex(index=opponents, columns=opponents)
                
                sns.heatmap(pivot, ax=ax, cmap='Blues', 
                           vmin=0, vmax=1, cbar=(j == 2),
                           annot=True, fmt='.2f', 
                           cbar_kws={'shrink': 0.8, 'label': 'Cooperation Rate'},
                           annot_kws={'fontsize': 9})
                
                ax.set_xlabel('Test Opponent' if i == 2 else '', fontsize=10)
                ax.set_ylabel('Train Opponent' if j == 0 else '', fontsize=10)
                ax.tick_params(labelsize=9)
            
            ax.set_title(f'Train: {game_names[train_game]} | Test: {game_names[test_game]}', 
                       fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_file = PLOTS_DIR / "cooperation_heatmap_3x3.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

# ============================================================================
# VISUALIZATION: NORMALIZED REWARD SUMMARY (3-PANEL)
# ============================================================================

def plot_normalized_reward_summary(test_df: pd.DataFrame):
    """Plot normalized reward as 3-panel line plot (like KLD format)."""
    print("\n" + "=" * 80)
    print("GENERATING PLOT: Normalized Reward Summary (3-Panel)")
    print("=" * 80)
    
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    opponents = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    opp_colors = {0.1: '#2E86AB', 0.3: '#54A8C7', 0.5: '#9E9E9E', 0.7: '#E07A5F', 0.9: '#C1121F'}
    game_names = {g: GAME_DISPLAY_NAMES[g] for g in games}
    sorted_games = sorted(games)
    
    # Create x-axis positions
    x_positions = {}
    x_labels = []
    x_ticks = []
    pos = 0
    
    for test_game in sorted_games:
        for test_opp in opponents:
            x_positions[(test_game, test_opp)] = pos
            x_labels.append(f'{game_names[test_game]}\n{test_opp:.1f}')
            x_ticks.append(pos)
            pos += 1
    
    # Plot: 3 rows (one per training game)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for i, train_game in enumerate(sorted_games):
        ax = axes[i]
        train_data = test_df[test_df['train_game'] == train_game]
        
        if len(train_data) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            continue
        
        for train_opp in opponents:
            opp_data = train_data[train_data['train_opponent'] == train_opp]
            
            if len(opp_data) == 0:
                continue
            
            # Aggregate across test conditions
            agg_data = opp_data.groupby(['test_game', 'test_opponent'])['normalized_reward'].mean().reset_index()
            agg_data = agg_data.sort_values(['test_game', 'test_opponent'])
            
            x_vals = [x_positions[(row['test_game'], row['test_opponent'])] 
                     for _, row in agg_data.iterrows()]
            y_vals = agg_data['normalized_reward'].values
            
            ax.plot(x_vals, y_vals, marker='o', linewidth=2, markersize=6,
                   label=f'Train opp={train_opp:.1f}', color=opp_colors[train_opp])
        
        ax.set_title(f'Training Game: {game_names[train_game]}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Reward', fontsize=11)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels if i == 2 else [''] * len(x_labels), fontsize=8)
        ax.legend(fontsize=9, loc='best', ncol=5)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axhline(y=1, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Add vertical separators between games
        for sep_pos in [4.5, 9.5]:
            ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5)
    
    axes[2].set_xlabel('Test Game and Opponent', fontsize=11)
    plt.suptitle('Task-Opponent Setup: Normalized Reward by Train/Test Condition', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'normalized_reward_summary_3panel.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

# ============================================================================
# METRIC 3.4: KLD FROM OPTIMAL POLICY
# ============================================================================

def metric_3_4_kld_from_optimal(df_test, df_train):
    """Compute KL divergence from optimal policy for each test condition."""
    print("\n" + "=" * 80)
    print("GENERATING PLOT: Metric 3.4 - KLD from Optimal Policy")
    print("=" * 80)
    
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    opponents = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    def kl_divergence_bernoulli(p, q, epsilon=1e-10):
        """Compute KL divergence between two Bernoulli distributions."""
        p = np.clip(p, epsilon, 1 - epsilon)
        q = np.clip(q, epsilon, 1 - epsilon)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    
    # Get final training cooperation rates for each condition (as optimal policies)
    optimal_policies = {}
    for game in games:
        for opp in opponents:
            cond_data = df_train[
                (df_train['train_game'] == game) &
                (df_train['train_opponent'] == opp) &
                (df_train['epoch'] == df_train['epoch'].max())
            ]
            
            if len(cond_data) > 0:
                optimal_policies[(game, opp)] = cond_data['cooperation_rate'].mean()
    
    # Compute KLD for each test condition
    kld_results = []
    
    for test_game in games:
        for test_opp in opponents:
            if (test_game, test_opp) not in optimal_policies:
                continue
            
            optimal_coop_rate = optimal_policies[(test_game, test_opp)]
            test_agents = df_test[
                (df_test['test_game'] == test_game) &
                (df_test['test_opponent'] == test_opp)
            ]
            
            for train_game in games:
                for train_opp in opponents:
                    agents = test_agents[
                        (test_agents['train_game'] == train_game) &
                        (test_agents['train_opponent'] == train_opp)
                    ]
                    
                    if len(agents) == 0:
                        continue
                    
                    mean_coop = agents['cooperation_rate'].mean()
                    kld = kl_divergence_bernoulli(mean_coop, optimal_coop_rate)
                    
                    kld_results.append({
                        'train_game': train_game,
                        'train_opponent': train_opp,
                        'test_game': test_game,
                        'test_opponent': test_opp,
                        'kld': kld,
                        'test_coop_rate': mean_coop,
                        'optimal_coop_rate': optimal_coop_rate
                    })
    
    df_kld = pd.DataFrame(kld_results)
    kld_csv = UNIFIED_DATA_DIR / 'task_opponent_kld_from_optimal.csv'
    df_kld.to_csv(kld_csv, index=False)
    print(f"  Saved KLD data: {kld_csv.name} ({len(df_kld)} records)")
    
    if len(df_kld) == 0:
        print("  WARNING: No KLD data to plot")
        return
    
    # Plot: 3 rows (one per training game)
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    opp_colors = {0.1: '#2E86AB', 0.3: '#54A8C7', 0.5: '#9E9E9E', 0.7: '#E07A5F', 0.9: '#C1121F'}
    game_names = {g: GAME_DISPLAY_NAMES[g] for g in games}
    sorted_games = sorted(games)
    
    # Create x-axis positions
    x_positions = {}
    x_labels = []
    x_ticks = []
    pos = 0
    
    for test_game in sorted_games:
        for test_opp in opponents:
            x_positions[(test_game, test_opp)] = pos
            x_labels.append(f'{game_names[test_game]}\n{test_opp:.1f}')
            x_ticks.append(pos)
            pos += 1
    
    global_kld_min = df_kld['kld'].min()
    global_kld_max = df_kld['kld'].max()
    y_margin = (global_kld_max - global_kld_min) * 0.05
    
    for i, train_game in enumerate(sorted_games):
        ax = axes[i]
        train_data = df_kld[df_kld['train_game'] == train_game]
        
        if len(train_data) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            continue
        
        for train_opp in opponents:
            opp_data = train_data[train_data['train_opponent'] == train_opp]
            
            if len(opp_data) == 0:
                continue
            
            opp_data = opp_data.sort_values(['test_game', 'test_opponent'])
            x_vals = [x_positions[(row['test_game'], row['test_opponent'])] 
                     for _, row in opp_data.iterrows()]
            y_vals = opp_data['kld'].values
            
            ax.plot(x_vals, y_vals, marker='o', linewidth=2, markersize=6,
                   label=f'Train opp={train_opp:.1f}', color=opp_colors[train_opp])
        
        ax.set_title(f'Training Game: {game_names[train_game]}', fontsize=12, fontweight='bold')
        ax.set_ylabel('KL Divergence', fontsize=11)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels if i == 2 else [''] * len(x_labels), fontsize=9)
        ax.legend(fontsize=9, loc='best', ncol=5)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(global_kld_min - y_margin, global_kld_max + y_margin)
        
        for sep_pos in [4.5, 9.5]:
            ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5)
    
    axes[2].set_xlabel('Test Game and Opponent', fontsize=11)
    plt.suptitle('Task-Opponent Setup: KL Divergence from Optimal Policy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_3_4_kld_from_optimal.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

# ============================================================================
# METRIC 3.5: CLUSTER ANALYSIS
# ============================================================================

def metric_3_5_cluster_analysis(df_test):
    """Cluster analysis of agents based on behavior."""
    print("\n" + "=" * 80)
    print("GENERATING PLOT: Metric 3.5 - Cluster Analysis")
    print("=" * 80)
    
    agent_metrics = []
    
    for model_id in df_test['model_id'].unique():
        agent_data = df_test[df_test['model_id'] == model_id]
        
        train_game = agent_data['train_game'].iloc[0]
        train_opponent = agent_data['train_opponent'].iloc[0]
        seed = agent_data['seed'].iloc[0]
        
        mean_coop = agent_data['cooperation_rate'].mean()
        mean_reward_norm = agent_data['normalized_reward'].mean()
        
        agent_metrics.append({
            'model_id': model_id,
            'train_game': train_game,
            'train_opponent': train_opponent,
            'seed': seed,
            'mean_cooperation': mean_coop,
            'mean_normalized_reward': mean_reward_norm
        })
    
    df_individual = pd.DataFrame(agent_metrics)
    
    df_cluster = df_individual.groupby(['train_game', 'train_opponent']).agg({
        'mean_cooperation': ['mean', 'sem'],
        'mean_normalized_reward': ['mean', 'sem']
    }).reset_index()
    
    df_cluster.columns = ['train_game', 'train_opponent', 'coop_mean', 'coop_sem', 'reward_mean', 'reward_sem']
    
    print(f"  Aggregated {len(df_individual)} agents into {len(df_cluster)} conditions")
    
    cluster_csv = UNIFIED_DATA_DIR / 'task_opponent_cluster_analysis.csv'
    df_cluster.to_csv(cluster_csv, index=False)
    print(f"  Saved: {cluster_csv.name}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 9))
    
    opp_colors = {0.1: '#2E86AB', 0.3: '#54A8C7', 0.5: '#9E9E9E', 0.7: '#E07A5F', 0.9: '#C1121F'}
    game_markers = {'prisoners-dilemma': 'o', 'hawk-dove': 's', 'stag-hunt': '^'}
    
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    opponents = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for game in games:
        for opp in opponents:
            game_opp_data = df_cluster[
                (df_cluster['train_game'] == game) &
                (df_cluster['train_opponent'] == opp)
            ]
            
            if len(game_opp_data) == 0:
                continue
            
            ax.errorbar(game_opp_data['coop_mean'], 
                       game_opp_data['reward_mean'],
                       xerr=game_opp_data['coop_sem'],
                       yerr=game_opp_data['reward_sem'],
                       fmt=game_markers[game], 
                       color=opp_colors[opp],
                       markersize=12, alpha=0.7, 
                       markeredgecolor='black', markeredgewidth=1.5,
                       capsize=4, capthick=1.5, elinewidth=1.5)
    
    ax.set_xlabel('Mean Cooperation Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Normalized Reward', fontsize=12, fontweight='bold')
    ax.set_title('Agent Clustering by Behavior (5 seeds/condition)', fontsize=14, fontweight='bold')
    
    from matplotlib.lines import Line2D
    
    game_legend = [Line2D([0], [0], marker=game_markers[g], color='w', 
                          markerfacecolor='gray', markersize=10, label=g)
                   for g in games]
    opp_legend = [Line2D([0], [0], marker='o', color='w', 
                         markerfacecolor=opp_colors[o], markersize=10, label=f'opp={o:.1f}')
                  for o in opponents]
    
    first_legend = ax.legend(handles=game_legend, title='Game', 
                            loc='upper right', bbox_to_anchor=(1.0, 1.0))
    ax.add_artist(first_legend)
    ax.legend(handles=opp_legend, title='Opponent', 
             loc='upper right', bbox_to_anchor=(1.0, 0.65))
    
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_3_5_cluster_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

# ============================================================================
# METRIC 3.6A: CROSS-GENERALIZATION ANALYSIS
# ============================================================================

def metric_3_6a_cross_generalization_analysis(df_test):
    """Cross-generalization analysis by transfer type."""
    print("\n" + "=" * 80)
    print("GENERATING PLOT: Metric 3.6a - Cross-Generalization Analysis")
    print("=" * 80)
    
    agent_metrics = []
    
    for model_id in df_test['model_id'].unique():
        agent_data = df_test[df_test['model_id'] == model_id]
        
        train_game = agent_data['train_game'].iloc[0]
        train_opponent = agent_data['train_opponent'].iloc[0]
        seed = agent_data['seed'].iloc[0]
        
        same_game_diff_opp = agent_data[
            (agent_data['test_game'] == train_game) &
            (agent_data['test_opponent'] != train_opponent)
        ]['normalized_reward'].mean()
        
        diff_game_same_opp = agent_data[
            (agent_data['test_game'] != train_game) &
            (agent_data['test_opponent'] == train_opponent)
        ]['normalized_reward'].mean()
        
        diff_game_diff_opp = agent_data[
            (agent_data['test_game'] != train_game) &
            (agent_data['test_opponent'] != train_opponent)
        ]['normalized_reward'].mean()
        
        agent_metrics.append({
            'model_id': model_id,
            'train_game': train_game,
            'train_opponent': train_opponent,
            'seed': seed,
            'same_game_diff_opp': same_game_diff_opp,
            'diff_game_same_opp': diff_game_same_opp,
            'diff_game_diff_opp': diff_game_diff_opp
        })
    
    df_individual = pd.DataFrame(agent_metrics)
    
    df_agg = df_individual.groupby(['train_game', 'train_opponent']).agg({
        'same_game_diff_opp': ['mean', 'sem'],
        'diff_game_same_opp': ['mean', 'sem'],
        'diff_game_diff_opp': ['mean', 'sem']
    }).reset_index()
    
    df_agg.columns = ['train_game', 'train_opponent', 
                      'same_game_mean', 'same_game_sem',
                      'same_opp_mean', 'same_opp_sem',
                      'diff_both_mean', 'diff_both_sem']
    
    print(f"  Aggregated {len(df_individual)} agents into {len(df_agg)} conditions")
    
    agg_csv = UNIFIED_DATA_DIR / 'task_opponent_cross_generalization.csv'
    df_agg.to_csv(agg_csv, index=False)
    print(f"  Saved: {agg_csv.name}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    conditions = []
    for _, row in df_agg.iterrows():
        game_abbrev = GAME_DISPLAY_NAMES[row['train_game']]
        conditions.append(f"{game_abbrev}\n{row['train_opponent']:.1f}")
    
    x = np.arange(len(conditions))
    width = 0.25
    
    ax.bar(x - width, df_agg['same_game_mean'], width, 
           yerr=df_agg['same_game_sem'], capsize=3,
           label='Same game, diff opponent', color='#3498db', alpha=0.8)
    ax.bar(x, df_agg['same_opp_mean'], width,
           yerr=df_agg['same_opp_sem'], capsize=3,
           label='Diff game, same opponent', color='#2ecc71', alpha=0.8)
    ax.bar(x + width, df_agg['diff_both_mean'], width,
           yerr=df_agg['diff_both_sem'], capsize=3,
           label='Diff game, diff opponent', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Training Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Normalized Reward', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Generalization Analysis (5 seeds/condition)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_3_6a_cross_generalization_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    
    # ETL: Training Data
    training_df = extract_training_data()
    
    if len(training_df) > 0:
        training_output = UNIFIED_DATA_DIR / "task_opponent_training_cooperation.csv"
        training_df.to_csv(training_output, index=False)
        print(f"\n✓ Saved training data: {training_output}")
        print(f"  Shape: {training_df.shape}")
    else:
        print("\n✗ No training data extracted")
    
    # ETL: Test Data
    test_df = extract_test_data()
    
    if len(test_df) > 0:
        test_output = UNIFIED_DATA_DIR / "task_opponent_test_results.csv"
        test_df.to_csv(test_output, index=False)
        print(f"\n✓ Saved test data: {test_output}")
        print(f"  Shape: {test_df.shape}")
    else:
        print("\n✗ No test data extracted")
    
    # Generate Visualizations
    if len(training_df) > 0:
        plot_cooperation_vs_epoch(training_df)
    
    if len(test_df) > 0:
        plot_normalized_reward_heatmap(test_df)
        plot_cooperation_heatmap(test_df)
        plot_normalized_reward_summary(test_df)  # 3-panel summary plot
        
        # Additional metrics
        if len(training_df) > 0:
            metric_3_4_kld_from_optimal(test_df, training_df)
        
        metric_3_5_cluster_analysis(test_df)
        metric_3_6a_cross_generalization_analysis(test_df)
    
    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput Directory: {OUTPUT_DIR}")
    print(f"\nGenerated Files:")
    print(f"  Data:")
    print(f"    - task_opponent_training_cooperation.csv ({len(training_df)} rows)")
    print(f"    - task_opponent_test_results.csv ({len(test_df)} rows)")
    print(f"    - task_opponent_kld_from_optimal.csv")
    print(f"    - task_opponent_cluster_analysis.csv")
    print(f"    - task_opponent_cross_generalization.csv")
    print(f"  Plots:")
    print(f"    - cooperation_vs_epoch_3x5.png")
    print(f"    - normalized_reward_heatmap_3x3.png")
    print(f"    - cooperation_heatmap_3x3.png")
    print(f"    - metric_3_4_kld_from_optimal.png")
    print(f"    - metric_3_5_cluster_analysis.png")
    print(f"    - metric_3_6a_cross_generalization_analysis.png")
    print("=" * 80)

if __name__ == "__main__":
    main()
