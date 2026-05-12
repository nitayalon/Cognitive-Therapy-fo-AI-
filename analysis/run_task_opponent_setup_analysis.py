"""
Task-Opponent Setup Analysis Script
====================================
Analyzes generalization matrix experiments with 3 games × 5 opponents (15 conditions).

ETL Pipeline:
1. Load master training registry
2. Create TASK_TO_CONDITION mapping (condition_id → (game, opponent))
3. Extract training metrics from condition directories
4. Extract test metrics from two test experiment directories
5. Generate visualizations (3×5 subplots for games × opponents)

Output Structure:
- unified_data/task_opponent_training_cooperation.csv
- unified_data/task_opponent_test_results.csv
- plots/cooperation_vs_epoch_3x5.png
- plots/normalized_reward_heatmap_3x5.png
- plots/cooperation_heatmap_3x5.png
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

# Experiment directories
TRAIN_EXPERIMENT = BASE_DIR / "experiments" / "generalization_matrix_train_913243"
TEST_EXPERIMENT_1 = BASE_DIR / "experiments" / "generalization_matrix_test_913244"
TEST_EXPERIMENT_2 = BASE_DIR / "experiments" / "generalization_matrix_test_913245"

# Output directory
OUTPUT_DIR = BASE_DIR / "Results" / "task_opponent_setup"
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
        'R': 3, 'S': 1, 'T': 5, 'P': 0,
        'min': 0, 'max': 5
    },
    'stag-hunt': {
        'R': 5, 'S': 0, 'T': 3, 'P': 1,
        'min': 0, 'max': 5
    }
}

# ============================================================================
# TASK-TO-CONDITION MAPPING
# ============================================================================

def create_condition_mapping() -> Dict[int, Tuple[str, float]]:
    """
    Create mapping from condition_id to (game, opponent).
    Used for training data extraction which uses directory names like condition_X_seed_Y.
    
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
    Used for test data extraction which references model IDs.
    
    Returns:
        Dict mapping task_id (0-74) to (game_name, opponent_prob)
    """
    # First create condition_id to (game, opponent) mapping
    condition_to_game_opp = create_condition_mapping()
    
    # Load master training registry to get task_id -> condition_id mapping
    registry_path = TRAIN_EXPERIMENT / "seed_manifests" / "MASTER_TRAINING_REGISTRY.csv"
    
    try:
        df = pd.read_csv(registry_path, on_bad_lines='skip').dropna()
        
        # Create task_id -> (game, opponent) mapping
        task_to_condition = {}
        for _, row in df.iterrows():
            task_id = int(row['array_task_id'])
            cond_id = int(row['condition_id'])
            
            if cond_id in condition_to_game_opp:
                task_to_condition[task_id] = condition_to_game_opp[cond_id]
        
        return task_to_condition
    
    except Exception as e:
        print(f"WARNING: Could not load training registry: {e}")
        print("Using default condition-based mapping (0-14)")
        return condition_to_game_opp

CONDITION_TO_GAME_OPP = create_condition_mapping()
TASK_TO_CONDITION = create_task_to_condition_mapping()

print("=" * 80)
print("TASK-OPPONENT SETUP ANALYSIS")
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
    """
    Extract training metrics from condition directories.
    
    Directory structure:
        experiments/generalization_matrix_train_913243/training/
            condition_X_seed_Y/generalization_matrix_task_Z/results/
                training_task_Z_metrics.csv
    
    Returns:
        DataFrame with columns: task_id, train_game, train_opponent, seed, 
                                epoch, cooperation_rate, cumulative_reward, loss
    """
    print("\n" + "=" * 80)
    print("EXTRACTING TRAINING DATA")
    print("=" * 80)
    
    training_dir = TRAIN_EXPERIMENT / "training"
    
    if not training_dir.exists():
        print(f"ERROR: Training directory not found: {training_dir}")
        return pd.DataFrame()
    
    all_training_data = []
    
    # Iterate through condition directories
    condition_dirs = sorted(training_dir.glob("condition_*_seed_*"))
    print(f"\nFound {len(condition_dirs)} condition/seed directories")
    
    for cond_dir in condition_dirs:
        # Parse condition_id and seed from directory name
        # Format: condition_X_seed_Y
        dir_name = cond_dir.name
        parts = dir_name.split('_')
        
        try:
            cond_id = int(parts[1])
            seed = int(parts[3])
        except (IndexError, ValueError):
            print(f"  WARNING: Could not parse directory name: {dir_name}")
            continue
        
        # Get game and opponent from condition_id (from directory name)
        if cond_id not in CONDITION_TO_GAME_OPP:
            print(f"  WARNING: Unknown condition_id: {cond_id}")
            continue
        
        train_game, train_opponent = CONDITION_TO_GAME_OPP[cond_id]
        
        # Find task directories
        task_dirs = sorted(cond_dir.glob("generalization_matrix_task_*"))
        
        for task_dir in task_dirs:
            # Parse task_id from directory name (format: generalization_matrix_task_X_TIMESTAMP)
            task_name = task_dir.name
            try:
                task_id = int(task_name.split('_')[3])
            except (IndexError, ValueError):
                print(f"  WARNING: Could not parse task_id from: {task_name}")
                continue
            
            # Load training metrics CSV
            results_dir = task_dir / "results"
            metrics_file = results_dir / f"training_task_{task_id}_metrics.csv"
            
            if not metrics_file.exists():
                print(f"  WARNING: Metrics file not found: {metrics_file}")
                continue
            
            try:
                df = pd.read_csv(metrics_file)
                
                # Extract required columns
                for _, row in df.iterrows():
                    all_training_data.append({
                        'task_id': task_id,
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
    
    # Create DataFrame
    training_df = pd.DataFrame(all_training_data)
    
    print(f"\nExtracted {len(training_df)} training records")
    if len(training_df) > 0:
        print(f"  Tasks: {sorted(training_df['task_id'].unique())}")
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
    """
    Extract test metrics from two test experiment directories.
    
    Directory structure:
        experiments/generalization_matrix_test_913244/testing/
            model_X_test_cond_Y/generalization_matrix_task_Z/results/
                test_model_X_on_condition_Y.csv
    
    Returns:
        DataFrame with columns: model_id, test_cond_id, train_game, train_opponent,
                                test_game, test_opponent, seed, cooperation_rate,
                                mean_reward, normalized_reward
    """
    print("\n" + "=" * 80)
    print("EXTRACTING TEST DATA")
    print("=" * 80)
    
    all_test_data = []
    
    # Process both test experiment directories
    test_experiments = [TEST_EXPERIMENT_1, TEST_EXPERIMENT_2]
    
    for test_exp in test_experiments:
        testing_dir = test_exp / "testing"
        
        if not testing_dir.exists():
            print(f"\nWARNING: Testing directory not found: {testing_dir}")
            continue
        
        print(f"\nProcessing: {test_exp.name}")
        
        # Iterate through model directories
        model_dirs = sorted(testing_dir.glob("model_*_test_cond_*"))
        print(f"  Found {len(model_dirs)} model/test_cond directories")
        
        for model_dir in model_dirs:
            # Parse model_id and test_cond_id from directory name
            # Format: model_X_test_cond_Y
            dir_name = model_dir.name
            parts = dir_name.split('_')
            
            try:
                model_id = int(parts[1])
                test_cond_id = int(parts[4])
            except (IndexError, ValueError):
                print(f"  WARNING: Could not parse directory name: {dir_name}")
                continue
            
            # Get training game/opponent from model_id
            if model_id not in TASK_TO_CONDITION:
                print(f"  WARNING: Unknown model_id: {model_id}")
                continue
            
            train_game, train_opponent = TASK_TO_CONDITION[model_id]
            
            # Get test game/opponent from test_cond_id
            if test_cond_id not in CONDITION_TO_GAME_OPP:
                print(f"  WARNING: Unknown test_cond_id: {test_cond_id}")
                continue
            
            test_game, test_opponent = CONDITION_TO_GAME_OPP[test_cond_id]
            
            # Find task directories
            task_dirs = sorted(model_dir.glob("generalization_matrix_task_*"))
            
            for task_dir in task_dirs:
                # Parse task_id from directory name
                task_name = task_dir.name
                try:
                    task_id = int(task_name.split('_')[-1])
                except (IndexError, ValueError):
                    continue
                
                # Load test results CSV (filename pattern uses "None" instead of model_id)
                results_dir = task_dir / "results"
                test_files = list(results_dir.glob("test_model_*_on_condition_*.csv"))
                
                if not test_files:
                    continue
                
                for test_file in test_files:
                    try:
                        df = pd.read_csv(test_file)
                        
                        # Extract metrics from CSV
                        if len(df) == 0:
                            continue
                        
                        row = df.iloc[0]
                        mean_reward = row.get('mean_reward', np.nan)
                        cooperation_rate = row.get('mean_cooperation_rate', np.nan)
                        
                        # Calculate normalized reward
                        normalized_reward = calculate_normalized_reward(test_game, mean_reward)
                        
                        all_test_data.append({
                            'model_id': model_id,
                            'test_cond_id': test_cond_id,
                            'train_game': train_game,
                            'train_opponent': train_opponent,
                            'test_game': test_game,
                            'test_opponent': test_opponent,
                            'seed': task_id,  # Using task_id as seed proxy
                            'cooperation_rate': cooperation_rate,
                            'mean_reward': mean_reward,
                            'normalized_reward': normalized_reward
                        })
                    
                    except Exception as e:
                        print(f"  ERROR loading {test_file}: {e}")
                        continue
    
    # Create DataFrame
    test_df = pd.DataFrame(all_test_data)
    
    print(f"\nExtracted {len(test_df)} test records")
    if len(test_df) > 0:
        print(f"  Models: {sorted(test_df['model_id'].unique())}")
        print(f"  Test conditions: {sorted(test_df['test_cond_id'].unique())}")
    
    return test_df

# ============================================================================
# VISUALIZATION: COOPERATION VS EPOCH (3 SEPARATE PLOTS BY GAME)
# ============================================================================

def plot_cooperation_vs_epoch(training_df: pd.DataFrame):
    """
    Create 3 separate plots (one per game) showing cooperation rate vs epoch.
    Each plot has 5 lines (one per opponent) with consistent color scheme.
    """
    print("\n" + "=" * 80)
    print("GENERATING PLOT: Cooperation vs Epoch (3 plots by game)")
    print("=" * 80)
    
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    game_names = {'prisoners-dilemma': 'Prisoner\'s Dilemma', 
                  'hawk-dove': 'Hawk-Dove', 
                  'stag-hunt': 'Stag Hunt'}
    opponents = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Consistent color scheme for opponents (blue=cooperative, red=defective)
    colors = {
        0.1: '#2E86AB',  # Dark blue (very cooperative)
        0.3: '#54A8C7',  # Medium blue
        0.5: '#9E9E9E',  # Gray (neutral)
        0.7: '#E07A5F',  # Orange-red
        0.9: '#C1121F'   # Dark red (very defective)
    }
    
    for game in games:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for opponent in opponents:
            # Filter data for this game/opponent
            subset = training_df[
                (training_df['train_game'] == game) & 
                (training_df['train_opponent'] == opponent)
            ]
            
            if len(subset) == 0:
                continue
            
            # Group by epoch and calculate mean/std across seeds
            grouped = subset.groupby('epoch')['cooperation_rate'].agg(['mean', 'std'])
            
            epochs = grouped.index
            mean_coop = grouped['mean']
            std_coop = grouped['std'].fillna(0)
            
            # Plot line with shaded error region
            color = colors[opponent]
            ax.plot(epochs, mean_coop, linewidth=2.5, color=color, 
                   label=f'Opponent p={opponent:.1f}', alpha=0.9)
            ax.fill_between(epochs, mean_coop - std_coop, mean_coop + std_coop, 
                           alpha=0.15, color=color)
        
        # Formatting
        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cooperation Rate', fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'Training Cooperation Rate: {game_names[game]}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.tick_params(labelsize=11)
        
        plt.tight_layout()
        game_short = game.replace('prisoners-dilemma', 'pd').replace('hawk-dove', 'hd').replace('stag-hunt', 'sh')
        output_file = PLOTS_DIR / f"cooperation_vs_epoch_{game_short}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

# ============================================================================
# VISUALIZATION: NORMALIZED REWARD HEATMAP (3×3 GRID)
# ============================================================================

def plot_normalized_reward_heatmap(test_df: pd.DataFrame):
    """
    Plot normalized reward heatmap with 3×3 grid (train game × test game).
    Each subplot shows train_opponent (rows) vs test_opponent (cols).
    Rewards are normalized within each subplot to [0,1].
    """
    print("\n" + "=" * 80)
    print("GENERATING PLOT: Normalized Reward Heatmap (3×3)")
    print("=" * 80)
    
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    game_names = {'prisoners-dilemma': 'PD', 'hawk-dove': 'HD', 'stag-hunt': 'SH'}
    opponents = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle('Normalized Reward by Train/Test Opponent (Task-Opponent Setup)', 
                 fontsize=16, fontweight='bold')
    
    for i, train_game in enumerate(games):
        for j, test_game in enumerate(games):
            ax = axes[i, j]
            
            # Filter data: models trained on train_game, tested on test_game
            subset = test_df[
                (test_df['train_game'] == train_game) & 
                (test_df['test_game'] == test_game)
            ]
            
            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'Train: {game_names[train_game]} | Test: {game_names[test_game]}', 
                           fontsize=11, fontweight='bold')
                continue
            
            # Create pivot table: train_opponent × test_opponent
            pivot = subset.pivot_table(
                index='train_opponent',
                columns='test_opponent',
                values='mean_reward',  # Use raw rewards, not pre-normalized
                aggfunc='mean'
            )
            
            # Reindex to ensure all opponents present
            pivot = pivot.reindex(index=opponents, columns=opponents)
            
            # Normalize within this subplot to [0, 1]
            pivot_min = pivot.min().min()
            pivot_max = pivot.max().max()
            
            if pivot_max > pivot_min:
                pivot_normalized = (pivot - pivot_min) / (pivot_max - pivot_min)
            else:
                pivot_normalized = pivot * 0  # All zeros if no variation
            
            # Plot heatmap
            sns.heatmap(pivot_normalized, ax=ax, cmap='RdYlGn', center=0.5, 
                       vmin=0, vmax=1, cbar=(j == 2),  # Only rightmost column gets colorbar
                       annot=True, fmt='.2f', 
                       cbar_kws={'shrink': 0.8, 'label': 'Normalized Reward'},
                       annot_kws={'fontsize': 9})
            
            # Formatting
            ax.set_title(f'Train: {game_names[train_game]} | Test: {game_names[test_game]}', 
                       fontsize=11, fontweight='bold')
            ax.set_xlabel('Test Opponent' if i == 2 else '', fontsize=10)
            ax.set_ylabel('Train Opponent' if j == 0 else '', fontsize=10)
            ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    output_file = PLOTS_DIR / "normalized_reward_heatmap_3x3.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

# ============================================================================
# VISUALIZATION: COOPERATION HEATMAP (3×3 GRID)
# ============================================================================

def plot_cooperation_heatmap(test_df: pd.DataFrame):
    """
    Plot cooperation rate heatmap with 3×3 grid (train game × test game).
    Each subplot shows train_opponent (rows) vs test_opponent (cols).
    """
    print("\n" + "=" * 80)
    print("GENERATING PLOT: Cooperation Rate Heatmap (3×3)")
    print("=" * 80)
    
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    game_names = {'prisoners-dilemma': 'PD', 'hawk-dove': 'HD', 'stag-hunt': 'SH'}
    opponents = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle('Cooperation Rate by Train/Test Opponent (Task-Opponent Setup)', 
                 fontsize=16, fontweight='bold')
    
    for i, train_game in enumerate(games):
        for j, test_game in enumerate(games):
            ax = axes[i, j]
            
            # Filter data
            subset = test_df[
                (test_df['train_game'] == train_game) & 
                (test_df['test_game'] == test_game)
            ]
            
            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'Train: {game_names[train_game]} | Test: {game_names[test_game]}', 
                           fontsize=11, fontweight='bold')
                continue
            
            # Create pivot table
            pivot = subset.pivot_table(
                index='train_opponent',
                columns='test_opponent',
                values='cooperation_rate',
                aggfunc='mean'
            )
            
            # Reindex
            pivot = pivot.reindex(index=opponents, columns=opponents)
            
            # Plot heatmap
            sns.heatmap(pivot, ax=ax, cmap='RdYlGn', 
                       vmin=0, vmax=1, cbar=(j == 2),  # Only rightmost column gets colorbar
                       annot=True, fmt='.2f', 
                       cbar_kws={'shrink': 0.8, 'label': 'Cooperation Rate'},
                       annot_kws={'fontsize': 9})
            
            # Formatting
            ax.set_title(f'Train: {game_names[train_game]} | Test: {game_names[test_game]}', 
                       fontsize=11, fontweight='bold')
            ax.set_xlabel('Test Opponent' if i == 2 else '', fontsize=10)
            ax.set_ylabel('Train Opponent' if j == 0 else '', fontsize=10)
            ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    output_file = PLOTS_DIR / "cooperation_heatmap_3x3.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

def metric_3_4_kld_from_optimal(df_test, df_train):
    """
    Metric 3.4: KLD from optimal policy.
    
    For each test condition (game, opponent), compute KL divergence between 
    each agent's policy and the optimal policy. The optimal policy is defined
    as the final training cooperation rate of agents trained on that condition.
    
    KLD for Bernoulli: KL(p||q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
    where p is test agent's cooperation rate, q is optimal agent's rate.
    """
    print("\nGENERATING PLOT: Metric 3.4 - KLD from Optimal Policy")
    
    # Define games and opponents
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
            # Get final epoch cooperation rate for agents trained on this condition
            cond_data = df_train[
                (df_train['train_game'] == game) &
                (df_train['train_opponent'] == opp) &
                (df_train['epoch'] == df_train['epoch'].max())  # Final epoch
            ]
            
            if len(cond_data) > 0:
                optimal_policies[(game, opp)] = cond_data['cooperation_rate'].mean()
            else:
                print(f"  WARNING: No training data for {game}, opp={opp}")
    
    # For each test condition, compute KLD from optimal
    kld_results = []
    
    for test_game in games:
        for test_opp in opponents:
            # Get optimal policy for this test condition
            if (test_game, test_opp) not in optimal_policies:
                continue
            
            optimal_coop_rate = optimal_policies[(test_game, test_opp)]
            
            # Get all agents tested on this condition
            test_agents = df_test[
                (df_test['test_game'] == test_game) &
                (df_test['test_opponent'] == test_opp)
            ]
            
            # Compute KLD for each training condition
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
    
    # Save unified data
    kld_csv = UNIFIED_DATA_DIR / 'task_opponent_kld_from_optimal.csv'
    df_kld.to_csv(kld_csv, index=False)
    print(f"  Saved unified KLD data: {kld_csv.name}")
    print(f"  KLD records: {len(df_kld)}")
    
    if len(df_kld) == 0:
        print("  WARNING: No KLD data to plot (no agents tested on their training conditions)")
        return
    
    # Plot: 3 rows (one per training game), x-axis grouped by test game/opponent
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Color scheme for training opponents
    opp_colors = {0.1: '#2E86AB', 0.3: '#54A8C7', 0.5: '#9E9E9E', 0.7: '#E07A5F', 0.9: '#C1121F'}
    game_names = {
        'hawk-dove': 'HD',
        'prisoners-dilemma': 'PD',
        'stag-hunt': 'SH'
    }
    
    # Sort games alphabetically
    sorted_games = sorted(games)
    
    # Create x-axis positions: group by test game, then by test opponent
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
    
    # Get global y-axis limits for consistent scaling
    global_kld_min = df_kld['kld'].min()
    global_kld_max = df_kld['kld'].max()
    y_margin = (global_kld_max - global_kld_min) * 0.05  # 5% margin
    
    # Plot each training game
    for i, train_game in enumerate(sorted_games):
        ax = axes[i]
        
        train_data = df_kld[df_kld['train_game'] == train_game]
        
        if len(train_data) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Plot line for each training opponent
        for train_opp in opponents:
            opp_data = train_data[train_data['train_opponent'] == train_opp]
            
            if len(opp_data) == 0:
                continue
            
            # Sort by test game and opponent
            opp_data = opp_data.sort_values(['test_game', 'test_opponent'])
            
            # Get x positions and KLD values
            x_vals = [x_positions[(row['test_game'], row['test_opponent'])] 
                     for _, row in opp_data.iterrows()]
            y_vals = opp_data['kld'].values
            
            ax.plot(x_vals, y_vals, marker='o', linewidth=2, markersize=6,
                   label=f'Train opp={train_opp:.1f}', color=opp_colors[train_opp])
        
        # Formatting
        ax.set_title(f'Training Game: {game_names[train_game]}', 
                   fontsize=12, fontweight='bold')
        ax.set_ylabel('KL Divergence', fontsize=11)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels if i == 2 else [''] * len(x_labels), fontsize=9, rotation=0)
        ax.legend(fontsize=9, loc='best', ncol=5)
        ax.grid(True, alpha=0.3)
        
        # Set consistent y-axis limits
        ax.set_ylim(global_kld_min - y_margin, global_kld_max + y_margin)
        
        # Add vertical lines to separate games
        for sep_pos in [4.5, 9.5]:  # Between games (5 opponents per game)
            ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    axes[2].set_xlabel('Test Game and Opponent Defection Probability', fontsize=11)
    
    plt.suptitle('Task-Opponent Setup: KL Divergence from Optimal Policy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_3.4_kld_from_optimal.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

def metric_3_5_cluster_analysis(df_test):
    """
    Metric 3.5: Cluster analysis of agents based on behavior (AGGREGATED ACROSS SEEDS).
    
    Shows 15 aggregated conditions (3 games × 5 opponents) with error bars.
    X-axis: Mean cooperation probability across all test conditions
    Y-axis: Mean normalized reward across all test conditions
    Color: Training opponent
    Shape: Training game
    """
    print("\nGENERATING PLOT: Metric 3.5 - Cluster Analysis (Aggregated)")
    
    # First compute per-agent metrics
    agent_metrics = []
    
    for model_id in df_test['model_id'].unique():
        agent_data = df_test[df_test['model_id'] == model_id]
        
        train_game = agent_data['train_game'].iloc[0]
        train_opponent = agent_data['train_opponent'].iloc[0]
        seed = agent_data['seed'].iloc[0]
        
        # Aggregates across all test conditions
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
    
    # Aggregate across seeds for each (game, opponent) condition
    df_cluster = df_individual.groupby(['train_game', 'train_opponent']).agg({
        'mean_cooperation': ['mean', 'sem'],
        'mean_normalized_reward': ['mean', 'sem']
    }).reset_index()
    
    # Flatten column names
    df_cluster.columns = ['train_game', 'train_opponent', 'coop_mean', 'coop_sem', 'reward_mean', 'reward_sem']
    
    print(f"  Aggregated {len(df_individual)} agents into {len(df_cluster)} conditions")
    
    # Save both individual and aggregated data
    individual_csv = UNIFIED_DATA_DIR / 'task_opponent_cluster_analysis_individual.csv'
    df_individual.to_csv(individual_csv, index=False)
    print(f"  Saved individual agent data: {individual_csv.name}")
    
    cluster_csv = UNIFIED_DATA_DIR / 'task_opponent_cluster_analysis_aggregated.csv'
    df_cluster.to_csv(cluster_csv, index=False)
    print(f"  Saved aggregated data: {cluster_csv.name}")
    
    # Plot scatter with error bars for aggregated data (15 points)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.set_style("whitegrid")
    
    # Color by opponent
    opp_colors = {0.1: '#2E86AB', 0.3: '#54A8C7', 0.5: '#9E9E9E', 0.7: '#E07A5F', 0.9: '#C1121F'}
    
    # Markers by game
    game_markers = {'prisoners-dilemma': 'o', 'hawk-dove': 's', 'stag-hunt': '^'}
    
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    opponents = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Plot each condition with error bars
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
                       markersize=12, 
                       alpha=0.7, 
                       markeredgecolor='black', 
                       markeredgewidth=1.5,
                       capsize=4,
                       capthick=1.5,
                       elinewidth=1.5,
                       label=f'{game[:2].upper()}, opp={opp:.1f}')
    
    ax.set_xlabel('Mean Cooperation Probability (across all test conditions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Normalized Reward (across all test conditions)', fontsize=12, fontweight='bold')
    ax.set_title('Task-Opponent Setup: Agent Clustering by Behavior (n=5 seeds/condition)', fontsize=14, fontweight='bold')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    
    # Game markers
    game_legend = [Line2D([0], [0], marker=game_markers[g], color='w', 
                          markerfacecolor='gray', markersize=10, label=g)
                   for g in games]
    
    # Opponent colors
    opp_legend = [Line2D([0], [0], marker='o', color='w', 
                         markerfacecolor=opp_colors[o], markersize=10, label=f'opp={o:.1f}')
                  for o in opponents]
    
    # Two legends in top right corner
    first_legend = ax.legend(handles=game_legend, title='Game', 
                            loc='upper right', bbox_to_anchor=(1.0, 1.0), 
                            fontsize=10, title_fontsize=11)
    ax.add_artist(first_legend)
    ax.legend(handles=opp_legend, title='Opponent', 
             loc='upper right', bbox_to_anchor=(1.0, 0.65),
             fontsize=10, title_fontsize=11)
    
    ax.grid(True, alpha=0.3)
    
    # Add reference lines at 0.5
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_3.5_cluster_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

def metric_3_6_generalization_ratio(df_test):
    """
    Metric 3.6: Generalization ratio plot (AGGREGATED ACROSS SEEDS).
    
    X-axis: Mean cooperation probability across all test conditions
    Y-axis: Generalization performance ratio
            ratio = mean_reward(non_training_conditions) / mean_reward(training_condition)
    
    Ratio = 1.0: Perfect generalization (equal performance on non-training vs training)
    Ratio > 1.0: Positive transfer (better on non-training than training)
    Ratio < 1.0: Overfitting (worse on non-training than training)
    """
    print("\nGENERATING PLOT: Metric 3.6 - Generalization Ratio (Aggregated)")
    
    # Compute metrics per agent (model_id)
    agent_metrics = []
    
    for model_id in df_test['model_id'].unique():
        agent_data = df_test[df_test['model_id'] == model_id]
        
        train_game = agent_data['train_game'].iloc[0]
        train_opponent = agent_data['train_opponent'].iloc[0]
        seed = agent_data['seed'].iloc[0]
        
        # Calculate mean normalized reward on training condition (same game + opponent)
        training_cond_data = agent_data[
            (agent_data['test_game'] == train_game) &
            (agent_data['test_opponent'] == train_opponent)
        ]
        mean_training_reward = training_cond_data['normalized_reward'].mean()
        
        # Calculate mean normalized reward on NON-training conditions (all other 14 conditions)
        non_training_data = agent_data[
            ~((agent_data['test_game'] == train_game) &
              (agent_data['test_opponent'] == train_opponent))
        ]
        mean_non_training_reward = non_training_data['normalized_reward'].mean()
        
        # Calculate ratio (non-training / training performance)
        if mean_training_reward > 0 and not np.isnan(mean_training_reward):
            generalization_ratio = mean_non_training_reward / mean_training_reward
        else:
            generalization_ratio = np.nan
        
        # Mean cooperation across all test conditions
        mean_coop = agent_data['cooperation_rate'].mean()
        
        agent_metrics.append({
            'model_id': model_id,
            'train_game': train_game,
            'train_opponent': train_opponent,
            'seed': seed,
            'mean_cooperation': mean_coop,
            'generalization_ratio': generalization_ratio,
            'mean_training_reward': mean_training_reward,
            'mean_non_training_reward': mean_non_training_reward
        })
    
    df_individual = pd.DataFrame(agent_metrics)
    
    # Aggregate across seeds for each (game, opponent) condition
    df_agg = df_individual.groupby(['train_game', 'train_opponent']).agg({
        'mean_cooperation': ['mean', 'sem'],
        'generalization_ratio': ['mean', 'sem']
    }).reset_index()
    
    # Flatten column names
    df_agg.columns = ['train_game', 'train_opponent', 'coop_mean', 'coop_sem', 'ratio_mean', 'ratio_sem']
    
    print(f"  Aggregated {len(df_individual)} agents into {len(df_agg)} conditions")
    
    # Save both individual and aggregated data
    individual_csv = UNIFIED_DATA_DIR / 'task_opponent_generalization_ratio_individual.csv'
    df_individual.to_csv(individual_csv, index=False)
    print(f"  Saved individual agent data: {individual_csv.name}")
    
    agg_csv = UNIFIED_DATA_DIR / 'task_opponent_generalization_ratio_aggregated.csv'
    df_agg.to_csv(agg_csv, index=False)
    print(f"  Saved aggregated data: {agg_csv.name}")
    
    # Plot scatter with error bars for aggregated data (15 points)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.set_style("whitegrid")
    
    # Color by opponent
    opp_colors = {0.1: '#2E86AB', 0.3: '#54A8C7', 0.5: '#9E9E9E', 0.7: '#E07A5F', 0.9: '#C1121F'}
    
    # Markers by game
    game_markers = {'prisoners-dilemma': 'o', 'hawk-dove': 's', 'stag-hunt': '^'}
    
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    opponents = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Plot each condition with error bars
    for game in games:
        for opp in opponents:
            game_opp_data = df_agg[
                (df_agg['train_game'] == game) &
                (df_agg['train_opponent'] == opp)
            ]
            
            if len(game_opp_data) == 0:
                continue
            
            ax.errorbar(game_opp_data['coop_mean'], 
                       game_opp_data['ratio_mean'],
                       xerr=game_opp_data['coop_sem'],
                       yerr=game_opp_data['ratio_sem'],
                       fmt=game_markers[game], 
                       color=opp_colors[opp],
                       markersize=12, 
                       alpha=0.7, 
                       markeredgecolor='black', 
                       markeredgewidth=1.5,
                       capsize=4,
                       capthick=1.5,
                       elinewidth=1.5,
                       label=f'{game[:2].upper()}, opp={opp:.1f}')
    
    ax.set_xlabel('Mean Cooperation Probability (across all test conditions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Generalization Ratio\n(Non-Training Conditions / Training Condition)', fontsize=12, fontweight='bold')
    ax.set_title('Task-Opponent Setup: Generalization Performance (n=5 seeds/condition)', fontsize=14, fontweight='bold')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    
    # Game markers
    game_legend = [Line2D([0], [0], marker=game_markers[g], color='w', 
                          markerfacecolor='gray', markersize=10, label=g)
                   for g in games]
    
    # Opponent colors
    opp_legend = [Line2D([0], [0], marker='o', color='w', 
                         markerfacecolor=opp_colors[o], markersize=10, label=f'opp={o:.1f}')
                  for o in opponents]
    
    # Two legends in top right corner
    first_legend = ax.legend(handles=game_legend, title='Game', 
                            loc='upper right', bbox_to_anchor=(1.0, 1.0), 
                            fontsize=10, title_fontsize=11)
    ax.add_artist(first_legend)
    ax.legend(handles=opp_legend, title='Opponent', 
             loc='upper right', bbox_to_anchor=(1.0, 0.65),
             fontsize=10, title_fontsize=11)
    
    ax.grid(True, alpha=0.3)
    
    # Add reference line at ratio = 1 (perfect generalization)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Perfect Generalization')
    
    # Trim y-axis to relevant range (filter out NaN values)
    all_ratios = df_agg['ratio_mean'].values
    all_sems = df_agg['ratio_sem'].values
    
    # Filter out NaN values before computing limits
    valid_lower = all_ratios - all_sems
    valid_upper = all_ratios + all_sems
    valid_lower_clean = valid_lower[~np.isnan(valid_lower)]
    valid_upper_clean = valid_upper[~np.isnan(valid_upper)]
    
    # Only set limits if we have valid data
    if len(valid_lower_clean) > 0 and len(valid_upper_clean) > 0:
        y_min = np.min(valid_lower_clean) * 0.95
        y_max = np.max(valid_upper_clean) * 1.05
        ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'metric_3.6_generalization_ratio.png'
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
        
        if len(training_df) > 0:
            metric_3_4_kld_from_optimal(test_df, training_df)
        else:
            print("\n⚠ Skipping KLD metric - no training data available")
        
        metric_3_5_cluster_analysis(test_df)
        metric_3_6_generalization_ratio(test_df)
    
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
    print(f"  Plots:")
    print(f"    - cooperation_vs_epoch_3x5.png")
    print(f"    - normalized_reward_heatmap_3x5.png")
    print(f"    - cooperation_heatmap_3x5.png")
    print(f"    - metric_3.4_kld_from_optimal.png")
    print(f"    - metric_3.5_cluster_analysis.png")
    print("=" * 80)

if __name__ == "__main__":
    main()
