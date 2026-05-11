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
    
    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput Directory: {OUTPUT_DIR}")
    print(f"\nGenerated Files:")
    print(f"  Data:")
    print(f"    - task_opponent_training_cooperation.csv ({len(training_df)} rows)")
    print(f"    - task_opponent_test_results.csv ({len(test_df)} rows)")
    print(f"  Plots:")
    print(f"    - cooperation_vs_epoch_3x5.png")
    print(f"    - normalized_reward_heatmap_3x5.png")
    print(f"    - cooperation_heatmap_3x5.png")
    print("=" * 80)

if __name__ == "__main__":
    main()
