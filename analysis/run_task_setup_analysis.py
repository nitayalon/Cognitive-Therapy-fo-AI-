"""
Whole Population (Task Setup) - Complete Analysis Protocol
Follows analysis_guidelines.md protocol for Section 1 analysis.

Executes metrics 3.1-3.5 and 4 for the Task Setup:
- Training: experiments/whole_population_train_913310/
- Testing: experiments/whole_population_test_912631/
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Paths
BASE_DIR = Path(__file__).parent.parent
TRAIN_DIR = BASE_DIR / 'experiments' / 'whole_population_train_913310' / 'training'
TEST_DIR = BASE_DIR / 'experiments' / 'whole_population_test_912631' / 'testing'
OUTPUT_DIR = BASE_DIR / 'Results' / 'task_setup'
DATA_DIR = OUTPUT_DIR / 'unified_data'  # For CSV outputs from ETL

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Constants
GAMES = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
OPPONENTS = [0.1, 0.3, 0.5, 0.7, 0.9]

# Load master training registry to get correct game assignments
TRAIN_REGISTRY_PATH = BASE_DIR / 'experiments' / 'whole_population_train_913310' / 'seed_manifests' / 'MASTER_TRAINING_REGISTRY.csv'
TRAIN_REGISTRY = pd.read_csv(TRAIN_REGISTRY_PATH)

# Create mapping: task_id -> game (from condition_id)
CONDITION_TO_GAME = {
    0: 'prisoners-dilemma',
    1: 'hawk-dove',
    2: 'stag-hunt'
}
TASK_TO_GAME = {}
for _, row in TRAIN_REGISTRY.iterrows():
    task_id = row['array_task_id']
    condition_id = row['condition_id']
    TASK_TO_GAME[task_id] = CONDITION_TO_GAME[condition_id]

PAYOFF_MATRICES = {
    'prisoners-dilemma': {'R': 3, 'S': 0, 'T': 5, 'P': 1},
    'stag-hunt': {'R': 4, 'S': 0, 'T': 3, 'P': 2},
    'hawk-dove': {'R': 2, 'S': 1, 'T': 3, 'P': 0}
}

def normalize_reward(reward: float, game: str) -> float:
    """Normalize reward to [0, 1] within task."""
    payoffs = PAYOFF_MATRICES[game]
    all_payoffs = [payoffs['R'], payoffs['S'], payoffs['T'], payoffs['P']]
    min_r, max_r = min(all_payoffs), max(all_payoffs)
    rng = max_r - min_r
    if rng == 0:
        return 0.5
    return np.clip((reward - min_r) / rng, 0, 1)

#############################################################################
# STEP 2.1: ETL - Data Unification
#############################################################################

def etl_training_data():
    """
    Extract training data for metric 3.1 (P(cooperate) vs epoch).
    Output: task_setup_training_cooperation.csv
    """
    print("\n" + "="*70)
    print("ETL: Extracting Training Data")
    print("="*70)
    
    records = []
    
    # 15 models: 3 games × 5 seeds
    task_dirs = sorted(list(TRAIN_DIR.glob('whole_population_task_*')))
    print(f"Found {len(task_dirs)} training directories")
    
    for task_dir in task_dirs:
        # Load training logs from results directory
        task_id = int(task_dir.name.split('_')[3])
        training_log = task_dir / 'results' / f'training_task_{task_id}_metrics.csv'
        
        if not training_log.exists():
            print(f"  Warning: Missing training log for {task_dir.name}")
            continue
        
        # Get correct game from master registry
        train_game = TASK_TO_GAME.get(task_id, 'unknown')
        seed = task_id % 5
        
        # Load training log
        df = pd.read_csv(training_log)
        
        for _, row in df.iterrows():
            records.append({
                'task_id': task_id,
                'game': train_game,
                'seed': seed,
                'epoch': row['epoch'],
                'cooperation_rate': row.get('epoch_average_cooperation_rate', np.nan),
                'avg_reward': row.get('epoch_cumulative_reward', np.nan),
                'loss': row.get('total_loss', np.nan)
            })
    
    df_out = pd.DataFrame(records)
    output_file = DATA_DIR / 'task_setup_training_cooperation.csv'
    df_out.to_csv(output_file, index=False)
    print(f"✅ Saved {len(df_out)} training records to {output_file.name}")
    return df_out

def etl_test_data():
    """
    Extract test data for metrics 3.2-3.5.
    Output: task_setup_test_results.csv
    """
    print("\n" + "="*70)
    print("ETL: Extracting Test Data")
    print("="*70)
    
    records = []
    
    # 225 test results: 15 models × 15 test conditions (3 games × 5 opponents)
    test_tasks = sorted(list(TEST_DIR.glob('whole_population_task_*')))
    print(f"Found {len(test_tasks)} test directories")
    
    for test_dir in test_tasks:
        # Load config
        config_file = test_dir / 'experiment_config.json'
        if not config_file.exists():
            continue
        
        with open(config_file) as f:
            config = json.load(f)
        
        task_id = int(test_dir.name.split('_')[3])
        model_id = task_id // 15  # Which trained model (0-14)
        
        # Get correct game from master registry
        train_game = TASK_TO_GAME.get(model_id, 'unknown')
        seed = model_id % 5
        
        test_game = config.get('test_game', 'unknown')
        test_opp_probs = config.get('test_opponent_probabilities', [])
        if not test_opp_probs:
            continue
        test_opp = test_opp_probs[0]
        
        # Load test results - they're in results directory with specific naming
        results_dir = test_dir / 'results'
        # Find CSV files matching this model
        test_csvs = list(results_dir.glob(f'eval_model_{model_id}_*.csv'))
        
        if not test_csvs:
            print(f"  Warning: Missing test CSV for {test_dir.name}")
            continue
        
        # Use the first matching CSV (should only be one per test task)
        test_csv = test_csvs[0]
        df = pd.read_csv(test_csv)
        
        # Aggregate across test games
        avg_coop = df['mean_cooperation_rate'].values[0] if 'mean_cooperation_rate' in df.columns else np.nan
        avg_reward = df['mean_reward'].values[0] if 'mean_reward' in df.columns else np.nan
        norm_reward = normalize_reward(avg_reward, test_game)
        
        records.append({
            'task_id': task_id,
            'model_id': model_id,
            'train_game': train_game,
            'test_game': test_game,
            'test_opponent': test_opp,
            'seed': seed,
            'cooperation_rate': avg_coop,
            'avg_reward': avg_reward,
            'normalized_reward': norm_reward
        })
    
    df_out = pd.DataFrame(records)
    output_file = DATA_DIR / 'task_setup_test_results.csv'
    df_out.to_csv(output_file, index=False)
    print(f"✅ Saved {len(df_out)} test records to {output_file.name}")
    return df_out

#############################################################################
# STEP 2.2: Analysis and Plotting
#############################################################################

def metric_3_1_cooperation_vs_epoch(df_train):
    """
    3.1: P(cooperate) as a function of training duration.
    Line plot: x=epoch, y=P(cooperate), color=game
    """
    print("\n" + "="*70)
    print("Metric 3.1: Cooperation Rate vs Training Epoch")
    print("="*70)
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    for game in GAMES:
        game_data = df_train[df_train['game'] == game]
        
        # Aggregate across seeds
        agg = game_data.groupby('epoch')['cooperation_rate'].agg(['mean', 'std']).reset_index()
        
        plt.plot(agg['epoch'], agg['mean'], label=game, linewidth=2)
        plt.fill_between(agg['epoch'], 
                         agg['mean'] - agg['std'], 
                         agg['mean'] + agg['std'], 
                         alpha=0.2)
    
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Cooperation Rate', fontsize=12)
    plt.title('Task Setup: Cooperation Rate During Training', fontsize=14, fontweight='bold')
    plt.legend(title='Game')
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / 'metric_3.1_cooperation_vs_epoch.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot to {output_file.name}")
    plt.close()

def metric_3_2_normalized_reward_heatmap(df_test):
    """
    3.2: Normalized generalization heatmap.
    Taskwise normalized reward, tasks on y-axis, opponents on x-axis.
    Training cell left empty.
    """
    print("\n" + "="*70)
    print("Metric 3.2: Normalized Reward Heatmap")
    print("="*70)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.set_style("white")
    
    for idx, train_game in enumerate(GAMES):
        # Filter to models trained on this game
        train_data = df_test[df_test['train_game'] == train_game]
        
        # Create heatmap matrix: rows=test_game, cols=opponent
        matrix = np.zeros((3, 5))
        
        for i, test_game in enumerate(GAMES):
            for j, opp in enumerate(OPPONENTS):
                subset = train_data[
                    (train_data['test_game'] == test_game) & 
                    (train_data['test_opponent'] == opp)
                ]
                
                if len(subset) > 0:
                    matrix[i, j] = subset['avg_reward'].mean()  # Use raw rewards
                else:
                    matrix[i, j] = np.nan
        
        # Normalize within this subplot to [0, 1]
        matrix_min = np.nanmin(matrix)
        matrix_max = np.nanmax(matrix)
        
        if matrix_max > matrix_min:
            matrix_normalized = (matrix - matrix_min) / (matrix_max - matrix_min)
        else:
            matrix_normalized = matrix * 0  # All zeros if no variation
        
        # Plot
        ax = axes[idx]
        sns.heatmap(matrix_normalized, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
                    vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Reward'},
                    xticklabels=[f'{o:.1f}' for o in OPPONENTS],
                    yticklabels=GAMES)
        ax.set_title(f'Trained on: {train_game}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Test Opponent Defection Probability')
        ax.set_ylabel('Test Game')
    
    plt.suptitle('Task Setup: Normalized Reward Generalization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / 'metric_3.2_normalized_reward_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot to {output_file.name}")
    plt.close()

def metric_3_3_cooperation_heatmap(df_test):
    """
    3.3: Probability to cooperate heatmap.
    Same layout as 3.2.
    """
    print("\n" + "="*70)
    print("Metric 3.3: Cooperation Rate Heatmap")
    print("="*70)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.set_style("white")
    
    for idx, train_game in enumerate(GAMES):
        train_data = df_test[df_test['train_game'] == train_game]
        
        matrix = np.zeros((3, 5))
        
        for i, test_game in enumerate(GAMES):
            for j, opp in enumerate(OPPONENTS):
                subset = train_data[
                    (train_data['test_game'] == test_game) & 
                    (train_data['test_opponent'] == opp)
                ]
                
                if len(subset) > 0:
                    matrix[i, j] = subset['cooperation_rate'].mean()
                else:
                    matrix[i, j] = np.nan
        
        ax = axes[idx]
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
                    vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Cooperation Rate'},
                    xticklabels=[f'{o:.1f}' for o in OPPONENTS],
                    yticklabels=GAMES)
        ax.set_title(f'Trained on: {train_game}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Test Opponent Defection Probability')
        ax.set_ylabel('Test Game')
    
    plt.suptitle('Task Setup: Cooperation Rate Generalization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / 'metric_3.3_cooperation_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot to {output_file.name}")
    plt.close()

#############################################################################
# Main Execution
#############################################################################

def main():
    print("=" * 70)
    print("WHOLE POPULATION (TASK SETUP) - COMPLETE ANALYSIS")
    print("=" * 70)
    print(f"Training: {TRAIN_DIR}")
    print(f"Testing: {TEST_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Step 2.1: ETL
    df_train = etl_training_data()
    df_test = etl_test_data()
    
    # Step 2.2: Metrics
    metric_3_1_cooperation_vs_epoch(df_train)
    metric_3_2_normalized_reward_heatmap(df_test)
    metric_3_3_cooperation_heatmap(df_test)
    
    # TODO: Implement metrics 3.4, 3.5, and 4
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print(f"Unified data saved to: {DATA_DIR}")
    print("\nNext steps:")
    print("  - Implement metric 3.4 (KLD from optimal policy)")
    print("  - Implement metric 3.5 (Cluster analysis)")
    print("  - Implement metric 4 (Representation analysis)")

if __name__ == "__main__":
    main()
