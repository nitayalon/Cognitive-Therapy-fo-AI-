#!/usr/bin/env python3
"""
Complexity Analysis Entry Point
===============================

Runs reciprocity-representation coupling analysis for both setups:
- Task-opponent (generalization matrix): 75 models → 15 aggregated
- Task (whole population): 15 models → 3 aggregated

Usage:
    python analysis/run_complexity_analysis.py --setup task-opponent
    python analysis/run_complexity_analysis.py --setup task
    python analysis/run_complexity_analysis.py --setup all

Author: Research Team
Date: May 12, 2026
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from cognitive_therapy_ai.network import GameLSTM
from complexity_metrics import (
    compute_task_complexity,
    compute_opponent_complexity,
    compute_behavioral_complexity,
    compute_representational_complexity,
    generate_test_states
)
from reciprocity_representation_coupling import (
    component_1_task_representation,
    component_2_opponent_social_learning,
    component_3_representation_behavior_coupling,
    component_4_generalization_similarity,
    component_5_integrated_complexity
)
from metric_4_embedding_analysis import (compute_representational_similarity, extract_hidden_state_representations)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> GameLSTM:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle nested state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Infer architecture
    input_size = 9
    hidden_size = 128
    
    model = GameLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=2,
        dropout=0.1
    )
    
    model.load_state_dict(state_dict)
    model.eval()
    return model


def analyze_task_opponent_setup(output_base: Path):
    """
    Analyze task-opponent setup (15 aggregated conditions from 75 models).
    """
    print("\n" + "="*80)
    print("TASK-OPPONENT SETUP: Complexity Analysis")
    print("="*80)
    
    # Paths
    train_dir = project_root / 'experiments' / 'generalization_matrix_train_913243' / 'training'
    test_dir = project_root / 'experiments' / 'generalization_matrix_test_913245' / 'testing'
    
    output_dir = output_base / 'task_opponent_setup' / 'complexity_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    data_dir = output_dir / 'unified_data'
    data_dir.mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load existing analysis data
    training_coop_csv = output_base / 'task_opponent_setup' / 'unified_data' / 'task_opponent_training_cooperation.csv'
    test_results_csv = output_base / 'task_opponent_setup' / 'unified_data' / 'task_opponent_test_results.csv'
    
    if not training_coop_csv.exists() or not test_results_csv.exists():
        print("Error: Required CSV files not found. Run run_task_opponent_setup_analysis.py first.")
        return
    
    df_training = pd.read_csv(training_coop_csv)
    df_test = pd.read_csv(test_results_csv)
    
    # Derive condition_id from task_id (task_id = condition_id * 5 + seed)
    df_training['condition_id'] = df_training['task_id'] // 5
    
    # Get unique conditions (15 aggregated)
    conditions = df_training[['condition_id', 'train_game', 'train_opponent']].drop_duplicates().sort_values('condition_id')
    
    print(f"\nFound {len(conditions)} conditions")
    
    # Generate test states for representation analysis
    test_states = generate_test_states(n_samples=1000, seed=42)
    
    # Compute complexity metrics for each condition
    complexity_records = []
    cka_models = {}  # Store models for CKA computation
    
    for idx, row in conditions.iterrows():
        condition_id = int(row['condition_id'])
        train_game = row['train_game']
        train_opponent = float(row['train_opponent'])
        
        print(f"\n[{idx+1}/{len(conditions)}] Condition {condition_id}: {train_game}, opponent={train_opponent}")
        
        # 1. Task complexity
        task_complexity = compute_task_complexity(train_game)
        
        # 2. Opponent complexity
        opp_complexity = compute_opponent_complexity(train_opponent)
        
        # 3. Behavioral complexity
        # Try to load detailed logs, otherwise use aggregated metrics
        all_actions = []
        all_opp_actions = []
        
        for seed in range(5):
            model_id = condition_id * 5 + seed
            
            # Find test directories for this model
            test_dirs = list(test_dir.glob(f'model_{model_id}_test_cond_*'))
            
            for test_dir_path in test_dirs:
                task_dirs = list(test_dir_path.glob('generalization_matrix_task_*'))
                if not task_dirs:
                    continue
                
                task_dir = task_dirs[0]
                log_dir = task_dir / 'logs'
                
                eval_dirs = list(log_dir.glob('eval_cond_*'))
                if not eval_dirs:
                    continue
                
                eval_dir = eval_dirs[0]
                detailed_log = eval_dir / 'detailed_testing_log.csv'
                
                if detailed_log.exists():
                    try:
                        log_df = pd.read_csv(detailed_log)
                        all_actions.extend(log_df['agent_sampled_action'].values)
                        all_opp_actions.extend(log_df['opponent_actual_action'].values)
                    except Exception as e:
                        print(f"  Warning: Error loading {detailed_log}: {e}")
        
        if len(all_actions) > 0:
            # Compute from detailed logs
            actions_array = np.array(all_actions)
            opp_actions_array = np.array(all_opp_actions)
            behavior_complexity = compute_behavioral_complexity(actions_array, opp_actions_array)
            print(f"  Behavioral complexity from {len(all_actions)} detailed actions")
        else:
            # Fall back to aggregated metrics from test CSV
            condition_test = df_test[
                (df_test['train_game'] == train_game) &
                (df_test['train_opponent'] == train_opponent)
            ]
            
            if len(condition_test) > 0:
                # Use cooperation rate as proxy
                mean_coop = condition_test['cooperation_rate'].mean()
                std_coop = condition_test['cooperation_rate'].std()
                
                # Compute simple entropy from mean cooperation
                p_coop = mean_coop
                if p_coop == 0 or p_coop == 1:
                    policy_entropy = 0
                else:
                    policy_entropy = -p_coop * np.log2(p_coop) - (1-p_coop) * np.log2(1-p_coop)
                
                behavior_complexity = {
                    'mean_cooperation': mean_coop,
                    'policy_entropy': policy_entropy,
                    'behavioral_variability': std_coop,
                    'reciprocity_strength': np.nan,  # Can't compute without action sequences
                    'p_coop_given_opp_coop': np.nan,
                    'p_coop_given_opp_defect': np.nan,
                    'temporal_consistency': np.nan
                }
                print(f"  Behavioral complexity from aggregated metrics (n={len(condition_test)} test conditions)")
            else:
                behavior_complexity = {
                    'reciprocity_strength': np.nan,
                    'policy_entropy': np.nan,
                    'mean_cooperation': np.nan,
                    'behavioral_variability': np.nan,
                    'temporal_consistency': np.nan,
                    'p_coop_given_opp_coop': np.nan,
                    'p_coop_given_opp_defect': np.nan
                }
                print(f"  Warning: No test data found for condition {condition_id}")
        
        # 4. Representational complexity (average across seeds)
        # Load all 5 seeds for this condition
        seed_repr_metrics = []
        for seed in range(5):
            # Note: task_id in directory name = condition_id (NOT condition_id * 5 + seed)
            checkpoint_dirs = list(train_dir.glob(f'condition_{condition_id}_seed_{seed}/generalization_matrix_task_{condition_id}_*'))
            
            if len(checkpoint_dirs) > 0:
                checkpoint_path = checkpoint_dirs[0] / 'checkpoints' / f'{train_game}_final_checkpoint.pth'
                
                if checkpoint_path.exists():
                    try:
                        model = load_checkpoint(checkpoint_path, device)
                        repr_complexity = compute_representational_complexity(model, test_states, device)
                        seed_repr_metrics.append(repr_complexity)
                        
                        # Store first model of each condition for CKA
                        if seed == 0:
                            cka_models[condition_id] = model
                    except Exception as e:
                        print(f"  Error loading checkpoint for seed {seed}: {e}")
                else:
                    if seed == 0:  # Only print for first seed to avoid spam
                        print(f"  Checkpoint not found: {checkpoint_path}")
            else:
                if seed == 0:
                    print(f"  Task directory not found for condition {condition_id}, seed {seed}")
        
        # Aggregate representation metrics
        if len(seed_repr_metrics) > 0:
            repr_complexity = {
                key: np.mean([m[key] for m in seed_repr_metrics])
                for key in seed_repr_metrics[0].keys()
            }
        else:
            repr_complexity = {
                'social_ratio': np.nan,
                'effective_dimensionality': np.nan,
                'weight_l2_norm': np.nan,
                'embedding_specialization': np.nan,
                'activation_sparsity': np.nan
            }
        
        # Combine all metrics
        record = {
            'condition_id': condition_id,
            'train_game': train_game,
            'train_opponent': train_opponent,
            **task_complexity,
            **opp_complexity,
            **behavior_complexity,
            **repr_complexity
        }
        complexity_records.append(record)
    
    # Create complexity DataFrame
    df_complexity = pd.DataFrame(complexity_records)
    
    # Save
    csv_path = data_dir / 'complexity_metrics.csv'
    df_complexity.to_csv(csv_path, index=False)
    print(f"\nSaved complexity metrics: {csv_path}")
    
    # Compute CKA similarity matrix
    print("\nComputing CKA similarity matrix...")
    n_conditions = len(cka_models)
    cka_matrix = np.zeros((n_conditions, n_conditions))
    
    condition_ids = sorted(cka_models.keys())
    for i, cond_i in enumerate(condition_ids):
        for j, cond_j in enumerate(condition_ids):
            if i <= j:
                if i == j:
                    cka_matrix[i, j] = 1.0
                else:
                    try:
                        # Extract hidden states from both models
                        hidden_i = extract_hidden_state_representations(cka_models[cond_i], test_states, device)
                        hidden_j = extract_hidden_state_representations(cka_models[cond_j], test_states, device)
                        
                        # Compute CKA similarity
                        cka_sim = compute_representational_similarity(
                            hidden_i,
                            hidden_j,
                            method='cka'
                        )
                        cka_matrix[i, j] = cka_sim
                        cka_matrix[j, i] = cka_sim
                    except Exception as e:
                        print(f"  Warning: CKA computation failed for {cond_i}, {cond_j}: {e}")
                        cka_matrix[i, j] = np.nan
                        cka_matrix[j, i] = np.nan
    
    # Save CKA matrix
    np.save(data_dir / 'cka_similarity_matrix.npy', cka_matrix)
    print(f"Saved CKA matrix: {data_dir / 'cka_similarity_matrix.npy'}")
    
    # Run all 5 components
    print("\n" + "="*80)
    print("RUNNING ANALYSIS COMPONENTS")
    print("="*80)
    
    all_figures = {}
    
    # Component 1: Task → Representation
    summary_1, figs_1 = component_1_task_representation(df_complexity, plots_dir)
    all_figures.update(figs_1)
    if len(summary_1) > 0:
        summary_1.to_csv(data_dir / 'component_1_summary.csv', index=True)
    
    # Component 2: Opponent → Social Learning
    summary_2, figs_2 = component_2_opponent_social_learning(df_complexity, plots_dir)
    all_figures.update(figs_2)
    if len(summary_2) > 0:
        summary_2.to_csv(data_dir / 'component_2_summary.csv', index=True)
    
    # Component 3: Representation-Behavior Coupling
    summary_3, figs_3 = component_3_representation_behavior_coupling(df_complexity, cka_matrix, plots_dir)
    all_figures.update(figs_3)
    if len(summary_3) > 0:
        summary_3.to_csv(data_dir / 'component_3_summary.csv', index=False)
    
    # Component 4: Generalization Similarity
    summary_4, figs_4 = component_4_generalization_similarity(df_complexity, df_test, cka_matrix, plots_dir)
    all_figures.update(figs_4)
    if len(summary_4) > 0:
        summary_4.to_csv(data_dir / 'component_4_summary.csv', index=False)
    
    # Component 5: Integrated Complexity
    summary_5, figs_5 = component_5_integrated_complexity(df_complexity, plots_dir)
    all_figures.update(figs_5)
    if len(summary_5) > 0:
        summary_5.to_csv(data_dir / 'component_5_summary.csv', index=False)
    
    # Save all figures
    for fig_name, fig in all_figures.items():
        fig_path = plots_dir / f'{fig_name}.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {fig_path}")
    
    print("\n" + "="*80)
    print("TASK-OPPONENT ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")


def analyze_task_setup(output_base: Path):
    """
    Analyze task setup (3 aggregated conditions from 15 models).
    """
    print("\n" + "="*80)
    print("TASK SETUP: Complexity Analysis")
    print("="*80)
    
    # Paths
    train_dir = project_root / 'experiments' / 'whole_population_train_913310' / 'training'
    test_dir = project_root / 'experiments' / 'whole_population_test_912631' / 'testing'
    
    output_dir = output_base / 'task_setup' / 'complexity_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    data_dir = output_dir / 'unified_data'
    data_dir.mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load existing analysis data
    training_coop_csv = output_base / 'task_setup' / 'unified_data' / 'task_training_cooperation.csv'
    test_results_csv = output_base / 'task_setup' / 'unified_data' / 'task_test_results.csv'
    
    if not training_coop_csv.exists() or not test_results_csv.exists():
        print("Error: Required CSV files not found. Run run_task_setup_analysis.py first.")
        return
    
    df_training = pd.read_csv(training_coop_csv)
    df_test = pd.read_csv(test_results_csv)
    
    # Get unique games (3 conditions)
    games = df_training['train_game'].unique()
    
    print(f"\nFound {len(games)} games")
    
    # Generate test states
    test_states = generate_test_states(n_samples=1000, seed=42)
    
    # Compute complexity metrics for each game
    complexity_records = []
    cka_models = {}
    
    game_id_map = {'prisoners-dilemma': 0, 'hawk-dove': 1, 'stag-hunt': 2}
    
    for game in sorted(games):
        game_id = game_id_map.get(game, 0)
        print(f"\n[{game_id+1}/{len(games)}] Game: {game}")
        
        # 1. Task complexity
        task_complexity = compute_task_complexity(game)
        
        # 2. Opponent complexity (N/A - trained on all opponents)
        opp_complexity = {
            'opponent_defect_prob': np.nan,
            'stationarity': 1.0,
            'predictability': np.nan,
            'adaptation_requirement': np.nan
        }
        
        # 3. Behavioral complexity
        game_test = df_test[df_test['train_game'] == game]
        
        if len(game_test) > 0:
            actions = game_test['agent_action'].values
            opponent_actions = game_test['opponent_action'].values
            behavior_complexity = compute_behavioral_complexity(actions, opponent_actions)
        else:
            behavior_complexity = {
                'reciprocity_strength': np.nan,
                'policy_entropy': np.nan,
                'mean_cooperation': np.nan
            }
        
        # 4. Representational complexity (average across 5 seeds)
        seed_repr_metrics = []
        for seed in range(5):
            task_id = game_id * 5 + seed
            task_dirs = list(train_dir.glob(f'whole_population_task_{task_id}_*'))
            
            if len(task_dirs) > 0:
                checkpoint_path = task_dirs[0] / 'checkpoints' / 'best_model.pth'
                
                if checkpoint_path.exists():
                    try:
                        model = load_checkpoint(checkpoint_path, device)
                        repr_complexity = compute_representational_complexity(model, test_states, device)
                        seed_repr_metrics.append(repr_complexity)
                        
                        if seed == 0:
                            cka_models[game_id] = model
                    except Exception as e:
                        print(f"  Warning: Error loading checkpoint for seed {seed}: {e}")
        
        if len(seed_repr_metrics) > 0:
            repr_complexity = {
                key: np.mean([m[key] for m in seed_repr_metrics])
                for key in seed_repr_metrics[0].keys()
            }
        else:
            repr_complexity = {
                'social_ratio': np.nan,
                'effective_dimensionality': np.nan,
                'weight_l2_norm': np.nan
            }
        
        record = {
            'game_id': game_id,
            'train_game': game,
            **task_complexity,
            **opp_complexity,
            **behavior_complexity,
            **repr_complexity
        }
        complexity_records.append(record)
    
    df_complexity = pd.DataFrame(complexity_records)
    
    csv_path = data_dir / 'complexity_metrics.csv'
    df_complexity.to_csv(csv_path, index=False)
    print(f"\nSaved complexity metrics: {csv_path}")
    
    # Compute CKA
    print("\nComputing CKA similarity matrix...")
    n_games = len(cka_models)
    cka_matrix = np.zeros((n_games, n_games))
    
    game_ids = sorted(cka_models.keys())
    for i, gid_i in enumerate(game_ids):
        for j, gid_j in enumerate(game_ids):
            if i <= j:
                if i == j:
                    cka_matrix[i, j] = 1.0
                else:
                    try:
                        # Extract hidden states from both models
                        hidden_i = extract_hidden_state_representations(cka_models[gid_i], test_states, device)
                        hidden_j = extract_hidden_state_representations(cka_models[gid_j], test_states, device)
                        
                        # Compute CKA similarity
                        cka_sim = compute_representational_similarity(
                            hidden_i,
                            hidden_j,
                            method='cka'
                        )
                        cka_matrix[i, j] = cka_sim
                        cka_matrix[j, i] = cka_sim
                    except Exception as e:
                        print(f"  Warning: CKA failed: {e}")
                        cka_matrix[i, j] = np.nan
                        cka_matrix[j, i] = np.nan
    
    np.save(data_dir / 'cka_similarity_matrix.npy', cka_matrix)
    print(f"Saved CKA matrix: {data_dir / 'cka_similarity_matrix.npy'}")
    
    # Run components (skip component 2 - no opponent-specific training)
    print("\n" + "="*80)
    print("RUNNING ANALYSIS COMPONENTS")
    print("="*80)
    
    all_figures = {}
    
    # Component 1
    summary_1, figs_1 = component_1_task_representation(df_complexity, plots_dir)
    all_figures.update(figs_1)
    if len(summary_1) > 0:
        summary_1.to_csv(data_dir / 'component_1_summary.csv', index=True)
    
    # Component 3
    summary_3, figs_3 = component_3_representation_behavior_coupling(df_complexity, cka_matrix, plots_dir)
    all_figures.update(figs_3)
    if len(summary_3) > 0:
        summary_3.to_csv(data_dir / 'component_3_summary.csv', index=False)
    
    # Component 4
    summary_4, figs_4 = component_4_generalization_similarity(df_complexity, df_test, cka_matrix, plots_dir)
    all_figures.update(figs_4)
    if len(summary_4) > 0:
        summary_4.to_csv(data_dir / 'component_4_summary.csv', index=False)
    
    # Component 5
    summary_5, figs_5 = component_5_integrated_complexity(df_complexity, plots_dir)
    all_figures.update(figs_5)
    if len(summary_5) > 0:
        summary_5.to_csv(data_dir / 'component_5_summary.csv', index=False)
    
    # Save figures
    for fig_name, fig in all_figures.items():
        fig_path = plots_dir / f'{fig_name}.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {fig_path}")
    
    print("\n" + "="*80)
    print("TASK SETUP ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run complexity analysis')
    parser.add_argument('--setup', type=str, choices=['task-opponent', 'task', 'all'],
                       default='all', help='Which setup to analyze')
    args = parser.parse_args()
    
    output_base = project_root / 'Results'
    
    if args.setup in ['task-opponent', 'all']:
        analyze_task_opponent_setup(output_base)
    
    if args.setup in ['task', 'all']:
        analyze_task_setup(output_base)
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
