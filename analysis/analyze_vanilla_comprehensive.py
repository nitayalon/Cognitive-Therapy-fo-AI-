"""
Comprehensive Vanilla RL Analysis (Fast Version)

Analysis pipeline:
1. Load training and test data
2. Normalize rewards (min-max per task)
3. Training analysis: losses, rewards, policy entropy
4. Test analysis: generalization by opponent and game
5. Summary plots and heatmaps

Note: KLD computation skipped for speed - will be computed separately.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

# Configuration
EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
OUTPUT_DIR = Path(__file__).parent / "experiments" / "results" / "vanilla_baseline_analysis"

VANILLA_DIRS = [
    "vanilla_rl_array_835338_20260223_065612",  
    "vanilla_rl_array_835338_20260223_152600",  
    "vanilla_rl_array_835338_20260223_152654",  
    "vanilla_rl_array_835338_20260223_152715",  
    "vanilla_rl_array_835338_20260223_153050",  
    "vanilla_rl_array_835338_20260223_153053",  
    "vanilla_rl_array_835338_20260223_153234",  
    "vanilla_rl_array_835338_20260223_153306",  
    "vanilla_rl_array_835338_20260223_153933",  
]

GAME_PAYOFF_RANGES = {
    'prisoners-dilemma': (0.0, 5.0),
    'hawk-dove': (-2.0, 6.0),
    'stag-hunt': (0.0, 4.0),
    'battle-of-sexes': (0.0, 3.0)
}

TRAINING_CONDITIONS = {
    0: ("prisoners-dilemma", "low", [0.1, 0.2, 0.3, 0.4]),
    1: ("prisoners-dilemma", "mid_low", [0.3, 0.4, 0.5, 0.6]),
    2: ("prisoners-dilemma", "mid_high", [0.5, 0.6, 0.7, 0.8]),
    3: ("prisoners-dilemma", "high", [0.7, 0.8, 0.9, 1.0]),
    4: ("hawk-dove", "low", [0.1, 0.2, 0.3, 0.4]),
    5: ("hawk-dove", "mid_low", [0.3, 0.4, 0.5, 0.6]),
    6: ("hawk-dove", "mid_high", [0.5, 0.6, 0.7, 0.8]),
    7: ("hawk-dove", "high", [0.7, 0.8, 0.9, 1.0]),
    8: ("stag-hunt", "low", [0.1, 0.2, 0.3, 0.4]),
    9: ("stag-hunt", "mid_low", [0.3, 0.4, 0.5, 0.6]),
    10: ("stag-hunt", "mid_high", [0.5, 0.6, 0.7, 0.8]),
    11: ("stag-hunt", "high", [0.7, 0.8, 0.9, 1.0]),
    12: ("battle-of-sexes", "low", [0.1, 0.2, 0.3, 0.4]),
    13: ("battle-of-sexes", "mid_low", [0.3, 0.4, 0.5, 0.6]),
    14: ("battle-of-sexes", "mid_high", [0.5, 0.6, 0.7, 0.8]),
    15: ("battle-of-sexes", "high", [0.7, 0.8, 0.9, 1.0]),
}


def normalize_reward(reward, game_name):
    """Min-max normalize reward to [0, 1]."""
    min_r, max_r = GAME_PAYOFF_RANGES[game_name]
    if max_r == min_r:
        return 0.5
    return np.clip((reward - min_r) / (max_r - min_r), 0, 1)


def find_task_path(task_id):
    """Find directory for specific task."""
    for dir_name in VANILLA_DIRS:
        dir_path = EXPERIMENTS_DIR / dir_name
        if not dir_path.exists():
            continue
        task_dir = dir_path / f"vanilla_matrix_task{task_id}"
        if task_dir.exists():
            return task_dir
    return None


def load_training_data(task_path):
    """Load training CSV."""
    csv_path = task_path / "checkpoints" / "detailed_training_logs" / "detailed_training_log.csv"
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None


def load_test_results(task_path):
    """Load test results pickle."""
    results_dir = task_path / "results"
    if not results_dir.exists():
        return None
    
    pkl_files = list(results_dir.glob("*.pkl"))
    if not pkl_files:
        return None
    
    try:
        with open(pkl_files[0], 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return None


def analyze_training(training_df, game_name):
    """Analyze training data: losses, rewards, policy entropy."""
    if training_df is None or len(training_df) == 0:
        return None
    
    # Group by epoch
    epoch_stats = training_df.groupby('epoch').agg({
        'rl_loss': 'mean',
        'agent_reward': 'mean',
        'opponent_reward': 'mean',
        'policy_prob_cooperate': 'mean',
        'policy_prob_defect': 'mean'
    }).reset_index()
    
    # Compute policy entropy per epoch
    entropies = []
    for _, row in epoch_stats.iterrows():
        p_coop = row['policy_prob_cooperate']
        p_defect = row['policy_prob_defect']
        if not np.isnan(p_coop) and not np.isnan(p_defect) and p_coop + p_defect > 0:
            ent = entropy([p_coop, p_defect])
            entropies.append(ent)
        else:
            entropies.append(np.nan)
    
    epoch_stats['policy_entropy'] = entropies
    
    # Normalize rewards
    epoch_stats['agent_reward_norm'] = epoch_stats['agent_reward'].apply(
        lambda r: normalize_reward(r, game_name)
    )
    epoch_stats['opponent_reward_norm'] = epoch_stats['opponent_reward'].apply(
        lambda r: normalize_reward(r, game_name)
    )
    
    # Final epoch stats (average last 10 epochs)
    final_epoch = epoch_stats['epoch'].max()
    final_stats = epoch_stats[epoch_stats['epoch'] >= final_epoch - 10].mean()
    
    return {
        'epoch_stats': epoch_stats,
        'final_entropy': final_stats['policy_entropy'],
        'final_reward_norm': final_stats['agent_reward_norm'],
        'final_coop_rate': final_stats['policy_prob_cooperate'],
        'final_loss': final_stats['rl_loss']
    }




def analyze_task(task_id):
    """Complete analysis for one task (without KLD computation)."""
    print(f"\n{'='*80}")
    print(f"TASK {task_id}")
    print(f"{'='*80}")
    
    game_name, opp_range, opp_probs = TRAINING_CONDITIONS[task_id]
    print(f"Training: {game_name} with {opp_range} opponents {opp_probs}")
    
    task_path = find_task_path(task_id)
    
    if not task_path:
        print(f"❌ Task path not found")
        return None
    
    print(f"Task path: {task_path}")
    
    result = {
        'task_id': task_id,
        'training_game': game_name,
        'training_opp_range': opp_range,
        'training_opp_probs': opp_probs
    }
    
    # 1. Training analysis
    print(f"\n[1/2] Training Analysis")
    training_df = load_training_data(task_path)
    if training_df is not None:
        print(f"  Loaded {len(training_df)} training records")
        train_analysis = analyze_training(training_df, game_name)
        if train_analysis:
            result['training'] = train_analysis
            print(f"  ✓ Final entropy: {train_analysis['final_entropy']:.4f}")
            print(f"  ✓ Final reward (normalized): {train_analysis['final_reward_norm']:.4f}")
            print(f"  ✓ Final cooperation rate: {train_analysis['final_coop_rate']:.4f}")
    else:
        print(f"  ❌ No training data found")
    
    # 2. Test results (summaries)
    print(f"\n[2/2] Test Results Analysis")
    test_results = load_test_results(task_path)
    if test_results and 'evaluation_summaries' in test_results:
        test_summaries = {}
        for cond_name, summary in test_results['evaluation_summaries'].items():
            if isinstance(summary, dict):
                reward = summary.get('mean_reward', np.nan)
                reward_norm = normalize_reward(reward, game_name)
                test_summaries[cond_name] = {
                    'reward': reward,
                    'reward_norm': reward_norm,
                    'coop_rate': summary.get('mean_cooperation_rate', np.nan)
                }
        result['test_summaries'] = test_summaries
        print(f"  ✓ Found {len(test_summaries)} test conditions")
    else:
        print(f"  ❌ No test results found")
    
    print(f"\n✓ Task {task_id} complete\n")
    return result


def create_plots(all_results, output_dir):
    """Create all requested plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    
    #  Plot 1: Training evolution (example from task 0)
    if 0 in all_results and 'training' in all_results[0]:
        epoch_stats = all_results[0]['training']['epoch_stats']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # RL Loss
        axes[0, 0].plot(epoch_stats['epoch'], epoch_stats['rl_loss'])
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('RL Loss')
        axes[0, 0].set_title('Training Loss')
        
        # Normalized Reward
        axes[0, 1].plot(epoch_stats['epoch'], epoch_stats['agent_reward_norm'], label='Agent')
        axes[0, 1].plot(epoch_stats['epoch'], epoch_stats['opponent_reward_norm'], label='Opponent')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Normalized Reward')
        axes[0, 1].set_title('Rewards Over Training')
        axes[0, 1].legend()
        
        # Policy Entropy
        axes[1, 0].plot(epoch_stats['epoch'], epoch_stats['policy_entropy'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].set_title('Policy Entropy')
        
        # Action Distribution
        axes[1, 1].plot(epoch_stats['epoch'], epoch_stats['policy_prob_cooperate'], label='Cooperate')
        axes[1, 1].plot(epoch_stats['epoch'], epoch_stats['policy_prob_defect'], label='Defect')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Probability')
        axes[1, 1].set_title('Action Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_evolution_task0.png', dpi=150)
        plt.close()
        print(f"  ✓ Saved training_evolution_task0.png")
    
    # Plot 2: Generalization to new opponents (same game)
    same_game_data = []
    for task_id, res in all_results.items():
        if 'test_summaries' in res:
            for cond_name in res['test_summaries'].keys():
                if cond_name.startswith('same_game') or cond_name == 'baseline':
                    # Extract opponent range from condition name
                    if cond_name == 'baseline':
                        opp_label = res['training_opp_range']
                    else:
                        opp_label = cond_name.replace('same_game_', '')
                    
                    reward_norm = res['test_summaries'][cond_name]['reward_norm']
                    
                    same_game_data.append({
                        'task_id': task_id,
                        'opponent_range': opp_label,
                        'reward_norm': reward_norm
                    })
    
    if same_game_data:
        df_same_game = pd.DataFrame(same_game_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Reward by opponent
        for opp_range in ['low', 'mid_low', 'mid_high', 'high']:
            subset = df_same_game[df_same_game['opponent_range'] == opp_range]
            ax.scatter([opp_range] * len(subset), subset['reward_norm'], alpha=0.6, s=100)
        ax.set_xlabel('Opponent Range')
        ax.set_ylabel('Normalized Reward')
        ax.set_title('Generalization to New Opponents (Same Game)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'same_game_generalization.png', dpi=150)
        plt.close()
        print(f"  ✓ Saved same_game_generalization.png")
    
    # Plot 4: Large training-to-test heatmap
    heatmap_data = []
    for task_id, res in all_results.items():
        if 'test_summaries' not in res:
            continue
        
        training_game = res['training_game']
        training_opp_range = res['training_opp_range']
        baseline_reward = res['test_summaries'].get('baseline', {}).get('reward_norm', np.nan)
        
        for cond_name, summary in res['test_summaries'].items():
            # Parse test condition
            if cond_name == 'baseline':
                test_game = training_game
                test_opp_range = training_opp_range
            elif cond_name.startswith('same_game'):
                test_game = training_game
                test_opp_range = cond_name.replace('same_game_', '')
            elif '_same_opponents' in cond_name:
                test_game = cond_name.replace('_same_opponents', '')
                test_opp_range = training_opp_range
            elif '_high' in cond_name:
                test_game = cond_name.replace('_high', '')
                test_opp_range = 'high'
            else:
                continue
            
            test_reward = summary['reward_norm']
            relative_reward = test_reward - baseline_reward if cond_name != 'baseline' else 0.0
            
            heatmap_data.append({
                'train_game': training_game,
                'train_opp': training_opp_range,
                'test_game': test_game,
                'test_opp': test_opp_range,
                'relative_reward': relative_reward
            })
    
    if heatmap_data:
        df_heatmap = pd.DataFrame(heatmap_data)
        
        # Create separate plots for each training game
        games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt', 'battle-of-sexes']
        game_labels = ['PD', 'HD', 'SH', 'BS']  # Proper abbreviations
        game_full_names = ['Prisoner\'s Dilemma', 'Hawk-Dove', 'Stag-Hunt', 'Battle of Sexes']
        opp_ranges = ['low', 'mid_low', 'mid_high', 'high']
        opp_labels = ['Low', 'ML', 'MH', 'High']  # Clearer opponent labels
        
        # Group by training condition and average
        grouped = df_heatmap.groupby(['train_game', 'train_opp', 'test_game', 'test_opp'])['relative_reward'].mean().reset_index()
        
        # Discretization function for 0.1 width bins
        def discretize_value(val):
            """Bin values into 0.1 width bins from -0.5 to 0.5"""
            # Bins: [-0.5,-0.4), [-0.4,-0.3), ..., [0.4,0.5]
            bins = np.arange(-0.5, 0.6, 0.1)  # 0.6 to include 0.5
            bin_centers = bins[:-1] + 0.05  # Center of each bin
            idx = np.digitize(val, bins) - 1
            idx = np.clip(idx, 0, len(bin_centers) - 1)
            return bin_centers[idx]
        
        # Create discrete colormap with 10 levels
        from matplotlib.colors import BoundaryNorm
        bounds = np.arange(-0.5, 0.6, 0.1)  # 11 boundaries for 10 bins
        norm = BoundaryNorm(bounds, ncolors=256)
        
        # Create one plot per training game
        for game_idx, train_game in enumerate(games):
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            im_for_colorbar = None  # Save one image for colorbar
            
            for j, train_opp in enumerate(opp_ranges):
                ax = axes[j]
                
                # Get data for this training condition
                train_data = grouped[
                    (grouped['train_game'] == train_game) & 
                    (grouped['train_opp'] == train_opp)
                ]
                
                if len(train_data) > 0:
                    # Create matrix
                    matrix = np.zeros((len(games), len(opp_ranges)))
                    for _, row in train_data.iterrows():
                        test_game_idx = games.index(row['test_game'])
                        test_opp_idx = opp_ranges.index(row['test_opp'])
                        # Discretize the value
                        matrix[test_game_idx, test_opp_idx] = discretize_value(row['relative_reward'])
                    
                    # Plot heatmap with discrete colormap
                    im = ax.imshow(matrix, cmap='RdYlGn', norm=norm, aspect='auto')
                    if im_for_colorbar is None:
                        im_for_colorbar = im
                    ax.set_xticks(range(len(opp_ranges)))
                    ax.set_yticks(range(len(games)))
                    ax.set_xticklabels(opp_labels, fontsize=12)
                    ax.set_yticklabels(game_labels, fontsize=12)
                
                ax.set_title(f"Training: {opp_labels[j]} Opponents", fontsize=14, fontweight='bold')
                ax.set_xlabel('Test Opponent Range', fontsize=12)
                if j == 0:
                    ax.set_ylabel('Test Game', fontsize=12)
            
            # Add colorbar with discrete levels
            if im_for_colorbar is not None:
                cbar = fig.colorbar(im_for_colorbar, ax=axes, orientation='vertical', 
                            label='Relative Reward (vs Baseline)', pad=0.02, fraction=0.046,
                            ticks=bounds)
                cbar.ax.tick_params(labelsize=11)
                cbar.set_label('Relative Reward (vs Baseline)', fontsize=12)
            
            plt.suptitle(f'{game_full_names[game_idx]} Agents: Generalization Performance (Binned)', 
                         fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            # Save with game-specific filename
            filename = f'heatmap_{game_labels[game_idx]}.png'
            plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved {filename}")


def main():
    print("="*80)
    print("COMPREHENSIVE VANILLA RL ANALYSIS - FAST VERSION")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Analyzing 16 tasks (KLD computation skipped):")
    print("  1. Training data (losses, rewards, entropy)")
    print("  2. Test summaries (rewards, cooperation)\n")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Analyze all tasks
    all_results = {}
    start_time = pd.Timestamp.now()
    
    for task_id in range(16):
        task_start = pd.Timestamp.now()
        
        result = analyze_task(task_id)
        if result:
            all_results[task_id] = result
            
            # Save incrementally after each task
            with open(OUTPUT_DIR / 'all_results_partial.pkl', 'wb') as f:
                pickle.dump(all_results, f)
            print(f"💾 Saved progress ({len(all_results)}/16 tasks)")
        
        task_duration = (pd.Timestamp.now() - task_start).total_seconds()
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        avg_time = elapsed / (task_id + 1)
        remaining = avg_time * (16 - task_id - 1)
        
        print(f"\nProgress: {task_id+1}/16 tasks | Task time: {task_duration:.1f}s | Est. remaining: {remaining/60:.1f} min\n")
    
    total_time = (pd.Timestamp.now() - start_time).total_seconds()
    print(f"\n✓ Analyzed {len(all_results)} tasks in {total_time/60:.1f} minutes")
    
    # Save final results
    with open(OUTPUT_DIR / 'all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\n✓ Saved final results to {OUTPUT_DIR / 'all_results.pkl'}")
    
    # Create plots
    print("\nGenerating plots...")
    create_plots(all_results, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}/")
    print(f"  - all_results.pkl (full data)")
    print(f"  - training_evolution_task0.png")
    print(f"  - same_game_generalization.png")
    print(f"  - heatmap_PD.png (Prisoner's Dilemma generalization)")
    print(f"  - heatmap_HD.png (Hawk-Dove generalization)")
    print(f"  - heatmap_SH.png (Stag-Hunt generalization)")
    print(f"  - heatmap_BS.png (Battle of Sexes generalization)")


if __name__ == "__main__":
    main()
