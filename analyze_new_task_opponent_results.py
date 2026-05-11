"""
Analyze the new task-opponent results with modified architecture (6-element input).
Job ID: 910969 (generalization_matrix_train)

This script:
1. Verifies all training data is present
2. Analyzes training performance
3. Generates plots comparing to baseline
4. Saves results to experiments/analysis_scripts/output/task_opponent_modified/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Experiment parameters
TRAIN_JOB_ID = "910969"
EXPERIMENT_DIR = Path(f"experiments/generalization_matrix_train_{TRAIN_JOB_ID}")
OUTPUT_DIR = Path("experiments/analysis_scripts/output/task_opponent_modified")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Expected structure
NUM_CONDITIONS = 15
NUM_SEEDS = 5
TOTAL_TASKS = NUM_CONDITIONS * NUM_SEEDS  # 75


def verify_data_completeness():
    """Verify all expected training data is present."""
    print("="*80)
    print("DATA VERIFICATION")
    print("="*80)
    
    training_dir = EXPERIMENT_DIR / "training"
    
    if not training_dir.exists():
        raise ValueError(f"Training directory not found: {training_dir}")
    
    # Check all conditions and seeds
    missing = []
    present = []
    
    for cond_id in range(NUM_CONDITIONS):
        for seed_id in range(NUM_SEEDS):
            cond_dir = training_dir / f"condition_{cond_id}_seed_{seed_id}"
            if not cond_dir.exists():
                missing.append((cond_id, seed_id))
            else:
                # Check for experiment directory
                exp_dirs = list(cond_dir.glob("generalization_matrix_*"))
                if not exp_dirs:
                    missing.append((cond_id, seed_id))
                else:
                    # Check for checkpoint
                    checkpoint = exp_dirs[0] / "checkpoints" / "*_final_checkpoint.pth"
                    if not list(exp_dirs[0].glob("checkpoints/*_final_checkpoint.pth")):
                        missing.append((cond_id, seed_id))
                    else:
                        present.append((cond_id, seed_id))
    
    print(f"Expected tasks: {TOTAL_TASKS}")
    print(f"Present: {len(present)}")
    print(f"Missing: {len(missing)}")
    
    if missing:
        print("\nMissing conditions:")
        for cond, seed in missing:
            print(f"  Condition {cond}, Seed {seed}")
        raise ValueError(f"Missing {len(missing)} training runs!")
    
    print("\n✓ All training data present!")
    return True


def load_all_training_data():
    """Load training logs from all conditions."""
    print("\n" + "="*80)
    print("LOADING TRAINING DATA")
    print("="*80)
    
    all_data = []
    training_dir = EXPERIMENT_DIR / "training"
    
    for cond_id in range(NUM_CONDITIONS):
        print(f"  Loading condition {cond_id}/{NUM_CONDITIONS-1}...", end='\r')
        for seed_id in range(NUM_SEEDS):
            cond_dir = training_dir / f"condition_{cond_id}_seed_{seed_id}"
            exp_dirs = list(cond_dir.glob("generalization_matrix_*"))
            
            if exp_dirs:
                log_file = exp_dirs[0] / "checkpoints" / "detailed_training_logs" / "detailed_training_log.csv"
                
                if log_file.exists():
                    # Load only essential columns for efficiency
                    try:
                        df = pd.read_csv(log_file, usecols=[
                            'epoch', 'iteration', 'total_loss', 'rl_loss', 
                            'opponent_policy_loss', 'agent_reward', 
                            'agent_cooperation_rate', 'policy_prob_cooperate'
                        ] + (['game_name'] if cond_id == 0 and seed_id == 0 else []))
                    except:
                        # Fallback if columns don't exist
                        df = pd.read_csv(log_file, usecols=['epoch', 'iteration', 'total_loss', 'rl_loss'])
                    
                    df['condition_id'] = cond_id
                    df['seed_id'] = seed_id
                    df['task_id'] = cond_id * NUM_SEEDS + seed_id
                    
                    # Load config to get game and opponent info
                    config_file = exp_dirs[0] / "experiment_config.json"
                    if config_file.exists():
                        with open(config_file) as f:
                            config = json.load(f)
                            df['train_game'] = config.get('train_game', 'unknown')
                            df['train_opponents'] = str(config.get('train_opponents', []))
                    
                    all_data.append(df)
    
    print("  " + " "*50, end='\r')  # Clear progress line
                    
    if not all_data:
        raise ValueError("No training logs found!")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"✓ Loaded {len(all_data)} training logs")
    print(f"  Total training iterations: {len(combined_df):,}")
    print(f"  Conditions covered: {combined_df['condition_id'].nunique()}")
    print(f"  Seeds per condition: {combined_df.groupby('condition_id')['seed_id'].nunique().mean():.0f}")
    
    return combined_df


def plot_training_convergence(df):
    """Plot training convergence across all conditions."""
    print("\n" + "="*80)
    print("GENERATING TRAINING CONVERGENCE PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Total Loss
    ax = axes[0, 0]
    for cond_id in range(NUM_CONDITIONS):
        cond_data = df[df['condition_id'] == cond_id]
        cond_mean = cond_data.groupby('epoch')['total_loss'].mean()
        ax.plot(cond_mean.index, cond_mean.values, alpha=0.3, linewidth=1)
    
    # Overall mean
    overall_mean = df.groupby('epoch')['total_loss'].mean()
    ax.plot(overall_mean.index, overall_mean.values, 'k-', linewidth=3, label='Mean across conditions')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Training: Total Loss Convergence (Modified 6-elem Input)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RL Loss
    ax = axes[0, 1]
    for cond_id in range(NUM_CONDITIONS):
        cond_data = df[df['condition_id'] == cond_id]
        cond_mean = cond_data.groupby('epoch')['rl_loss'].mean()
        ax.plot(cond_mean.index, cond_mean.values, alpha=0.3, linewidth=1)
    
    overall_mean = df.groupby('epoch')['rl_loss'].mean()
    ax.plot(overall_mean.index, overall_mean.values, 'k-', linewidth=3, label='Mean across conditions')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('RL Loss', fontsize=12)
    ax.set_title('Training: RL Loss Convergence', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Opponent Loss
    ax = axes[1, 0]
    if 'opponent_loss' in df.columns:
        for cond_id in range(NUM_CONDITIONS):
            cond_data = df[df['condition_id'] == cond_id]
            cond_mean = cond_data.groupby('epoch')['opponent_loss'].mean()
            ax.plot(cond_mean.index, cond_mean.values, alpha=0.3, linewidth=1)
        
        overall_mean = df.groupby('epoch')['opponent_loss'].mean()
        ax.plot(overall_mean.index, overall_mean.values, 'k-', linewidth=3, label='Mean across conditions')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Opponent Prediction Loss', fontsize=12)
        ax.set_title('Training: Opponent Prediction Loss', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Reward
    ax = axes[1, 1]
    if 'agent_reward' in df.columns:
        for cond_id in range(NUM_CONDITIONS):
            cond_data = df[df['condition_id'] == cond_id]
            cond_mean = cond_data.groupby('epoch')['agent_reward'].mean()
            ax.plot(cond_mean.index, cond_mean.values, alpha=0.3, linewidth=1)
        
        overall_mean = df.groupby('epoch')['agent_reward'].mean()
        ax.plot(overall_mean.index, overall_mean.values, 'k-', linewidth=3, label='Mean across conditions')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.set_title('Training: Average Reward', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'training_convergence_modified_6elem.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_loss_by_game(df):
    """Plot loss convergence by game type."""
    print("\n" + "="*80)
    print("GENERATING GAME-SPECIFIC PLOTS")
    print("="*80)
    
    if 'train_game' not in df.columns:
        print("  ⚠ Game information not available")
        return
    
    games = df['train_game'].unique()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, game in enumerate(['prisoners-dilemma', 'hawk-dove', 'stag-hunt']):
        ax = axes[idx]
        game_data = df[df['train_game'] == game]
        
        if len(game_data) > 0:
            # Plot each seed
            for seed_id in range(NUM_SEEDS):
                seed_data = game_data[game_data['seed_id'] == seed_id]
                if len(seed_data) > 0:
                    seed_mean = seed_data.groupby('epoch')['total_loss'].mean()
                    ax.plot(seed_mean.index, seed_mean.values, alpha=0.3, linewidth=1)
            
            # Plot mean
            game_mean = game_data.groupby('epoch')['total_loss'].mean()
            ax.plot(game_mean.index, game_mean.values, 'k-', linewidth=3)
            
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Total Loss', fontsize=11)
        ax.set_title(f'{game.replace("-", " ").title()}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'training_convergence_by_game_modified_6elem.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def compute_final_statistics(df):
    """Compute final training statistics."""
    print("\n" + "="*80)
    print("COMPUTING FINAL STATISTICS")
    print("="*80)
    
    # Get final epoch data
    final_epoch = df['epoch'].max()
    final_data = df[df['epoch'] >= final_epoch - 10]  # Last 10 epochs
    
    # Prepare aggregation dictionary based on available columns
    agg_dict = {
        'total_loss': ['mean', 'std'],
        'rl_loss': ['mean', 'std']
    }
    
    if 'opponent_policy_loss' in df.columns:
        agg_dict['opponent_policy_loss'] = ['mean', 'std']
    if 'agent_reward' in df.columns:
        agg_dict['agent_reward'] = ['mean', 'std']
    if 'agent_cooperation_rate' in df.columns:
        agg_dict['agent_cooperation_rate'] = ['mean', 'std']
    
    stats = final_data.groupby('condition_id').agg(agg_dict).reset_index()
    
    # Save statistics
    stats_file = OUTPUT_DIR / 'final_training_statistics_modified_6elem.csv'
    stats.to_csv(stats_file, index=False)
    print(f"✓ Saved: {stats_file}")
    
    # Print summary
    final_10 = df[df['epoch'] >= final_epoch - 10]
    print("\nFinal Training Performance (last 10 epochs, averaged):")
    print(f"  Total Loss:     {final_10['total_loss'].mean():.4f} ± {final_10['total_loss'].std():.4f}")
    print(f"  RL Loss:        {final_10['rl_loss'].mean():.4f} ± {final_10['rl_loss'].std():.4f}")
    if 'opponent_policy_loss' in df.columns:
        print(f"  Opponent Loss:  {final_10['opponent_policy_loss'].mean():.4f} ± {final_10['opponent_policy_loss'].std():.4f}")
    if 'agent_reward' in df.columns:
        print(f"  Average Reward: {final_10['agent_reward'].mean():.4f} ± {final_10['agent_reward'].std():.4f}")
    
    return stats


def generate_summary_report():
    """Generate summary report."""
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    report_file = OUTPUT_DIR / 'analysis_summary_modified_6elem.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TASK-OPPONENT MODIFIED ARCHITECTURE ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Job ID: {TRAIN_JOB_ID}\n")
        f.write(f"Architecture: 6-element input (with opponent's previous action)\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Training Conditions: {NUM_CONDITIONS}\n")
        f.write(f"Seeds per Condition: {NUM_SEEDS}\n")
        f.write(f"Total Training Runs: {TOTAL_TASKS}\n\n")
        f.write("="*80 + "\n")
        f.write("INPUT MODIFICATION\n")
        f.write("="*80 + "\n\n")
        f.write("Previous architecture: 5 elements\n")
        f.write("  [payoff_matrix(4), round_number(1)]\n\n")
        f.write("Modified architecture: 6 elements\n")
        f.write("  [payoff_matrix(4), round_number(1), opponent_prev_action(1)]\n\n")
        f.write("Opponent previous action encoding:\n")
        f.write("  -1.0 : First trial (no previous action)\n")
        f.write("   0.0 : Opponent cooperated\n")
        f.write("   1.0 : Opponent defected\n\n")
        f.write("="*80 + "\n")
        f.write("GENERATED FILES\n")
        f.write("="*80 + "\n\n")
        f.write("- training_convergence_modified_6elem.png\n")
        f.write("- training_convergence_by_game_modified_6elem.png\n")
        f.write("- final_training_statistics_modified_6elem.csv\n")
        f.write("- training_data_all_modified_6elem.csv\n")
        f.write("- analysis_summary_modified_6elem.txt (this file)\n\n")
        f.write("="*80 + "\n")
        f.write("NEXT STEPS\n")
        f.write("="*80 + "\n\n")
        f.write("1. Run testing phase:\n")
        f.write(f"   TRAINING_JOB_ID={TRAIN_JOB_ID} sbatch run_generalization_matrix_test.sh\n")
        f.write(f"   TRAINING_JOB_ID={TRAIN_JOB_ID} sbatch run_generalization_matrix_test_part2.sh\n\n")
        f.write("2. After testing completes, compare to baseline:\n")
        f.write("   python experiments/analysis_scripts/compare_input_modifications.py \\\n")
        f.write("     --baseline-train <old_job_id> \\\n")
        f.write("     --baseline-test <old_test_id> \\\n")
        f.write(f"     --modified-train {TRAIN_JOB_ID} \\\n")
        f.write("     --modified-test <new_test_id> \\\n")
        f.write("     --experiment-type generalization-matrix\n")
    
    print(f"✓ Saved: {report_file}")


def main():
    """Run complete analysis."""
    print("\n" + "="*80)
    print("TASK-OPPONENT MODIFIED ARCHITECTURE ANALYSIS")
    print("="*80)
    print(f"Job ID: {TRAIN_JOB_ID}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("="*80 + "\n")
    
    try:
        # Verify data
        verify_data_completeness()
        
        # Load data
        df = load_all_training_data()
        
        # Save complete data
        data_file = OUTPUT_DIR / 'training_data_all_modified_6elem.csv'
        df.to_csv(data_file, index=False)
        print(f"\n✓ Saved complete data: {data_file}")
        
        # Generate plots
        plot_training_convergence(df)
        plot_loss_by_game(df)
        
        # Compute statistics
        stats = compute_final_statistics(df)
        
        # Generate report
        generate_summary_report()
        
        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nAll results saved to: {OUTPUT_DIR}")
        print(f"\nNext step: Run testing phase with TRAINING_JOB_ID={TRAIN_JOB_ID}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
