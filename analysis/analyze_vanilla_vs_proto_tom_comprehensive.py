"""
Comprehensive analysis of vanilla RL experiments.

Complete analysis pipeline:
1. Load training and test data
2. Normalize rewards (min-max per task)
3. Training analysis: losses, rewards, policy entropy
4. Test analysis: generalization by opponent and game
5. Policy change analysis (KLD between training and test)
6. Summary heatmap: training-to-test performance map
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import softmax
from scipy.stats import entropy
import torch
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
from cognitive_therapy_ai import GameLSTM, GameFactory, OpponentFactory

# Directories and configuration
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

# Game payoff ranges for min-max normalization
GAME_PAYOFF_RANGES = {
    'prisoners-dilemma': (0.0, 5.0),  # S to T
    'hawk-dove': (-2.0, 6.0),  # (V-C)/2 to V
    'stag-hunt': (0.0, 4.0),  # stag_failure to stag_payoff
    'battle-of-sexes': (0.0, 3.0)  # miscoordination to player_preferred
}

TRAINING_CONDITIONS = {
    0: ("prisoners-dilemma", "low", [0.1, 0.3]),
    1: ("prisoners-dilemma", "mid_low", [0.3, 0.5]),
    2: ("prisoners-dilemma", "mid_high", [0.5, 0.7]),
    3: ("prisoners-dilemma", "high", [0.7, 0.9]),
    4: ("hawk-dove", "low", [0.1, 0.3]),
    5: ("hawk-dove", "mid_low", [0.3, 0.5]),
    6: ("hawk-dove", "mid_high", [0.5, 0.7]),
    7: ("hawk-dove", "high", [0.7, 0.9]),
    8: ("stag-hunt", "low", [0.1, 0.3]),
    9: ("stag-hunt", "mid_low", [0.3, 0.5]),
    10: ("stag-hunt", "mid_high", [0.5, 0.7]),
    11: ("stag-hunt", "high", [0.7, 0.9]),
    12: ("battle-of-sexes", "low", [0.1, 0.3]),
    13: ("battle-of-sexes", "mid_low", [0.3, 0.5]),
    14: ("battle-of-sexes", "mid_high", [0.5, 0.7]),
    15: ("battle-of-sexes", "high", [0.7, 0.9]),
}

def normalize_reward(reward, game_name):
    """Min-max normalize reward to [0, 1] based on game payoff range."""
    min_reward, max_reward = GAME_PAYOFF_RANGES[game_name]
    if max_reward == min_reward:
        return 0.5
    return (reward - min_reward) / (max_reward - min_reward)


def find_task_path(task_id):
    """Find directory for specific task in vanilla experiments."""
    for dir_name in VANILLA_DIRS:
        dir_path = EXPERIMENTS_DIR / dir_name
        if not dir_path.exists():
            continue
        task_dir = dir_path / f"vanilla_matrix_task{task_id}"
        if task_dir.exists():
            return task_dir
    return None


def load_training_data(task_path):
    """Load training data from CSV file."""
    csv_path = task_path / "checkpoints" / "detailed_training_logs" / "detailed_training_log.csv"
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None


def load_test_results(task_path):
    """Load test evaluation results."""
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


def compute_policy_entropy_from_logits(logit_coop, logit_defect):
    """Compute entropy of policy from logits."""
    logits = np.array([logit_coop, logit_defect])
    probs = softmax(logits, axis=0)
    return entropy(probs)


def analyze_training_data(training_df, game_name):
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
        if not np.isnan(p_coop) and not np.isnan(p_defect):
            ent = entropy([p_coop, p_defect])
            entropies.append(ent)
        else:
            entropies.append(np.nan)
    
    epoch_stats['policy_entropy'] = entropies
    
    # Normalize rewards
    epoch_stats['agent_reward_normalized'] = epoch_stats['agent_reward'].apply(
        lambda r: normalize_reward(r, game_name)
    )
    epoch_stats['opponent_reward_normalized'] = epoch_stats['opponent_reward'].apply(
        lambda r: normalize_reward(r, game_name)
    )
    
    # Final epoch statistics
    final_epoch_idx = epoch_stats['epoch'].max()
    final_stats = epoch_stats[epoch_stats['epoch'] >= final_epoch_idx - 10].mean()
    
    return {
        'epoch_stats': epoch_stats,
        'final_stats': final_stats,
        'final_policy_entropy': final_stats['policy_entropy'],
        'final_reward_normalized': final_stats['agent_reward_normalized'],
        'final_cooperation_rate': final_stats['policy_prob_cooperate']
    }


def load_checkpoint_and_get_policy(task_path, game_name, opponent_probs):
    """Load final checkpoint and compute policy distribution for test opponents."""
    checkpoint_path = task_path / "checkpoints" / "final_model.pt"
    if not checkpoint_path.exists():
        return None
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create network with same architecture
        network = GameLSTM(
            input_size=5,
            hidden_size=128,
            num_layers=2,
            num_actions=2
        )
        network.load_state_dict(checkpoint['model_state_dict'])
        network.eval()
        
        # Create game
        game = GameFactory.create_game(game_name)
        
        # Compute average policy over multiple games
        policies = []
        
        for opp_prob in opponent_probs:
            opponent = OpponentFactory.create_opponent('probabilistic', defection_prob=opp_prob)
            
            # Run a few games to get average policy
            game_policies = []
            for _ in range(10):  # 10 games per opponent
                game.reset()
                hidden_state = None
                
                for round_num in range(100):  # 100 rounds per game
                    state = game.get_state_vector()
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
                    
                    with torch.no_grad():
                        policy_logits, _, _, hidden_state = network(state_tensor, hidden_state)
                    
                    probs = torch.softmax(policy_logits.squeeze(), dim=0).numpy()
                    game_policies.append(probs)
                    
                    # Sample action and play
                    action_idx = np.random.choice(2, p=probs)
                    from cognitive_therapy_ai.games import Action
                    action = Action.COOPERATE if action_idx == 0 else Action.DEFECT
                    opp_action = opponent.play_action(game.history, round_num)
                    game.play_round(action, opp_action)
            
            # Average policy for this opponent
            avg_policy = np.mean(game_policies, axis=0)
            policies.append({
                'opponent_prob': opp_prob,
                'policy': avg_policy,
                'entropy': entropy(avg_policy)
            })
        
        return policies
        
    except Exception as e:
        print(f"Error loading checkpoint or computing policy: {e}")
        return None


def compute_kld(p, q):
    """Compute KL divergence KL(p||q), handling zeros."""
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))
            
            rows.append(row)
    
    return pd.DataFrame(rows)

def print_comprehensive_summary(df):
    """Print detailed summary statistics."""
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS: VANILLA RL VS PROTO-TOM")
    print("="*80 + "\n")
    
    # Overall statistics
    print("OVERALL PERFORMANCE (ALL CONDITIONS):")
    print("-" * 80)
    
    if 'vanilla_reward' in df.columns and 'proto_tom_reward' in df.columns:
        vanilla_mean = df['vanilla_reward'].mean()
        proto_tom_mean = df['proto_tom_reward'].mean()
        
        print(f"\nAverage Reward:")
        print(f"  Vanilla RL:  {vanilla_mean:.4f} (±{df['vanilla_reward'].std():.4f})")
        print(f"  Proto-ToM:   {proto_tom_mean:.4f} (±{df['proto_tom_reward'].std():.4f})")
        print(f"  Advantage:   {proto_tom_mean - vanilla_mean:.4f}")
        
        # Statistical test
        if len(df) > 1:
            t_stat, p_value = stats.ttest_rel(df['proto_tom_reward'].dropna(), 
                                               df['vanilla_reward'].dropna())
            print(f"  t-test:      t={t_stat:.3f}, p={p_value:.4f}")
    
    if 'vanilla_cooperation' in df.columns and 'proto_tom_cooperation' in df.columns:
        vanilla_coop = df['vanilla_cooperation'].mean()
        proto_tom_coop = df['proto_tom_cooperation'].mean()
        
        print(f"\nCooperation Rate:")
        print(f"  Vanilla RL:  {vanilla_coop:.4f} (±{df['vanilla_cooperation'].std():.4f})")
        print(f"  Proto-ToM:   {proto_tom_coop:.4f} (±{df['proto_tom_cooperation'].std():.4f})")
        print(f"  Advantage:   {proto_tom_coop - vanilla_coop:.4f}")
    
    # Baseline vs Generalization
    print("\n" + "="*80)
    print("BASELINE VS GENERALIZATION PERFORMANCE:")
    print("-" * 80)
    
    baseline_df = df[df['test_condition'] == 'baseline']
    generalization_df = df[df['test_condition'] != 'baseline']
    
    print("\nBaseline (In-Distribution) Performance:")
    if len(baseline_df) > 0 and 'vanilla_reward' in baseline_df.columns:
        print(f"  Vanilla RL:  {baseline_df['vanilla_reward'].mean():.4f}")
        print(f"  Proto-ToM:   {baseline_df['proto_tom_reward'].mean():.4f}")
        print(f"  Advantage:   {baseline_df['reward_advantage'].mean():.4f}")
    
    print("\nGeneralization (OOD) Performance:")
    if len(generalization_df) > 0 and 'vanilla_reward' in generalization_df.columns:
        print(f"  Vanilla RL:  {generalization_df['vanilla_reward'].mean():.4f}")
        print(f"  Proto-ToM:   {generalization_df['proto_tom_reward'].mean():.4f}")
        print(f"  Advantage:   {generalization_df['reward_advantage'].mean():.4f}")
    
    # By condition type
    print("\n" + "="*80)
    print("PERFORMANCE BY GENERALIZATION TYPE:")
    print("-" * 80)
    
    condition_types = {
        'Baseline': ['baseline'],
        'Same Game': [c for c in df['test_condition'].unique() if c.startswith('same_game')],
        'New Game (Same Opponents)': [c for c in df['test_condition'].unique() 
                                       if 'same_opponents' in c],
        'Cross-Generalization': [c for c in df['test_condition'].unique() 
                                  if c not in ['baseline'] and 'same_game' not in c 
                                  and 'same_opponents' not in c]
    }
    
    for cond_type, conditions in condition_types.items():
        cond_df = df[df['test_condition'].isin(conditions)]
        if len(cond_df) > 0 and 'vanilla_reward' in cond_df.columns:
            print(f"\n{cond_type}:")
            print(f"  Vanilla:     {cond_df['vanilla_reward'].mean():.4f}")
            print(f"  Proto-ToM:   {cond_df['proto_tom_reward'].mean():.4f}")
            print(f"  Advantage:   {cond_df['reward_advantage'].mean():.4f}")
            print(f"  N conditions: {len(conditions)}")
    
    # By game
    print("\n" + "="*80)
    print("PERFORMANCE BY TRAINING GAME:")
    print("-" * 80)
    
    for game in GAMES:
        game_df = df[df['training_game'] == game]
        if len(game_df) > 0 and 'vanilla_reward' in game_df.columns:
            print(f"\n{game.replace('-', ' ').title()}:")
            print(f"  Vanilla:     {game_df['vanilla_reward'].mean():.4f}")
            print(f"  Proto-ToM:   {game_df['proto_tom_reward'].mean():.4f}")
            print(f"  Advantage:   {game_df['reward_advantage'].mean():.4f}")
    
    print("\n" + "="*80)

def create_visualizations(df, output_dir):
    """Create comprehensive visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")
    
    # 1. Overall reward comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    baseline_df = df[df['test_condition'] == 'baseline']
    generalization_df = df[df['test_condition'] != 'baseline']
    
    x = np.arange(2)
    width = 0.35
    
    vanilla_means = [baseline_df['vanilla_reward'].mean(), generalization_df['vanilla_reward'].mean()]
    proto_tom_means = [baseline_df['proto_tom_reward'].mean(), generalization_df['proto_tom_reward'].mean()]
    
    ax.bar(x - width/2, vanilla_means, width, label='Vanilla RL', alpha=0.8)
    ax.bar(x + width/2, proto_tom_means, width, label='Proto-ToM', alpha=0.8)
    
    ax.set_ylabel('Average Reward', fontweight='bold', fontsize=12)
    ax.set_title('Baseline vs Generalization Performance', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline\n(In-Distribution)', 'Generalization\n(Out-of-Distribution)'])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Advantage heatmap by training condition
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Group by training condition and test condition
    pivot_data = []
    for task_id in range(16):
        task_df = df[df['task_id'] == task_id]
        game, opp_range, _ = TRAINING_CONDITIONS[task_id]
        
        for _, row in task_df.iterrows():
            pivot_data.append({
                'Training': f"{game[:2].upper()}-{opp_range}",
                'Test': row['test_condition'],
                'Advantage': row.get('reward_advantage', np.nan)
            })
    
    pivot_df = pd.DataFrame(pivot_data)
    pivot_table = pivot_df.pivot(index='Training', columns='Test', values='Advantage')
    
    sns.heatmap(pivot_table, annot=False, cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Proto-ToM Advantage'},
                linewidths=0.5, ax=ax)
    
    ax.set_title('Proto-ToM Advantage Across All Training Conditions', 
                 fontweight='bold', fontsize=14)
    ax.set_xlabel('Test Condition', fontweight='bold', fontsize=12)
    ax.set_ylabel('Training Condition', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'advantage_heatmap_full.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Visualizations saved to {output_dir}/")

def main():
    print("="*80)
    print("COMPREHENSIVE VANILLA RL VS PROTO-TOM ANALYSIS")
    print("="*80)
    print()
    
    # Load data
    print("Loading data...")
    vanilla_paths = find_task_paths('vanilla')
    proto_tom_paths = find_task_paths('proto-tom')
    
    print(f"  Vanilla RL:  {len(vanilla_paths)} tasks")
    print(f"  Proto-ToM:   {len(proto_tom_paths)} tasks")
    
    vanilla_data = {}
    proto_tom_data = {}
    
    for task_id in range(16):
        if task_id in vanilla_paths:
            results = load_task_results(vanilla_paths[task_id], 'vanilla')
            metrics = extract_all_metrics(results, 'vanilla')
            if metrics:
                vanilla_data[task_id] = metrics
        
        if task_id in proto_tom_paths:
            results = load_task_results(proto_tom_paths[task_id], 'proto-tom')
            metrics = extract_all_metrics(results, 'proto-tom')
            if metrics:
                proto_tom_data[task_id] = metrics
    
    print(f"\nSuccessfully loaded:")
    print(f"  Vanilla RL:  {len(vanilla_data)} tasks")
    print(f"  Proto-ToM:   {len(proto_tom_data)} tasks")
    
    # Create dataframe
    df = create_comprehensive_dataframe(vanilla_data, proto_tom_data)
    
    # Save results
    output_dir = Path("experiments/results/vanilla_vs_proto_tom_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / "comprehensive_comparison.csv", index=False)
    print(f"\n✅ Data saved to {output_dir}/comprehensive_comparison.csv")
    
    # Print statistics
    print_comprehensive_summary(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(df, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    
if __name__ == "__main__":
    main()
