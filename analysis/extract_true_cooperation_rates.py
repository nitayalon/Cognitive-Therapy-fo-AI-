#!/usr/bin/env python3
"""
Extract TRUE cooperation rates from detailed testing logs.

The detailed logs contain actual agent actions, not just rewards.
This lets us measure cooperation rate directly instead of inferring from rewards.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

# Paths
BASE_DIR = Path(__file__).parent
TEST_DIR = Path("experiments/whole_population_test_902267/testing")
OUTPUT_DIR = BASE_DIR / 'output' / 'whole_population_generalization' / 'data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Map numeric actions to labels
ACTION_MAP = {
    0: 'cooperate',
    1: 'defect'
}


def load_detailed_testing_log(task_dir):
    """
    Load detailed testing log from a task directory.
    
    Returns:
        DataFrame with agent actions or None if not found
    """
    log_path = task_dir / 'logs' / 'eval_task_0' / 'detailed_testing_log.csv'
    
    if not log_path.exists():
        # Try alternate path structure
        eval_dirs = list((task_dir / 'logs').glob('eval_task_*'))
        if eval_dirs:
            log_path = eval_dirs[0] / 'detailed_testing_log.csv'
    
    if not log_path.exists():
        return None
    
    try:
        df = pd.read_csv(log_path)
        return df
    except Exception as e:
        print(f"  Error reading {log_path}: {e}")
        return None


def extract_cooperation_rate(df):
    """
    Calculate agent cooperation rate from actual actions.
    
    Args:
        df: DataFrame with 'agent_sampled_action' column
        
    Returns:
        float: Cooperation rate (0.0 to 1.0)
    """
    if df is None or len(df) == 0:
        return np.nan
    
    # Action 0 = cooperate, 1 = defect
    total_actions = len(df)
    cooperate_count = (df['agent_sampled_action'] == 0).sum()
    
    return cooperate_count / total_actions


def parse_task_metadata(task_dir_name):
    """
    Extract metadata from task directory name and config.
    
    Returns:
        dict with train_game, test_game, opponent_prob, agent_id
    """
    # Read experiment config
    config_path = task_dir_name / 'experiment_config.json'
    
    if not config_path.exists():
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract checkpoint path to get trained game
        checkpoint_path = config.get('args', {}).get('checkpoint_path', '')
        if 'prisoners-dilemma' in checkpoint_path:
            train_game = 'prisoners-dilemma'
        elif 'hawk-dove' in checkpoint_path:
            train_game = 'hawk-dove'
        elif 'stag-hunt' in checkpoint_path:
            train_game = 'stag-hunt'
        else:
            train_game = 'unknown'
        
        # Test game from args
        test_game = config.get('args', {}).get('test_game', 'unknown')
        
        # Opponent probability from args
        test_opponents = config.get('args', {}).get('test_opponents', '')
        # test_opponents might be a string like "0.1" or a list
        if isinstance(test_opponents, str):
            opponent_prob = float(test_opponents) if test_opponents else np.nan
        elif isinstance(test_opponents, list):
            opponent_prob = test_opponents[0] if test_opponents else np.nan
        else:
            opponent_prob = float(test_opponents) if test_opponents is not None else np.nan
        
        return {
            'train_game': train_game,
            'test_game': test_game,
            'opponent_prob': opponent_prob,
            'config': config
        }
    
    except Exception as e:
        print(f"  Error parsing config: {e}")
        return None


def collect_all_test_data():
    """
    Collect actual cooperation rates from all test logs.
    
    Returns:
        DataFrame with columns: train_game, test_game, opponent_prob, 
                                true_coop_rate, mean_reward, agent_id
    """
    print("=" * 80)
    print("EXTRACTING TRUE COOPERATION RATES FROM DETAILED LOGS")
    print("=" * 80)
    print()
    
    test_dirs = sorted(TEST_DIR.glob('whole_population_task_*'))
    print(f"Found {len(test_dirs)} test task directories")
    print()
    
    results = []
    
    for task_dir in tqdm(test_dirs, desc="Processing test logs"):
        # Parse metadata
        metadata = parse_task_metadata(task_dir)
        if metadata is None:
            continue
        
        # Load detailed log
        log_df = load_detailed_testing_log(task_dir)
        if log_df is None or len(log_df) == 0:
            continue
        
        # Extract cooperation rate from actual actions
        true_coop_rate = extract_cooperation_rate(log_df)
        
        # Calculate mean reward
        mean_reward = log_df['agent_reward'].mean()
        
        # Get agent ID (network serial ID)
        agent_id = log_df['network_serial_id'].iloc[0] if 'network_serial_id' in log_df.columns else 'unknown'
        
        results.append({
            'agent_id': agent_id,
            'train_game': metadata['train_game'],
            'test_game': metadata['test_game'],
            'opponent_prob': metadata['opponent_prob'],
            'true_coop_rate': true_coop_rate,
            'mean_reward': mean_reward
        })
    
    df = pd.DataFrame(results)
    
    print(f"\nCollected data for {len(df)} test conditions")
    print(f"  Unique train games: {df['train_game'].nunique()}")
    print(f"  Unique test games: {df['test_game'].nunique()}")
    print(f"  Unique agents: {df['agent_id'].nunique()}")
    print()
    
    return df


def aggregate_by_condition(df):
    """
    Aggregate cooperation rates across seeds for each test condition.
    
    Returns:
        DataFrame with mean and std of true cooperation rates
    """
    print("Aggregating across seeds...")
    
    agg_df = df.groupby(['train_game', 'test_game', 'opponent_prob']).agg({
        'true_coop_rate': ['mean', 'std', 'count'],
        'mean_reward': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = [
        'train_game', 'test_game', 'opponent_prob',
        'true_coop_rate_mean', 'true_coop_rate_std', 'count',
        'reward_mean', 'reward_std'
    ]
    
    print(f"  Aggregated into {len(agg_df)} conditions")
    print(f"  Seeds per condition: {agg_df['count'].min()}-{agg_df['count'].max()}")
    print()
    
    return agg_df


def compare_to_inferred_coop_rates(true_df):
    """
    Load the inferred cooperation rates and compare to true rates.
    """
    print("=" * 80)
    print("COMPARING TRUE vs INFERRED COOPERATION RATES")
    print("=" * 80)
    print()
    
    # Load inferred rates (from KLD analysis)
    kld_file = OUTPUT_DIR / 'kld_analysis.csv'
    
    if not kld_file.exists():
        print("⚠ KLD analysis file not found - cannot compare")
        return
    
    inferred_df = pd.read_csv(kld_file)
    
    # Ensure opponent_prob is float in both dataframes
    true_df['opponent_prob'] = true_df['opponent_prob'].astype(float)
    inferred_df['opponent_prob'] = inferred_df['opponent_prob'].astype(float)
    
    # Merge on test condition
    merged = pd.merge(
        true_df,
        inferred_df[['train_game', 'test_game', 'opponent_prob', 'agent_coop_prob']],
        on=['train_game', 'test_game', 'opponent_prob'],
        how='inner'
    )
    
    merged['difference'] = merged['true_coop_rate_mean'] - merged['agent_coop_prob']
    merged['abs_difference'] = merged['difference'].abs()
    
    print(f"Merged {len(merged)} conditions for comparison")
    print()
    print("Summary Statistics:")
    print("-" * 80)
    print(f"  Mean absolute difference: {merged['abs_difference'].mean():.4f}")
    print(f"  Max difference: {merged['abs_difference'].max():.4f}")
    print(f"  Correlation: {merged['true_coop_rate_mean'].corr(merged['agent_coop_prob']):.4f}")
    print()
    
    # Show worst mismatches
    print("Top 10 Largest Discrepancies:")
    print("-" * 80)
    worst = merged.nlargest(10, 'abs_difference')[
        ['train_game', 'test_game', 'opponent_prob', 'true_coop_rate_mean', 'agent_coop_prob', 'difference']
    ]
    print(worst.to_string(index=False))
    print()
    
    # Save comparison
    comparison_file = OUTPUT_DIR / 'coop_rate_comparison.csv'
    merged.to_csv(comparison_file, index=False)
    print(f"✓ Saved comparison to: {comparison_file.name}")
    print()
    
    return merged


def main():
    """Main execution."""
    # Collect true cooperation rates from detailed logs
    df = collect_all_test_data()
    
    # Save per-agent data
    per_agent_file = OUTPUT_DIR / 'true_coop_rates_per_agent.csv'
    df.to_csv(per_agent_file, index=False)
    print(f"✓ Saved per-agent data: {per_agent_file.name}")
    print()
    
    # Aggregate across seeds
    agg_df = aggregate_by_condition(df)
    
    # Save aggregated data
    aggregated_file = OUTPUT_DIR / 'true_coop_rates_aggregated.csv'
    agg_df.to_csv(aggregated_file, index=False)
    print(f"✓ Saved aggregated data: {aggregated_file.name}")
    print()
    
    # Compare to inferred rates
    comparison = compare_to_inferred_coop_rates(agg_df)
    
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print()
    print("Key files created:")
    print(f"  - {per_agent_file.name}: True cooperation rates (225 agents)")
    print(f"  - {aggregated_file.name}: Averaged across seeds (45 conditions)")
    print(f"  - coop_rate_comparison.csv: True vs Inferred comparison")
    print()
    print(f"All files saved to: {OUTPUT_DIR}")
    print()


if __name__ == '__main__':
    main()
