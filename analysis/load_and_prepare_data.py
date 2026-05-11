"""
Load and prepare all whole population data.
Saves processed datasets as CSV files for downstream analysis.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

class DataLoader:
    # Payoff matrices for reward normalization (standard parametrization)
    payoff_matrices = {
        'prisoners-dilemma': {'R': 3, 'S': 0, 'T': 5, 'P': 1},
        'stag-hunt': {'R': 4, 'S': 0, 'T': 3, 'P': 2},
        'hawk-dove': {'R': 2, 'S': 1, 'T': 3, 'P': 0}
    }

    @classmethod
    def normalize_reward(cls, reward, game):
        """Normalize reward by game-specific payoff matrix."""
        if game not in cls.payoff_matrices:
            return 0.5
        payoffs = cls.payoff_matrices[game]
        all_payoffs = [payoffs['R'], payoffs['S'], payoffs['T'], payoffs['P']]
        min_r = min(all_payoffs)
        max_r = max(all_payoffs)
        rng = max_r - min_r
        if rng == 0:
            return 0.5
        norm = (reward - min_r) / rng
        return np.clip(norm, 0, 1)

    def __init__(self, train_root, test_root, output_dir):
        self.train_root = Path(train_root)
        self.test_root = Path(test_root)
        self.output_dir = Path(output_dir)
        self.data_dir = self.output_dir / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_training_data(self):
        """Load all training logs and save aggregated version for plotting."""
        print("Loading training data from all tasks...")
        
        # Instead of loading all raw data, aggregate by task/epoch for efficient plotting
        aggregated_records = []
        
        for task_dir in sorted(self.train_root.glob('training/whole_population_task_*')):
            log = task_dir / 'checkpoints/detailed_training_logs/detailed_training_log.csv'
            if not log.exists():
                continue
            
            # Read in chunks to avoid memory issues
            chunk_size = 100000
            for chunk in pd.read_csv(log, chunksize=chunk_size):
                # Group by epoch and compute statistics
                epoch_stats = chunk.groupby('epoch').agg({
                    'total_loss': ['mean', 'std', 'count'],
                    'policy_prob_cooperate': ['mean', 'std'],
                }).reset_index()
                
                # Flatten column names
                epoch_stats.columns = ['epoch', 'total_loss_mean', 'total_loss_std', 'count',
                                      'policy_prob_cooperate_mean', 'policy_prob_cooperate_std']
                
                # Add metadata
                if 'game_name' in chunk.columns:
                    epoch_stats['game_name'] = chunk['game_name'].iloc[0]
                if 'true_opponent_defect_prob' in chunk.columns:
                    epoch_stats['true_opponent_defect_prob'] = chunk['true_opponent_defect_prob'].iloc[0]
                
                epoch_stats['task_id'] = task_dir.name
                
                aggregated_records.append(epoch_stats)
        
        if aggregated_records:
            df_train = pd.concat(aggregated_records, ignore_index=True)
            
            # Further aggregate by (task_id, epoch) to combine all opponents
            df_train_final = df_train.groupby(['task_id', 'game_name', 'epoch']).agg({
                'total_loss_mean': 'mean',
                'total_loss_std': 'mean',
                'policy_prob_cooperate_mean': 'mean',
                'policy_prob_cooperate_std': 'mean',
                'count': 'sum'
            }).reset_index()
            
            # Save aggregated data
            output_csv = self.data_dir / 'training_data_aggregated.csv'
            df_train_final.to_csv(output_csv, index=False)
            print(f"  Saved aggregated training data to {output_csv}")
            print(f"  Total aggregated records: {len(df_train_final)}")
            return df_train_final
        else:
            print("  No training data found!")
            return pd.DataFrame()

    def load_test_data_per_agent(self):
        """
        Load all test results and aggregate by agent.
        Returns DataFrame with: agent_id, train_game, test_game, opponent_prob, mean_reward, normalized_reward
        """
        print("\nLoading test data from all tasks...")
        records = []
        test_dirs = sorted(self.test_root.glob('testing/whole_population_task_*'))
        
        for task_dir in test_dirs:
            # Read experiment config
            config_path = task_dir / 'experiment_config.json'
            if not config_path.exists():
                continue
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            test_game = config.get('test_game', None)
            
            # checkpoint_path is nested in args
            checkpoint_path = None
            if 'args' in config and 'checkpoint_path' in config['args']:
                checkpoint_path = config['args']['checkpoint_path']
            elif 'checkpoint_path' in config:
                checkpoint_path = config['checkpoint_path']
            
            if test_game is None or checkpoint_path is None:
                continue
            
            # Extract training game from checkpoint_path
            # Format: experiments/whole_population_train_902266/training/whole_population_task_X/checkpoints/<GAME>_final_checkpoint.pth
            train_game = None
            if 'prisoners-dilemma' in checkpoint_path:
                train_game = 'prisoners-dilemma'
            elif 'stag-hunt' in checkpoint_path:
                train_game = 'stag-hunt'
            elif 'hawk-dove' in checkpoint_path:
                train_game = 'hawk-dove'
            
            if train_game is None:
                print(f"  Warning: Could not extract train_game from checkpoint: {checkpoint_path}")
                continue
            
            # Find the detailed_testing_log.csv
            game_id_map = {
                'prisoners-dilemma': 0,
                'stag-hunt': 100,
                'hawk-dove': 200
            }
            expected_eval_id = game_id_map.get(test_game, 0)
            eval_dir = task_dir / 'logs' / f'eval_task_{expected_eval_id}'
            
            if not eval_dir.exists():
                # Try to find any eval_task_* directory
                eval_dirs = list((task_dir / 'logs').glob('eval_task_*'))
                if not eval_dirs:
                    continue
                eval_dir = eval_dirs[0]
            
            log_path = eval_dir / 'detailed_testing_log.csv'
            if not log_path.exists():
                continue
            
            # Read test results
            try:
                df = pd.read_csv(log_path)
            except Exception as e:
                print(f"  Failed to read {log_path}: {e}")
                continue
            
            # Extract information
            if 'network_serial_id' not in df.columns:
                continue
            agent_id = df['network_serial_id'].iloc[0]
            
            if 'true_opponent_defect_prob' not in df.columns:
                continue
            opponent_prob = round(df['true_opponent_defect_prob'].iloc[0], 1)
            
            if 'agent_reward' not in df.columns:
                continue
            mean_reward = df['agent_reward'].mean()
            
            # Normalize reward by test game
            normalized_reward = self.normalize_reward(mean_reward, test_game)
            
            records.append({
                'agent_id': agent_id,
                'train_game': train_game,
                'test_game': test_game,
                'opponent_prob': opponent_prob,
                'mean_reward': mean_reward,
                'normalized_reward': normalized_reward
            })
        
        df_test = pd.DataFrame.from_records(records)
        
        # Save to CSV
        output_csv = self.data_dir / 'test_data_per_agent.csv'
        df_test.to_csv(output_csv, index=False)
        print(f"  Saved per-agent test data to {output_csv}")
        print(f"  Total test records: {len(df_test)}")
        print(f"  Unique agents: {df_test['agent_id'].nunique()}")
        
        return df_test
    
    def aggregate_by_game_agent(self, df_test):
        """
        Average performance across 5 seed replicates for each game-agent.
        Returns DataFrame with: train_game, test_game, opponent_prob, 
        mean_normalized_reward, std_normalized_reward, n_seeds
        """
        print("\nAggregating test results by game-agent (averaging across seeds)...")
        
        # Group by (train_game, test_game, opponent_prob)
        grouped = df_test.groupby(['train_game', 'test_game', 'opponent_prob']).agg({
            'normalized_reward': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['train_game', 'test_game', 'opponent_prob', 
                          'mean_normalized_reward', 'std_normalized_reward', 'n_seeds']
        
        # Save to CSV
        output_csv = self.data_dir / 'game_agent_averaged.csv'
        grouped.to_csv(output_csv, index=False)
        print(f"  Saved game-agent averaged data to {output_csv}")
        
        # Print summary
        print(f"\n  Summary statistics:")
        for train_game in sorted(grouped['train_game'].unique()):
            subset = grouped[grouped['train_game'] == train_game]
            print(f"    {train_game}: {len(subset)} test conditions")
            print(f"      Seeds per condition: {subset['n_seeds'].min()}-{subset['n_seeds'].max()}")
            if subset['n_seeds'].min() != 5:
                print(f"      WARNING: Expected 5 seeds, found {subset['n_seeds'].min()}-{subset['n_seeds'].max()}")
        
        return grouped

    def run(self):
        """Load and prepare all datasets."""
        print("="*60)
        print("WHOLE POPULATION DATA LOADER")
        print("="*60)
        
        # Load training data
        df_train = self.load_training_data()
        
        # Load test data
        df_test_per_agent = self.load_test_data_per_agent()
        
        # Aggregate by game-agent
        if not df_test_per_agent.empty:
            df_game_agent_avg = self.aggregate_by_game_agent(df_test_per_agent)
        else:
            df_game_agent_avg = pd.DataFrame()
        
        print("\n" + "="*60)
        print("DATA PREPARATION COMPLETE")
        print("="*60)
        print(f"Output directory: {self.data_dir}")
        print("\nGenerated files:")
        print(f"  - training_data_aggregated.csv ({len(df_train)} records)")
        print(f"  - test_data_per_agent.csv ({len(df_test_per_agent)} records)")
        print(f"  - game_agent_averaged.csv ({len(df_game_agent_avg)} records)")
        print("\nUse these files for downstream plotting scripts.")
        
        return df_train, df_test_per_agent, df_game_agent_avg


if __name__ == '__main__':
    loader = DataLoader(
        train_root='experiments/whole_population_train_902266',
        test_root='experiments/whole_population_test_902267',
        output_dir='experiments/analysis_scripts/output/whole_population_generalization'
    )
    loader.run()
