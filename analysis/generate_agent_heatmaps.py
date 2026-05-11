"""
Generate Per-Agent Heatmaps for Whole Population Test Results
Aggregates test results and generates one heatmap per agent showing globally normalized rewards.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class AgentHeatmapGenerator:
    # Payoff matrices for reward normalization (standard parametrization)
    payoff_matrices = {
        'prisoners-dilemma': {'R': 3, 'S': 0, 'T': 5, 'P': 1},
        'stag-hunt': {'R': 4, 'S': 0, 'T': 3, 'P': 2},
        'hawk-dove': {'R': 2, 'S': 1, 'T': 3, 'P': 0}
    }

    @classmethod
    def get_reward_bounds(cls, game):
        if game not in cls.payoff_matrices:
            return 0, 1, 1  # fallback
        payoffs = cls.payoff_matrices[game]
        all_payoffs = [payoffs['R'], payoffs['S'], payoffs['T'], payoffs['P']]
        min_r = min(all_payoffs)
        max_r = max(all_payoffs)
        rng = max_r - min_r
        return min_r, max_r, rng

    @classmethod
    def normalize_reward(cls, reward, game):
        min_r, max_r, rng = cls.get_reward_bounds(game)
        if rng == 0:
            return 0.5
        norm = (reward - min_r) / rng
        return np.clip(norm, 0, 1)

    def __init__(self, test_root, output_dir):
        self.test_root = Path(test_root)
        self.output_dir = Path(output_dir)
        self.fig_dir = self.output_dir / 'figures'
        self.data_dir = self.output_dir / 'data'
        for d in [self.fig_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def aggregate_test_results_per_agent(self):
        """
        Aggregate all test results for each agent across all 15 setups (3 games × 5 opponents).
        Returns a DataFrame with columns: agent_id, test_game, opponent_prob, mean_reward, normalized_reward
        """
        print("Aggregating test results per agent across all setups...")
        
        records = []
        test_dirs = sorted(self.test_root.glob('testing/whole_population_task_*'))
        total_dirs = len(list(test_dirs))
        print(f"  Found {total_dirs} test directories")
        
        processed = 0
        for task_dir in test_dirs:
            # Read experiment config to get train/test game info
            config_path = task_dir / 'experiment_config.json'
            if not config_path.exists():
                continue
            
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except Exception as e:
                print(f"  Failed to read config {config_path}: {e}")
                continue
            
            test_game = config.get('test_game', None)
            if test_game is None:
                continue
            
            # Find the detailed_testing_log.csv in the corresponding eval_task_* subfolder
            # The eval task ID should match the test game encoding
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
                df = pd.read_csv(log_path, encoding='utf-8')
            except Exception as e:
                print(f"  Failed to read {log_path}: {e}")
                continue
            
            # Extract agent ID from network_serial_id
            if 'network_serial_id' not in df.columns:
                continue
            agent_id = df['network_serial_id'].iloc[0]
            
            # Extract opponent probability
            if 'true_opponent_defect_prob' not in df.columns:
                continue
            opponent_prob = round(df['true_opponent_defect_prob'].iloc[0], 1)
            
            # Compute mean reward
            if 'agent_reward' not in df.columns:
                continue
            mean_reward = df['agent_reward'].mean()
            
            # Normalize reward by test game
            normalized_reward = self.normalize_reward(mean_reward, test_game)
            
            records.append({
                'agent_id': agent_id,
                'test_game': test_game,
                'opponent_prob': opponent_prob,
                'mean_reward': mean_reward,
                'normalized_reward': normalized_reward
            })
            
            processed += 1
            if processed % 50 == 0:
                print(f"  Processed {processed}/{total_dirs} directories...")
        
        df_results = pd.DataFrame.from_records(records)
        
        # Save to CSV for inspection
        csv_path = self.data_dir / 'per_agent_test_results.csv'
        df_results.to_csv(csv_path, index=False)
        print(f"  Saved aggregated results to {csv_path}")
        print(f"  Total records: {len(df_results)}")
        
        return df_results
    
    def validate_agent_coverage(self, df_results):
        """
        Validate that each agent has results for all 15 setups (3 games × 5 opponents).
        """
        print("\nValidating agent coverage across all 15 setups...")
        
        expected_games = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
        expected_opponents = [0.1, 0.3, 0.5, 0.7, 0.9]
        expected_setups = len(expected_games) * len(expected_opponents)
        
        agents = df_results['agent_id'].unique()
        print(f"  Found {len(agents)} unique agents")
        
        validation_results = []
        for agent in agents:
            agent_data = df_results[df_results['agent_id'] == agent]
            
            # Check coverage
            games_present = agent_data['test_game'].unique()
            opponents_present = agent_data['opponent_prob'].unique()
            
            missing_setups = []
            for game in expected_games:
                for opp in expected_opponents:
                    match = agent_data[(agent_data['test_game'] == game) & 
                                      (agent_data['opponent_prob'] == opp)]
                    if match.empty:
                        missing_setups.append(f"{game}_{opp}")
            
            validation_results.append({
                'agent_id': agent,
                'total_setups': len(agent_data),
                'expected_setups': expected_setups,
                'complete': len(agent_data) == expected_setups,
                'missing_setups': ', '.join(missing_setups) if missing_setups else 'None'
            })
        
        df_validation = pd.DataFrame.from_records(validation_results)
        
        # Save validation results
        val_csv = self.data_dir / 'agent_coverage_validation.csv'
        df_validation.to_csv(val_csv, index=False)
        print(f"  Saved validation results to {val_csv}")
        
        # Print summary
        complete_agents = df_validation['complete'].sum()
        print(f"  {complete_agents}/{len(agents)} agents have complete coverage")
        
        if complete_agents < len(agents):
            print("\n  Agents with incomplete coverage:")
            for _, row in df_validation[~df_validation['complete']].iterrows():
                print(f"    {row['agent_id']}: {row['total_setups']}/{row['expected_setups']} setups")
                if row['missing_setups'] != 'None':
                    print(f"      Missing: {row['missing_setups']}")
        
        return df_validation
    
    def plot_per_agent_heatmaps(self, df_results):
        """
        Generate one heatmap per agent showing globally normalized reward across all 15 setups.
        Each heatmap is 3 games (rows) × 5 opponents (columns).
        """
        print("\nGenerating per-agent heatmaps with globally normalized rewards...")
        
        # Compute global min/max for normalization across ALL agents and setups
        global_min = df_results['normalized_reward'].min()
        global_max = df_results['normalized_reward'].max()
        print(f"  Global normalized reward range: [{global_min:.3f}, {global_max:.3f}]")
        
        # Globally normalize
        if global_max > global_min:
            df_results['global_normalized_reward'] = (
                (df_results['normalized_reward'] - global_min) / (global_max - global_min)
            )
        else:
            df_results['global_normalized_reward'] = 0.5
        
        agents = sorted(df_results['agent_id'].unique())
        print(f"  Generating {len(agents)} heatmaps...")
        
        for i, agent in enumerate(agents, 1):
            agent_data = df_results[df_results['agent_id'] == agent]
            
            # Create pivot table: games (rows) × opponents (columns)
            pivot = agent_data.pivot_table(
                index='test_game',
                columns='opponent_prob',
                values='global_normalized_reward',
                aggfunc='mean'
            )
            
            # Ensure consistent ordering
            game_order = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
            opponent_order = [0.1, 0.3, 0.5, 0.7, 0.9]
            
            # Reindex to ensure all rows/cols are present (fill missing with NaN)
            pivot = pivot.reindex(index=game_order, columns=opponent_order)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                pivot,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Globally Normalized Reward'},
                ax=ax,
                linewidths=0.5,
                linecolor='gray'
            )
            
            # Format labels
            ax.set_ylabel('Test Game', fontsize=12)
            ax.set_xlabel('Opponent Defection Probability', fontsize=12)
            ax.set_yticklabels([g.replace('-', ' ').title() for g in game_order], rotation=0)
            ax.set_xticklabels(opponent_order, rotation=0)
            
            # Title with agent ID (truncated for readability)
            agent_short = agent[:30] if len(agent) > 30 else agent
            ax.set_title(f'Agent: {agent_short}\nGlobally Normalized Test Reward', 
                        fontsize=14, fontweight='bold')
            
            fig.tight_layout()
            
            # Save with sanitized filename
            safe_agent_name = agent.replace('/', '_').replace('\\', '_').replace(':', '_')[:50]
            fname = self.fig_dir / f'agent_heatmap_{safe_agent_name}.png'
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            if i % 10 == 0:
                print(f"  Generated {i}/{len(agents)} heatmaps...")
        
        print(f"  All agent heatmaps saved to {self.fig_dir}")
    
    def run(self):
        print("=== Per-Agent Heatmap Generation ===\n")
        
        # Aggregate test results
        df_results = self.aggregate_test_results_per_agent()
        
        if df_results.empty:
            print("No test results found!")
            return
        
        # Validate coverage
        df_validation = self.validate_agent_coverage(df_results)
        
        # Generate heatmaps
        self.plot_per_agent_heatmaps(df_results)
        
        print("\n=== Generation Complete ===")
        print(f"Output directory: {self.output_dir}")


if __name__ == '__main__':
    generator = AgentHeatmapGenerator(
        test_root='experiments/whole_population_test_902267',
        output_dir='experiments/analysis_scripts/output/whole_population_generalization'
    )
    generator.run()
