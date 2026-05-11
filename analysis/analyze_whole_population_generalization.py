"""
Whole Population Generalization Analysis
Adapts analyze_vanilla_generalization.py for the whole population paradigm.
Loads all training and test results, unifies them, and produces 3x5 heatmaps for PD, SH, and HD.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

class WholePopulationGeneralizationAnalyzer:
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
        import numpy as np
        return np.clip(norm, 0, 1)
    
    def export_cooperation_probability_summary(self, output_csv=None):
        """Export summary stats for cooperation probability per epoch per agent to a CSV for better plotting."""
        import scipy.stats as st
        if output_csv is None:
            output_csv = self.fig_dir / 'cooperation_probability_summary.csv'
        records = []
        for game in self.games:
            group = []
            pattern = str(self.train_root / 'training/whole_population_task_*/checkpoints/detailed_training_logs/detailed_training_log.csv')
            import glob
            for log_path in glob.glob(pattern):
                try:
                    df = pd.read_csv(log_path)
                    if 'policy_prob_cooperate' not in df.columns:
                        continue
                    if 'game_name' not in df.columns:
                        continue
                    if game not in df['game_name'].iloc[0].lower():
                        continue
                    group.append(df.sort_values('epoch')['policy_prob_cooperate'].values)
                except Exception:
                    continue
            if not group:
                continue
            max_len = max(len(arr) for arr in group)
            arrs = [np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) for arr in group]
            arrs = np.stack(arrs)
            mean = np.nanmean(arrs, axis=0)
            p05 = np.nanpercentile(arrs, 5, axis=0)
            p95 = np.nanpercentile(arrs, 95, axis=0)
            for epoch in range(max_len):
                records.append({
                    "Agent": game.replace('-', ' ').title(),
                    "Iteration": epoch,
                    "Mean_P_cooperation": mean[epoch],
                    "P05": p05[epoch],
                    "P95": p95[epoch]
                })
        df_out = pd.DataFrame.from_records(records)
        df_out.to_csv(output_csv, index=False)
        print(f"  Saved cooperation probability summary to {output_csv}")
    
    def plot_cooperation_probability_by_game(self):
        """Plot mean cooperation probability trajectory for each game, aggregated over all opponents."""
        print("Plotting mean cooperation probability by game (aggregated over opponents)...")
        import glob
        game_map = {
            'Prisoners-Dilemma': 'prisoners-dilemma',
            'Stag-Hunt': 'stag-hunt',
            'Hawk-Dove': 'hawk-dove',
        }
        coop_dict = {g: [] for g in self.games}
        # Find all training log files
        pattern = str(self.train_root / 'training/whole_population_task_*/checkpoints/detailed_training_logs/detailed_training_log.csv')
        for log_path in glob.glob(pattern):
            try:
                df = pd.read_csv(log_path)
                if 'policy_prob_cooperate' not in df.columns:
                    print(f"  [DEBUG] policy_prob_cooperate column missing in {log_path}")
                    continue
                df = df[['game_name', 'epoch', 'policy_prob_cooperate']]
            except Exception:
                continue
            game = game_map.get(df['game_name'].iloc[0], None)
            if game is None:
                continue
            coop_traj = df.sort_values('epoch')['policy_prob_cooperate'].values
            coop_dict[game].append(coop_traj)
        # Debug: print how many trajectories per game and example values
        for game in self.games:
            print(f"  [DEBUG] {game}: {len(coop_dict[game])} trajectories")
            if coop_dict[game]:
                print(f"    [DEBUG] Example values (first 5): {coop_dict[game][0][:5]}")
        # Plot
        # Aggregate mean and CI per episode per game for memory efficiency
        import scipy.stats as st
        agg_records = []
        for game in self.games:
            group = coop_dict[game]
            if not group:
                continue
            max_len = max(len(arr) for arr in group)
            arrs = [np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) for arr in group]
            arrs = np.stack(arrs)
            mean = np.nanmean(arrs, axis=0)
            sem = st.sem(arrs, axis=0, nan_policy='omit')
            ci95 = sem * 1.96
            for epoch in range(max_len):
                agg_records.append({
                    'Game': game.replace('-', ' ').title(),
                    'Episode': epoch,
                    'Mean': mean[epoch],
                    'CI95': ci95[epoch]
                })
        df_agg = pd.DataFrame.from_records(agg_records)
        # Use seaborn lineplot with CI, hue by Game
        ax = sns.lineplot(
            data=df_agg,
            x='Episode',
            y='Mean',
            hue='Game',
            errorbar=None,
            linewidth=2
        )
        ax.set(title='Mean Cooperation Probability by Game (Aggregated)',
               xlabel='Episode', ylabel='Cooperation Probability', ylim=(0, 1))
        ax.legend(title='Game')
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(self.fig_dir / 'mean_cooperation_probability_by_game.png', dpi=150)
        fig.clf()
        print("  Saved mean_cooperation_probability_by_game.png")
    
    def plot_training_loss_by_opponent(self, _):
        """Plot training loss by opponent type per game (mean ± 95% CI), processing each file individually."""
        print("Plotting training loss by opponent type (chunked)...")
        import glob
        game_map = {
            'Prisoners-Dilemma': 'prisoners-dilemma',
            'Stag-Hunt': 'stag-hunt',
            'Hawk-Dove': 'hawk-dove',
        }
        opp_prob_map = {
            0.1: 'very_low',
            0.3: 'low',
            0.5: 'mid',
            0.7: 'high',
            0.9: 'very_high',
        }
        # Prepare storage for all loss trajectories
        records = []
        pattern = str(self.train_root / 'training/whole_population_task_*/checkpoints/detailed_training_logs/detailed_training_log.csv')
        for log_path in glob.glob(pattern):
            try:
                df = pd.read_csv(log_path, usecols=['game_name', 'true_opponent_defect_prob', 'epoch', 'total_loss'])
            except Exception:
                continue
            game = game_map.get(df['game_name'].iloc[0], None)
            opp = opp_prob_map.get(round(float(df['true_opponent_defect_prob'].iloc[0]), 2), None)
            if game is None or opp is None:
                continue
            for idx, row in df.iterrows():
                records.append({
                    'Game': game.replace('-', ' ').title(),
                    'Opponent': opp,
                    'Epoch': row['epoch'],
                    'Loss': row['total_loss']
                })
        df_loss = pd.DataFrame.from_records(records)
        if not df_loss.empty:
            g = sns.FacetGrid(df_loss, col="Game", hue="Opponent", sharey=False, height=5, aspect=1.2)
            g.map_dataframe(sns.lineplot, x="Epoch", y="Loss", errorbar="ci", linewidth=2)
            g.add_legend(title="Opponent")
            g.set_axis_labels("Epoch", "Training Loss")
            g.set_titles(col_template="{col_name}")
            for ax in g.axes.flat:
                ax.set_ylim(bottom=0)
            g.fig.tight_layout()
            g.savefig(self.fig_dir / 'training_loss_by_opponent.png', dpi=150)
            g.fig.clf()
            print("  Saved training_loss_by_opponent.png")

    def plot_generalization_matrix(self, test_df):
        """Plot generalization matrix: mean reward for each train/test condition."""
        print("Plotting generalization matrix heatmap...")
        # Assume test_df has columns: train_game, train_opponent_range, test_game, test_opponent_range, mean_reward
        # If not, try to infer from available columns
        if 'train_game' not in test_df.columns or 'test_game' not in test_df.columns:
            print("  Skipping generalization matrix: required columns not found.")
            return

        pivot = test_df.pivot_table(index=['train_game', 'train_opponent_range'],
                                    columns=['test_game', 'test_opponent_range'],
                                    values='mean_reward', aggfunc='mean')
        ax = sns.heatmap(pivot, annot=False, cmap='RdYlGn', cbar=True)
        ax.set(title='Generalization Matrix: Mean Reward', ylabel='Training Condition (Game, Opponent)', xlabel='Test Condition (Game, Opponent)')
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(self.fig_dir / 'generalization_matrix_heatmap.png', dpi=150)
        fig.clf()
        print("  Saved generalization_matrix_heatmap.png")

    def plot_training_policy_kld_grid(self, train_df):
        """Plot 3x5 grid of KLD between agent's training policy and optimal policy."""
        print("Plotting training policy vs optimal KLD grid...")
        # Placeholder: actual KLD computation requires agent and optimal policy rates
        # Here, just plot a dummy grid for structure using seaborn heatmap
        dummy = np.random.rand(3, 5)
        ax = sns.heatmap(dummy, annot=True, fmt='.2f', cmap='Blues', cbar=True)
        ax.set(title='Training Policy vs Optimal KLD (Dummy)', ylabel='Game', xlabel='Opponent Range')
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(self.fig_dir / 'training_vs_optimal_kld_grid_3x5.png', dpi=150)
        fig.clf()
        print("  Saved training_vs_optimal_kld_grid_3x5.png")

    def __init__(self, train_root, test_root, output_dir):
        self.train_root = Path(train_root)
        self.test_root = Path(test_root)
        self.output_dir = Path(output_dir)
        self.fig_dir = self.output_dir / 'figures'
        self.data_dir = self.output_dir / 'data'
        for d in [self.fig_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.games = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
        self.opp_ranges = ['very_low', 'low', 'mid', 'high', 'very_high']

    def load_all_training(self):
        dfs = []
        for task_dir in sorted(self.train_root.glob('training/whole_population_task_*')):
            log = task_dir / 'checkpoints/detailed_training_logs/detailed_training_log.csv'
            if log.exists():
                df = pd.read_csv(log)
                df['task_id'] = task_dir.name
                dfs.append(df)
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()

    def load_all_testing(self):
        dfs = []
        # Recursively search for all detailed_testing_log.csv files in all eval_task_* folders
        for log in self.test_root.glob('testing/whole_population_task_*/logs/eval_task_*/detailed_testing_log.csv'):
            try:
                df = pd.read_csv(log)
                df['task_id'] = log.parts[-5]  # whole_population_task_xxx
                df['eval_task'] = log.parts[-3]  # eval_task_xxx
                dfs.append(df)
            except Exception as e:
                print(f"Failed to load {log}: {e}")
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()

    def plot_heatmaps(self, df, value_col, title, fname):
        # Assumes df has columns: 'game', 'opponent_range', value_col
        pivot = df.pivot(index='game', columns='opponent_range', values=value_col)
        ax = sns.heatmap(pivot.loc[self.games, self.opp_ranges], annot=True, fmt='.3f', cmap='RdYlGn', cbar=True)
        ax.set(title=title, ylabel='Game', xlabel='Opponent Range')
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(self.fig_dir / fname)
        fig.clf()

    def aggregate_test_results_per_agent(self):
        """
        Aggregate all test results for each agent across all 15 setups (3 games × 5 opponents).
        Returns a DataFrame with columns: agent_id, train_game, test_game, opponent_prob, mean_reward, normalized_reward
        """
        print("Aggregating test results per agent across all setups...")
        import json
        
        records = []
        test_dirs = sorted(self.test_root.glob('testing/whole_population_task_*'))
        
        for task_dir in test_dirs:
            # Read experiment config to get train/test game info
            config_path = task_dir / 'experiment_config.json'
            if not config_path.exists():
                continue
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            test_game = config.get('test_game', None)
            train_game = config.get('train_game', None)
            
            if test_game is None or train_game is None:
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
                df = pd.read_csv(log_path)
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
                'train_game': train_game,
                'test_game': test_game,
                'opponent_prob': opponent_prob,
                'mean_reward': mean_reward,
                'normalized_reward': normalized_reward
            })
        
        df_results = pd.DataFrame.from_records(records)
        
        # Save to CSV for inspection
        csv_path = self.data_dir / 'per_agent_test_results.csv'
        df_results.to_csv(csv_path, index=False)
        print(f"  Saved aggregated results to {csv_path}")
        
        return df_results
    
    def aggregate_by_game_agent(self, df_results):
        """
        Average performance across the 5 seed replicates for each game-agent.
        Returns a DataFrame with columns: train_game, test_game, opponent_prob, 
        mean_normalized_reward, std_normalized_reward, n_seeds
        """
        print("\nAveraging performance across 5 seeds for each game-agent...")
        
        # Group by (train_game, test_game, opponent_prob) and compute statistics
        grouped = df_results.groupby(['train_game', 'test_game', 'opponent_prob']).agg({
            'normalized_reward': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['train_game', 'test_game', 'opponent_prob', 
                          'mean_normalized_reward', 'std_normalized_reward', 'n_seeds']
        
        # Save aggregated results
        csv_path = self.data_dir / 'game_agent_averaged_results.csv'
        grouped.to_csv(csv_path, index=False)
        print(f"  Saved averaged results to {csv_path}")
        
        # Print summary
        print(f"\n  Summary statistics:")
        for train_game in grouped['train_game'].unique():
            subset = grouped[grouped['train_game'] == train_game]
            print(f"    {train_game}: {len(subset)} test conditions")
            print(f"      Seeds per condition: {subset['n_seeds'].min()}-{subset['n_seeds'].max()}")
            if subset['n_seeds'].min() != 5:
                print(f"      WARNING: Expected 5 seeds per condition, found {subset['n_seeds'].min()}-{subset['n_seeds'].max()}")
        
        return grouped
    
    def plot_game_agent_heatmaps(self, df_aggregated):
        """
        Generate one heatmap per trained game-agent showing mean normalized reward across all 15 test setups.
        Each heatmap is 3 test games (rows) × 5 opponents (columns).
        """
        print("\nGenerating heatmaps for each game-agent (averaged across 5 seeds)...")
        
        game_order = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
        opponent_order = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for train_game in sorted(df_aggregated['train_game'].unique()):
            agent_data = df_aggregated[df_aggregated['train_game'] == train_game]
            
            # Create pivot table: test games (rows) × opponents (columns)
            pivot = agent_data.pivot_table(
                index='test_game',
                columns='opponent_prob',
                values='mean_normalized_reward',
                aggfunc='mean'
            )
            
            # Reindex to ensure consistent ordering
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
                cbar_kws={'label': 'Mean Normalized Reward'},
                ax=ax,
                linewidths=0.5,
                center=0.5
            )
            
            # Format labels
            ax.set_ylabel('Test Game', fontsize=12)
            ax.set_xlabel('Opponent Defection Probability', fontsize=12)
            ax.set_yticklabels([g.replace('-', ' ').title() for g in game_order], rotation=0)
            ax.set_xticklabels(opponent_order, rotation=0)
            
            # Title
            train_game_title = train_game.replace('-', ' ').title()
            ax.set_title(f'Agent Trained on {train_game_title}\n(Mean across 5 seeds)', 
                        fontsize=14, fontweight='bold')
            
            fig.tight_layout()
            
            # Save
            fname = self.fig_dir / f'game_agent_heatmap_{train_game}.png'
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  Saved heatmap for {train_game_title} agent")
        
        print(f"  All game-agent heatmaps saved to {self.fig_dir}")
    
    def plot_combined_game_agent_heatmap(self, df_aggregated):
        """
        Generate a single large heatmap showing all 3 game-agents side by side.
        Layout: 3 rows (test games) × 15 columns (5 opponents × 3 train games)
        """
        print("\nGenerating combined heatmap for all game-agents...")
        
        game_order = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
        opponent_order = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        # Create figure with 3 subplots side by side
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        for idx, train_game in enumerate(game_order):
            agent_data = df_aggregated[df_aggregated['train_game'] == train_game]
            
            # Create pivot table
            pivot = agent_data.pivot_table(
                index='test_game',
                columns='opponent_prob',
                values='mean_normalized_reward',
                aggfunc='mean'
            )
            
            # Reindex to ensure consistent ordering
            pivot = pivot.reindex(index=game_order, columns=opponent_order)
            
            # Create heatmap
            ax = axes[idx]
            sns.heatmap(
                pivot,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                vmin=0,
                vmax=1,
                cbar=True if idx == 2 else False,
                cbar_kws={'label': 'Mean Normalized Reward'} if idx == 2 else {},
                ax=ax,
                linewidths=0.5,
                center=0.5
            )
            
            # Format labels
            if idx == 0:
                ax.set_ylabel('Test Game', fontsize=12)
                ax.set_yticklabels([g.replace('-', ' ').title() for g in game_order], rotation=0)
            else:
                ax.set_ylabel('')
            
            ax.set_xlabel('Opponent Defection Prob.', fontsize=12)
            ax.set_xticklabels(opponent_order, rotation=0)
            
            # Title
            train_game_title = train_game.replace('-', ' ').title()
            ax.set_title(f'Trained on\n{train_game_title}', fontsize=12, fontweight='bold')
        
        fig.suptitle('Generalization Performance: All Game-Agents (Mean across 5 seeds)', 
                    fontsize=16, fontweight='bold', y=1.02)
        fig.tight_layout()
        
        # Save
        fname = self.fig_dir / 'all_game_agents_combined_heatmap.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved combined heatmap to {fname}")
    
    def validate_agent_coverage(self, df_results):
        """
        Validate that each agent has results for all 15 setups (3 games × 5 opponents).
        """
        print("Validating agent coverage across all 15 setups...")
        
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
            print("  Agents with incomplete coverage:")
            for _, row in df_validation[~df_validation['complete']].iterrows():
                print(f"    {row['agent_id']}: {row['total_setups']}/{row['expected_setups']} setups")
                print(f"      Missing: {row['missing_setups']}")
        
        return df_validation
    
    def plot_per_agent_heatmaps(self, df_results):
        """
        Generate one heatmap per agent showing globally normalized reward across all 15 setups.
        Each heatmap is 3 games (rows) × 5 opponents (columns).
        """
        print("Generating per-agent heatmaps with globally normalized rewards...")
        
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
        
        for agent in agents:
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
                linewidths=0.5
            )
            
            # Format labels
            ax.set_ylabel('Test Game', fontsize=12)
            ax.set_xlabel('Opponent Defection Probability', fontsize=12)
            ax.set_yticklabels([g.replace('-', ' ').title() for g in game_order], rotation=0)
            ax.set_xticklabels(opponent_order, rotation=0)
            
            # Title with agent ID (truncated for readability)
            agent_short = agent[:20] if len(agent) > 20 else agent
            ax.set_title(f'Agent: {agent_short}\nGlobally Normalized Test Reward', fontsize=14, fontweight='bold')
            
            fig.tight_layout()
            
            # Save with sanitized filename
            safe_agent_name = agent.replace('/', '_').replace('\\', '_')[:50]
            fname = self.fig_dir / f'agent_heatmap_{safe_agent_name}.png'
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  Saved heatmap for agent {agent_short}")
        
        print(f"  All agent heatmaps saved to {self.fig_dir}")

    def run(self):
        print('Loading training data...')
        train_df = self.load_all_training()
        print('Loading testing data...')
        test_df = self.load_all_testing()
        if train_df.empty or test_df.empty:
            print('No data found!')
            return

        # Map game_name to canonical names
        game_map = {
            'Prisoners-Dilemma': 'prisoners-dilemma',
            'Stag-Hunt': 'stag-hunt',
            'Hawk-Dove': 'hawk-dove',
        }
        test_df['game'] = test_df['game_name'].map(game_map)

        # Map true_opponent_defect_prob to opponent_range
        opp_prob_map = {
            0.1: 'very_low',
            0.3: 'low',
            0.5: 'mid',
            0.7: 'high',
            0.9: 'very_high',
        }
        # Ensure float mapping
        test_df['opponent_range'] = test_df['true_opponent_defect_prob'].round(2).map(opp_prob_map)

        # Compute mean agent_reward by game/opponent, then normalize by test game
        summary = test_df.groupby(['game', 'opponent_range'])['agent_reward'].mean().reset_index()
        summary['normalized_reward'] = summary.apply(
            lambda row: self.normalize_reward(row['agent_reward'], row['game']), axis=1)
        self.plot_heatmaps(summary, 'normalized_reward', 'Mean Test Agent Normalized Reward (Whole Population)', 'mean_test_reward_heatmap.png')

        # Training loss plot
        self.plot_training_loss_by_opponent(train_df)

        # Mean cooperation probability by game (aggregated)
        self.plot_cooperation_probability_by_game()

        # Export cooperation probability summary for better plotting
        self.export_cooperation_probability_summary()

        # Generalization matrix plot (if possible)
        # This requires test_df to have train/test game/opponent columns and mean_reward
        if {'train_game', 'train_opponent_range', 'test_game', 'test_opponent_range', 'mean_reward'}.issubset(test_df.columns):
            self.plot_generalization_matrix(test_df)
        else:
            print('Skipping generalization matrix: required columns not found in test_df.')

        # KLD analysis (placeholder)
        self.plot_training_policy_kld_grid(train_df)

        # NEW: Per-agent heatmap analysis
        print('\n=== Per-Agent Heatmap Analysis ===')
        df_agent_results = self.aggregate_test_results_per_agent()
        
        # NEW: Game-agent averaged analysis
        print('\n=== Game-Agent Averaged Analysis (5 seeds per game-agent) ===')
        df_game_agent_avg = self.aggregate_by_game_agent(df_agent_results)
        self.plot_game_agent_heatmaps(df_game_agent_avg)
        self.plot_combined_game_agent_heatmap(df_game_agent_avg)
        
        # Validation and individual agent heatmaps
        df_validation = self.validate_agent_coverage(df_agent_results)
        # self.plot_per_agent_heatmaps(df_agent_results)  # Commented out - too many plots

        print('Analysis complete. Figures saved to', self.fig_dir)

if __name__ == '__main__':
    # Example usage: update these paths as needed
    analyzer = WholePopulationGeneralizationAnalyzer(
        train_root='experiments/whole_population_train_902266',
        test_root='experiments/whole_population_test_902267',
        output_dir='experiments/analysis_scripts/output/whole_population_generalization'
    )
    analyzer.run()
