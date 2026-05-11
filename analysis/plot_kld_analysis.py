"""
KLD Analysis - Compare agent policies to optimal policies using KL divergence.
Requires: test_data_per_agent.csv and game_agent_averaged.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class KLDAnalyzer:
    # Payoff matrices for computing optimal policies
    payoff_matrices = {
        'prisoners-dilemma': {'R': 3, 'S': 0, 'T': 5, 'P': 1},
        'stag-hunt': {'R': 4, 'S': 0, 'T': 3, 'P': 2},
        'hawk-dove': {'R': 2, 'S': 1, 'T': 3, 'P': 0}
    }
    
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.fig_dir = self.output_dir / 'figures'
        self.fig_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_optimal_cooperation_prob(self, game, opponent_coop_prob):
        """
        Compute optimal cooperation probability given opponent's cooperation probability.
        This is based on maximizing expected reward.
        """
        if game not in self.payoff_matrices:
            return 0.5
        
        payoffs = self.payoff_matrices[game]
        R, S, T, P = payoffs['R'], payoffs['S'], payoffs['T'], payoffs['P']
        
        # Expected payoff for cooperating:
        # E[cooperate] = opponent_coop_prob * R + (1 - opponent_coop_prob) * S
        exp_coop = opponent_coop_prob * R + (1 - opponent_coop_prob) * S
        
        # Expected payoff for defecting:
        # E[defect] = opponent_coop_prob * T + (1 - opponent_coop_prob) * P
        exp_defect = opponent_coop_prob * T + (1 - opponent_coop_prob) * P
        
        # Optimal pure strategy
        if exp_coop > exp_defect:
            return 1.0  # Always cooperate
        elif exp_defect > exp_coop:
            return 0.0  # Always defect
        else:
            return 0.5  # Indifferent (any mixed strategy is optimal)
    
    def calculate_kld(self, p_agent, p_optimal, epsilon=1e-10):
        """
        Calculate KL divergence between agent and optimal policies.
        KLD(P_agent || P_optimal) for binary actions (cooperate/defect).
        
        p_agent: probability agent cooperates
        p_optimal: probability optimal policy cooperates
        epsilon: small value to avoid log(0)
        """
        # Clip probabilities to avoid numerical issues
        p_agent = np.clip(p_agent, epsilon, 1 - epsilon)
        p_optimal = np.clip(p_optimal, epsilon, 1 - epsilon)
        
        # KL divergence for binary distribution
        # KLD = p_agent * log(p_agent / p_optimal) + (1-p_agent) * log((1-p_agent) / (1-p_optimal))
        kld = (p_agent * np.log(p_agent / p_optimal) + 
               (1 - p_agent) * np.log((1 - p_agent) / (1 - p_optimal)))
        
        return kld
    
    def load_test_data_with_policies(self):
        """Load test data that includes cooperation probabilities."""
        print("Loading TRUE cooperation rates for KLD analysis...")
        
        csv_path = self.data_dir / 'true_coop_rates_aggregated.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"Data not found at {csv_path}. Run extract_true_cooperation_rates.py first.")
        
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} aggregated test records")
        
        return df
    
    def compute_kld_for_all_conditions(self, df):
        """
        Compute KLD for all test conditions using TRUE cooperation rates from detailed logs.
        """
        print("\nComputing KLD using TRUE cooperation rates...")
        
        records = []
        
        for idx, row in df.iterrows():
            train_game = row['train_game']
            test_game = row['test_game']
            opponent_prob = row['opponent_prob']
            
            # Opponent cooperation probability (inverse of defection probability)
            opponent_coop_prob = 1.0 - opponent_prob
            
            # Compute optimal cooperation probability
            optimal_coop = self.compute_optimal_cooperation_prob(test_game, opponent_coop_prob)
            
            # Use TRUE cooperation rate from detailed logs
            agent_coop = row['true_coop_rate_mean']
            
            # Calculate KLD
            kld = self.calculate_kld(agent_coop, optimal_coop)
            
            records.append({
                'train_game': train_game,
                'test_game': test_game,
                'opponent_prob': opponent_prob,
                'opponent_coop_prob': opponent_coop_prob,
                'optimal_coop_prob': optimal_coop,
                'agent_coop_prob': agent_coop,
                'kld': kld
            })
        
        df_kld = pd.DataFrame.from_records(records)
        
        # Save
        csv_path = self.data_dir / 'kld_analysis.csv'
        df_kld.to_csv(csv_path, index=False)
        print(f"  Saved KLD analysis to {csv_path}")
        
        return df_kld
    
    def estimate_agent_coop_from_reward(self, normalized_reward, game, opponent_coop_prob):
        """
        Estimate agent's cooperation probability from observed normalized reward.
        This is a rough heuristic based on expected payoffs.
        """
        if game not in self.payoff_matrices:
            return 0.5
        
        payoffs = self.payoff_matrices[game]
        R, S, T, P = payoffs['R'], payoffs['S'], payoffs['T'], payoffs['P']
        
        # Denormalize reward
        all_payoffs = [R, S, T, P]
        min_r, max_r = min(all_payoffs), max(all_payoffs)
        rng = max_r - min_r
        if rng == 0:
            return 0.5
        reward = normalized_reward * rng + min_r
        
        # Expected reward given agent cooperates with prob p_agent:
        # E[R] = p_agent * opponent_coop_prob * R + 
        #        p_agent * (1 - opponent_coop_prob) * S +
        #        (1 - p_agent) * opponent_coop_prob * T +
        #        (1 - p_agent) * (1 - opponent_coop_prob) * P
        
        # Solve for p_agent:
        # E[R] = p_agent * [opponent_coop_prob * R + (1-opponent_coop_prob) * S] +
        #        (1 - p_agent) * [opponent_coop_prob * T + (1-opponent_coop_prob) * P]
        # E[R] = p_agent * A + (1 - p_agent) * B
        # E[R] = p_agent * A + B - p_agent * B
        # E[R] = p_agent * (A - B) + B
        # p_agent = (E[R] - B) / (A - B)
        
        A = opponent_coop_prob * R + (1 - opponent_coop_prob) * S
        B = opponent_coop_prob * T + (1 - opponent_coop_prob) * P
        
        if abs(A - B) < 1e-10:
            return 0.5
        
        p_agent = (reward - B) / (A - B)
        p_agent = np.clip(p_agent, 0.0, 1.0)
        
        return p_agent
    
    def plot_kld_heatmaps(self, df_kld):
        """Generate KLD heatmaps: 3 test games × 5 opponents for each trained agent."""
        print("\nGenerating KLD heatmaps...")
        
        game_order = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
        opponent_order = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for train_game in sorted(df_kld['train_game'].unique()):
            agent_data = df_kld[df_kld['train_game'] == train_game]
            
            # Create pivot table
            pivot = agent_data.pivot_table(
                index='test_game',
                columns='opponent_prob',
                values='kld',
                aggfunc='mean'
            )
            
            # Reindex
            pivot = pivot.reindex(index=game_order, columns=opponent_order)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                pivot,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
                cbar_kws={'label': 'KL Divergence'},
                ax=ax,
                linewidths=0.5
            )
            
            # Format labels
            ax.set_ylabel('Test Game', fontsize=12)
            ax.set_xlabel('Opponent Defection Probability', fontsize=12)
            ax.set_yticklabels([g.replace('-', ' ').title() for g in game_order], rotation=0)
            ax.set_xticklabels(opponent_order, rotation=0)
            
            # Title
            train_game_title = train_game.replace('-', ' ').title()
            ax.set_title(f'KL Divergence: Agent Trained on {train_game_title}\n(vs Optimal Policy)', 
                        fontsize=14, fontweight='bold')
            
            fig.tight_layout()
            
            # Save
            fname = self.fig_dir / f'kld_heatmap_{train_game}.png'
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  Saved: {fname.name}")
    
    def plot_combined_kld_heatmap(self, df_kld):
        """Generate combined KLD heatmap for all game-agents."""
        print("\nGenerating combined KLD heatmap...")
        
        game_order = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
        opponent_order = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        for idx, train_game in enumerate(game_order):
            agent_data = df_kld[df_kld['train_game'] == train_game]
            
            # Create pivot table
            pivot = agent_data.pivot_table(
                index='test_game',
                columns='opponent_prob',
                values='kld',
                aggfunc='mean'
            )
            
            # Reindex
            pivot = pivot.reindex(index=game_order, columns=opponent_order)
            
            # Create heatmap
            ax = axes[idx]
            sns.heatmap(
                pivot,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
                cbar=True if idx == 2 else False,
                cbar_kws={'label': 'KL Divergence'} if idx == 2 else {},
                ax=ax,
                linewidths=0.5
            )
            
            # Format labels
            if idx == 0:
                ax.set_ylabel('Test Game', fontsize=12)
                ax.set_yticklabels([g.replace('-', ' ').title() for g in game_order], rotation=0)
            else:
                ax.set_ylabel('')
            
            ax.set_xlabel('Opponent Defect. Prob.', fontsize=12)
            ax.set_xticklabels(opponent_order, rotation=0)
            
            # Title
            train_game_title = train_game.replace('-', ' ').title()
            ax.set_title(f'Trained on\n{train_game_title}', fontsize=12, fontweight='bold')
        
        fig.suptitle('KL Divergence from Optimal Policy: All Game-Agents', 
                    fontsize=16, fontweight='bold', y=1.02)
        fig.tight_layout()
        
        # Save
        fname = self.fig_dir / 'all_game_agents_kld_combined.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {fname.name}")
    
    def plot_kld_vs_reward(self, df_kld, df_reward):
        """Plot relationship between KLD and normalized reward."""
        print("\nGenerating KLD vs reward scatter plot...")
        
        # Merge KLD and reward data
        merged = df_kld.merge(
            df_reward[['train_game', 'test_game', 'opponent_prob', 'reward_mean']],
            on=['train_game', 'test_game', 'opponent_prob']
        )
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color by within-task vs between-task
        merged['task_match'] = merged['train_game'] == merged['test_game']
        
        for match, label in [(True, 'Within-Task'), (False, 'Between-Task')]:
            subset = merged[merged['task_match'] == match]
            ax.scatter(subset['kld'], subset['reward_mean'],
                      alpha=0.6, s=100, label=label)
        
        ax.set_xlabel('KL Divergence from Optimal', fontsize=12)
        ax.set_ylabel('Mean Normalized Reward', fontsize=12)
        ax.set_title('KL Divergence vs Performance', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        fig.tight_layout()
        
        # Save
        fname = self.fig_dir / 'kld_vs_reward.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {fname.name}")
    
    def run(self):
        """Generate all KLD analysis plots."""
        print("="*60)
        print("KLD ANALYZER")
        print("="*60)
        
        # Load data
        df_reward = self.load_test_data_with_policies()
        
        # Compute KLD
        df_kld = self.compute_kld_for_all_conditions(df_reward)
        
        # Generate plots
        self.plot_kld_heatmaps(df_kld)
        self.plot_combined_kld_heatmap(df_kld)
        self.plot_kld_vs_reward(df_kld, df_reward)
        
        print("\n" + "="*60)
        print("KLD ANALYSIS COMPLETE")
        print("="*60)
        print(f"Figures saved to: {self.fig_dir}")


if __name__ == '__main__':
    analyzer = KLDAnalyzer(
        data_dir='experiments/analysis_scripts/output/whole_population_generalization/data',
        output_dir='experiments/analysis_scripts/output/whole_population_generalization'
    )
    analyzer.run()
