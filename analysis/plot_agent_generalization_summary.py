"""
Plot agent generalization summary metrics:
1. Average normalized reward across all test conditions
2. Average KLD from optimal policy across all test conditions

Both plots show bar plots with confidence intervals for each trained agent.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

class AgentGeneralizationSummaryPlotter:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.fig_dir = self.output_dir / 'figures'
        self.fig_dir.mkdir(parents=True, exist_ok=True)
    
    def load_reward_data(self):
        """Load reward data from true cooperation rates file."""
        csv_path = self.data_dir / 'true_coop_rates_aggregated.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"Data not found at {csv_path}")
        
        print(f"Loading reward data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} test records")
        return df
    
    def load_kld_data(self):
        """Load KLD data."""
        csv_path = self.data_dir / 'kld_analysis.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"KLD data not found at {csv_path}")
        
        print(f"Loading KLD data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} KLD records")
        return df
    
    def compute_agent_average_reward(self, df):
        """
        Compute average normalized reward per agent across all test conditions.
        Equal weight for each task-opponent combination.
        """
        print("\nComputing average normalized reward per agent...")
        
        # Group by train_game and compute mean/CI across all test conditions
        results = []
        
        for train_game in df['train_game'].unique():
            agent_data = df[df['train_game'] == train_game]
            
            # All rewards for this agent across all test conditions
            rewards = agent_data['reward_mean'].values
            
            # Compute statistics
            mean_reward = np.mean(rewards)
            sem_reward = stats.sem(rewards)
            ci_95 = 1.96 * sem_reward  # 95% confidence interval
            
            results.append({
                'train_game': train_game,
                'mean_reward': mean_reward,
                'ci_95': ci_95,
                'n_conditions': len(rewards)
            })
            
            print(f"  {train_game}: mean={mean_reward:.3f}, CI95=±{ci_95:.3f}, n={len(rewards)}")
        
        return pd.DataFrame(results)
    
    def compute_agent_average_kld(self, df):
        """
        Compute average KLD per agent across all test conditions.
        Equal weight for each task-opponent combination.
        """
        print("\nComputing average KLD per agent...")
        
        # Group by train_game and compute mean/CI across all test conditions
        results = []
        
        for train_game in df['train_game'].unique():
            agent_data = df[df['train_game'] == train_game]
            
            # All KLD values for this agent across all test conditions
            kld_values = agent_data['kld'].values
            
            # Compute statistics
            mean_kld = np.mean(kld_values)
            sem_kld = stats.sem(kld_values)
            ci_95 = 1.96 * sem_kld  # 95% confidence interval
            
            results.append({
                'train_game': train_game,
                'mean_kld': mean_kld,
                'ci_95': ci_95,
                'n_conditions': len(kld_values)
            })
            
            print(f"  {train_game}: mean={mean_kld:.3f}, CI95=±{ci_95:.3f}, n={len(kld_values)}")
        
        return pd.DataFrame(results)
    
    def plot_average_reward_per_agent(self, df_summary):
        """Create bar plot of average normalized reward per agent."""
        print("\nGenerating average reward per agent plot...")
        
        # Game order and styling
        game_order = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
        game_labels = {
            'prisoners-dilemma': 'Prisoners\nDilemma',
            'stag-hunt': 'Stag Hunt',
            'hawk-dove': 'Hawk Dove'
        }
        colors = {
            'prisoners-dilemma': '#1f77b4',
            'stag-hunt': '#ff7f0e',
            'hawk-dove': '#2ca02c'
        }
        
        # Sort data
        df_summary['game_order'] = df_summary['train_game'].map({g: i for i, g in enumerate(game_order)})
        df_summary = df_summary.sort_values('game_order')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create bar positions
        x = np.arange(len(game_order))
        width = 0.6
        
        # Plot bars
        bars = ax.bar(
            x,
            df_summary['mean_reward'],
            width,
            yerr=df_summary['ci_95'],
            color=[colors[g] for g in df_summary['train_game']],
            capsize=8,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8,
            error_kw={'linewidth': 2, 'ecolor': 'black'}
        )
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(df_summary.iterrows()):
            ax.text(
                i,
                row['mean_reward'] + row['ci_95'] + 0.02,
                f"{row['mean_reward']:.3f}",
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )
        
        # Styling
        ax.set_ylabel('Average Normalized Reward', fontsize=13, fontweight='bold')
        ax.set_xlabel('Agent Training Game', fontsize=13, fontweight='bold')
        ax.set_title('Generalization Performance: Average Normalized Reward\n' +
                    'Averaged across all test conditions (equal weights)',
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels([game_labels[g] for g in df_summary['train_game']], fontsize=11)
        ax.set_ylim(0, max(df_summary['mean_reward'] + df_summary['ci_95']) * 1.15)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)
        
        # Add horizontal line at y=0.5 (midpoint)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Midpoint (0.5)')
        
        plt.tight_layout()
        
        # Save
        fname = self.fig_dir / 'agent_generalization_avg_reward.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {fname.name}")
    
    def plot_average_kld_per_agent(self, df_summary):
        """Create bar plot of average KLD per agent."""
        print("\nGenerating average KLD per agent plot...")
        
        # Game order and styling
        game_order = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
        game_labels = {
            'prisoners-dilemma': 'Prisoners\nDilemma',
            'stag-hunt': 'Stag Hunt',
            'hawk-dove': 'Hawk Dove'
        }
        colors = {
            'prisoners-dilemma': '#1f77b4',
            'stag-hunt': '#ff7f0e',
            'hawk-dove': '#2ca02c'
        }
        
        # Sort data
        df_summary['game_order'] = df_summary['train_game'].map({g: i for i, g in enumerate(game_order)})
        df_summary = df_summary.sort_values('game_order')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create bar positions
        x = np.arange(len(game_order))
        width = 0.6
        
        # Plot bars
        bars = ax.bar(
            x,
            df_summary['mean_kld'],
            width,
            yerr=df_summary['ci_95'],
            color=[colors[g] for g in df_summary['train_game']],
            capsize=8,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8,
            error_kw={'linewidth': 2, 'ecolor': 'black'}
        )
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(df_summary.iterrows()):
            ax.text(
                i,
                row['mean_kld'] + row['ci_95'] + 0.3,
                f"{row['mean_kld']:.2f}",
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )
        
        # Styling
        ax.set_ylabel('Average KL Divergence from Optimal', fontsize=13, fontweight='bold')
        ax.set_xlabel('Agent Training Game', fontsize=13, fontweight='bold')
        ax.set_title('Generalization Performance: Average KLD from Optimal Policy\n' +
                    'Averaged across all test conditions (equal weights)',
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels([game_labels[g] for g in df_summary['train_game']], fontsize=11)
        ax.set_ylim(0, max(df_summary['mean_kld'] + df_summary['ci_95']) * 1.15)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save
        fname = self.fig_dir / 'agent_generalization_avg_kld.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {fname.name}")
    
    def run(self):
        """Generate all agent generalization summary plots."""
        print("="*70)
        print("AGENT GENERALIZATION SUMMARY PLOTTER")
        print("="*70)
        
        # Load data
        df_reward = self.load_reward_data()
        df_kld = self.load_kld_data()
        
        # Compute summaries
        reward_summary = self.compute_agent_average_reward(df_reward)
        kld_summary = self.compute_agent_average_kld(df_kld)
        
        # Generate plots
        self.plot_average_reward_per_agent(reward_summary)
        self.plot_average_kld_per_agent(kld_summary)
        
        print("\n" + "="*70)
        print("PLOTS COMPLETE")
        print("="*70)
        print(f"Figures saved to: {self.fig_dir}")


if __name__ == '__main__':
    plotter = AgentGeneralizationSummaryPlotter(
        data_dir='experiments/analysis_scripts/output/whole_population_generalization/data',
        output_dir='experiments/analysis_scripts/output/whole_population_generalization'
    )
    plotter.run()
