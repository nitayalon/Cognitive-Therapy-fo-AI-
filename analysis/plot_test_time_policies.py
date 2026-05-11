"""
Plot test-time cooperation probabilities across all test conditions.
Shows how agents' policies change across different test setups.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class TestTimePolicyPlotter:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.fig_dir = self.output_dir / 'figures'
        self.fig_dir.mkdir(parents=True, exist_ok=True)
    
    def load_per_agent_test_data(self):
        """Load per-agent test data with TRUE cooperation rates from detailed logs."""
        csv_path = self.data_dir / 'true_coop_rates_per_agent.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"True cooperation data not found at {csv_path}. Run extract_true_cooperation_rates.py first.")
        
        print(f"Loading TRUE cooperation rates from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} test records")
        return df
    
    def compute_cooperation_probabilities(self, df):
        """
        Use TRUE cooperation rates from detailed logs.
        No inference needed - rates are measured directly from agent actions.
        """
        print("\nUsing TRUE cooperation rates from detailed logs...")
        
        # Rename column for consistency with rest of code
        df['agent_coop_prob'] = df['true_coop_rate']
        
        return df
    
    def aggregate_by_condition(self, df):
        """Average cooperation probabilities across 5 seeds for each condition."""
        print("\nAggregating cooperation probabilities across seeds...")
        
        grouped = df.groupby(['train_game', 'test_game', 'opponent_prob']).agg({
            'agent_coop_prob': ['mean', 'std', 'count']
        }).reset_index()
        
        grouped.columns = ['train_game', 'test_game', 'opponent_prob', 
                          'mean_coop_prob', 'std_coop_prob', 'n_seeds']
        
        print(f"  Aggregated into {len(grouped)} conditions")
        print(f"  Seeds per condition: {grouped['n_seeds'].min()}-{grouped['n_seeds'].max()}")
        
        return grouped
    
    def create_test_condition_labels(self, df):
        """Create readable labels for test conditions."""
        game_codes = {
            'prisoners-dilemma': 'PD',
            'stag-hunt': 'SH',
            'hawk-dove': 'HD'
        }
        
        opp_codes = {
            0.1: 'VL',
            0.3: 'L',
            0.5: 'M',
            0.7: 'H',
            0.9: 'VH'
        }
        
        df['test_condition'] = df.apply(
            lambda row: f"{game_codes.get(row['test_game'], row['test_game'])},{opp_codes.get(row['opponent_prob'], row['opponent_prob'])}",
            axis=1
        )
        
        # Create sort key
        game_order = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
        opp_order = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        df['game_order'] = df['test_game'].map({g: i for i, g in enumerate(game_order)})
        df['opp_order'] = df['opponent_prob'].map({o: i for i, o in enumerate(opp_order)})
        df['condition_order'] = df['game_order'] * len(opp_order) + df['opp_order']
        
        return df
    
    def plot_coop_prob_by_test_condition_faceted(self, df):
        """
        Create faceted plot showing cooperation probability across test conditions.
        One subplot per trained game-agent.
        """
        print("\nGenerating cooperation probability by test condition plot (faceted)...")
        
        df = self.create_test_condition_labels(df)
        df_sorted = df.sort_values('condition_order')
        
        test_conditions = df_sorted[['test_condition', 'condition_order']].drop_duplicates().sort_values('condition_order')
        test_condition_labels = test_conditions['test_condition'].tolist()
        
        train_games = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
        train_game_titles = {
            'prisoners-dilemma': "Prisoners Dilemma",
            'stag-hunt': "Stag Hunt",
            'hawk-dove': "Hawk Dove"
        }
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle('Test-Time Policy: Cooperation Probability Across Test Conditions\n' +
                    'Each subplot shows one trained game-agent (averaged across 5 seeds)',
                    fontsize=14, fontweight='bold', y=0.995)
        
        colors = {'prisoners-dilemma': '#1f77b4', 'stag-hunt': '#ff7f0e', 'hawk-dove': '#2ca02c'}
        
        for idx, train_game in enumerate(train_games):
            ax = axes[idx]
            agent_data = df_sorted[df_sorted['train_game'] == train_game]
            
            if agent_data.empty:
                continue
            
            x_positions = range(len(test_condition_labels))
            coop_values = []
            std_values = []
            
            for condition in test_condition_labels:
                cond_data = agent_data[agent_data['test_condition'] == condition]
                if not cond_data.empty:
                    coop_values.append(cond_data['mean_coop_prob'].values[0])
                    std_values.append(cond_data['std_coop_prob'].values[0])
                else:
                    coop_values.append(np.nan)
                    std_values.append(0)
            
            # Plot line with error bars
            color = colors.get(train_game, '#333333')
            ax.plot(x_positions, coop_values, 'o-', linewidth=2, markersize=6, 
                   color=color, label=train_game_titles[train_game])
            
            # Add shaded error region (std)
            coop_array = np.array(coop_values)
            std_array = np.array(std_values)
            ax.fill_between(x_positions, 
                           coop_array - std_array, 
                           coop_array + std_array,
                           alpha=0.2, color=color)
            
            # Styling
            ax.set_ylabel('P(Cooperate)', fontsize=11)
            ax.set_title(f'Training Game: {train_game_titles[train_game]}', 
                        fontsize=12, fontweight='bold', loc='left')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(-0.05, 1.05)
            
            # Add horizontal line at 0.5
            ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            
            # Add vertical lines to separate games
            for i in [4.5, 9.5]:
                ax.axvline(i, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        axes[-1].set_xticks(range(len(test_condition_labels)))
        axes[-1].set_xticklabels(test_condition_labels, rotation=45, ha='right')
        axes[-1].set_xlabel('Test Condition (Game, Opponent Defection Level)', fontsize=11)
        
        plt.tight_layout()
        
        fname = self.fig_dir / 'cooperation_prob_by_test_condition_faceted.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {fname.name}")
    
    def plot_coop_prob_by_test_condition_combined(self, df):
        """Create single plot with all three trained agents overlaid."""
        print("\nGenerating cooperation probability by test condition plot (combined)...")
        
        df = self.create_test_condition_labels(df)
        df_sorted = df.sort_values('condition_order')
        
        test_conditions = df_sorted[['test_condition', 'condition_order']].drop_duplicates().sort_values('condition_order')
        test_condition_labels = test_conditions['test_condition'].tolist()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        train_games = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
        train_game_titles = {
            'prisoners-dilemma': "Prisoners Dilemma",
            'stag-hunt': "Stag Hunt",
            'hawk-dove': "Hawk Dove"
        }
        colors = {'prisoners-dilemma': '#1f77b4', 'stag-hunt': '#ff7f0e', 'hawk-dove': '#2ca02c'}
        markers = {'prisoners-dilemma': 'o', 'stag-hunt': 's', 'hawk-dove': '^'}
        
        for train_game in train_games:
            agent_data = df_sorted[df_sorted['train_game'] == train_game]
            
            if agent_data.empty:
                continue
            
            x_positions = range(len(test_condition_labels))
            coop_values = []
            
            for condition in test_condition_labels:
                cond_data = agent_data[agent_data['test_condition'] == condition]
                if not cond_data.empty:
                    coop_values.append(cond_data['mean_coop_prob'].values[0])
                else:
                    coop_values.append(np.nan)
            
            ax.plot(x_positions, coop_values, 
                   marker=markers[train_game], linewidth=2, markersize=8,
                   color=colors[train_game], 
                   label=f'Trained on {train_game_titles[train_game]}',
                   alpha=0.8)
        
        ax.set_ylabel('P(Cooperate)', fontsize=12)
        ax.set_xlabel('Test Condition (Game, Opponent Defection Level)', fontsize=12)
        ax.set_title('Test-Time Policy: Cooperation Probability Across Test Conditions\n' +
                    'All trained agents (averaged across 5 seeds)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=11, loc='best')
        
        # Add horizontal line at 0.5
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        
        # Add vertical lines to separate games
        for i in [4.5, 9.5]:
            ax.axvline(i, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xticks(range(len(test_condition_labels)))
        ax.set_xticklabels(test_condition_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        
        fname = self.fig_dir / 'cooperation_prob_by_test_condition_combined.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {fname.name}")
    
    def run(self):
        """Generate all test-time policy plots."""
        print("="*60)
        print("TEST-TIME POLICY PLOTTER")
        print("="*60)
        
        # Load data
        df = self.load_per_agent_test_data()
        
        # Compute cooperation probabilities
        df = self.compute_cooperation_probabilities(df)
        
        # Aggregate across seeds
        df_agg = self.aggregate_by_condition(df)
        
        # Generate plots
        self.plot_coop_prob_by_test_condition_faceted(df_agg)
        self.plot_coop_prob_by_test_condition_combined(df_agg)
        
        print("\n" + "="*60)
        print("PLOTS COMPLETE")
        print("="*60)
        print(f"Figures saved to: {self.fig_dir}")


if __name__ == '__main__':
    plotter = TestTimePolicyPlotter(
        data_dir='experiments/analysis_scripts/output/whole_population_generalization/data',
        output_dir='experiments/analysis_scripts/output/whole_population_generalization'
    )
    plotter.run()
