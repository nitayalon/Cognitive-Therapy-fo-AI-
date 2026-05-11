"""
Plot KLD from optimal policy across all test conditions for whole population agents.
Similar to the task-opponent analysis but adapted for whole population paradigm.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class KLDTestConditionPlotter:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.fig_dir = self.output_dir / 'figures'
        self.fig_dir.mkdir(parents=True, exist_ok=True)
    
    def load_kld_data(self):
        """Load pre-computed KLD analysis."""
        csv_path = self.data_dir / 'kld_analysis.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"KLD data not found at {csv_path}. Run plot_kld_analysis.py first.")
        
        print(f"Loading KLD data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} KLD records")
        return df
    
    def create_test_condition_labels(self, df):
        """Create readable labels for test conditions."""
        # Map games to short codes
        game_codes = {
            'prisoners-dilemma': 'PD',
            'stag-hunt': 'SH',
            'hawk-dove': 'HD'
        }
        
        # Map opponent probs to short codes
        opp_codes = {
            0.1: 'VL',  # Very Low
            0.3: 'L',   # Low
            0.5: 'M',   # Mid
            0.7: 'H',   # High
            0.9: 'VH'   # Very High
        }
        
        # Create condition labels
        df['test_condition'] = df.apply(
            lambda row: f"{game_codes.get(row['test_game'], row['test_game'])},{opp_codes.get(row['opponent_prob'], row['opponent_prob'])}",
            axis=1
        )
        
        # Create sort key for ordering
        game_order = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
        opp_order = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        df['game_order'] = df['test_game'].map({g: i for i, g in enumerate(game_order)})
        df['opp_order'] = df['opponent_prob'].map({o: i for i, o in enumerate(opp_order)})
        df['condition_order'] = df['game_order'] * len(opp_order) + df['opp_order']
        
        return df
    
    def plot_kld_by_test_condition_faceted(self, df):
        """
        Create faceted plot showing KLD across test conditions.
        One subplot per trained game-agent.
        """
        print("\nGenerating KLD by test condition plot (faceted)...")
        
        # Prepare data
        df = self.create_test_condition_labels(df)
        df_sorted = df.sort_values('condition_order')
        
        # Get unique test conditions in order
        test_conditions = df_sorted[['test_condition', 'condition_order']].drop_duplicates().sort_values('condition_order')
        test_condition_labels = test_conditions['test_condition'].tolist()
        
        # Create figure with 3 subplots (one per training game)
        train_games = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
        train_game_titles = {
            'prisoners-dilemma': "Prisoners Dilemma",
            'stag-hunt': "Stag Hunt",
            'hawk-dove': "Hawk Dove"
        }
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle('Test Policy Optimality: KLD Between Agent Test Policy and Optimal Policy\n' +
                    'Each subplot shows one trained game-agent tested across all conditions',
                    fontsize=14, fontweight='bold', y=0.995)
        
        colors = {'prisoners-dilemma': '#1f77b4', 'stag-hunt': '#ff7f0e', 'hawk-dove': '#2ca02c'}
        
        for idx, train_game in enumerate(train_games):
            ax = axes[idx]
            agent_data = df_sorted[df_sorted['train_game'] == train_game]
            
            if agent_data.empty:
                continue
            
            # Plot KLD values
            x_positions = range(len(test_condition_labels))
            kld_values = []
            
            for condition in test_condition_labels:
                cond_data = agent_data[agent_data['test_condition'] == condition]
                if not cond_data.empty:
                    kld_values.append(cond_data['kld'].values[0])
                else:
                    kld_values.append(np.nan)
            
            # Plot line
            color = colors.get(train_game, '#333333')
            ax.plot(x_positions, kld_values, 'o-', linewidth=2, markersize=6, 
                   color=color, label=train_game_titles[train_game])
            
            # Styling
            ax.set_ylabel('KLD (Moment | Optimal)', fontsize=11)
            ax.set_title(f'Training Game: {train_game_titles[train_game]}', 
                        fontsize=12, fontweight='bold', loc='left')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(bottom=0)
            
            # Add vertical lines to separate games
            for i in [4.5, 9.5]:  # Between PD-SH and SH-HD
                ax.axvline(i, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Set x-axis labels on bottom plot only
        axes[-1].set_xticks(range(len(test_condition_labels)))
        axes[-1].set_xticklabels(test_condition_labels, rotation=45, ha='right')
        axes[-1].set_xlabel('Test Condition (Game, Opponent Defection Level)', fontsize=11)
        
        plt.tight_layout()
        
        # Save
        fname = self.fig_dir / 'kld_by_test_condition_faceted.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {fname.name}")
    
    def plot_kld_by_test_condition_combined(self, df):
        """
        Create single plot with all three trained agents overlaid.
        """
        print("\nGenerating KLD by test condition plot (combined)...")
        
        # Prepare data
        df = self.create_test_condition_labels(df)
        df_sorted = df.sort_values('condition_order')
        
        # Get unique test conditions in order
        test_conditions = df_sorted[['test_condition', 'condition_order']].drop_duplicates().sort_values('condition_order')
        test_condition_labels = test_conditions['test_condition'].tolist()
        
        # Create figure
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
            
            # Plot KLD values
            x_positions = range(len(test_condition_labels))
            kld_values = []
            
            for condition in test_condition_labels:
                cond_data = agent_data[agent_data['test_condition'] == condition]
                if not cond_data.empty:
                    kld_values.append(cond_data['kld'].values[0])
                else:
                    kld_values.append(np.nan)
            
            # Plot line
            ax.plot(x_positions, kld_values, 
                   marker=markers[train_game], linewidth=2, markersize=8,
                   color=colors[train_game], 
                   label=f'Trained on {train_game_titles[train_game]}',
                   alpha=0.8)
        
        # Styling
        ax.set_ylabel('KLD (Moment | Optimal)', fontsize=12)
        ax.set_xlabel('Test Condition (Game, Opponent Defection Level)', fontsize=12)
        ax.set_title('Test Policy Optimality: KLD Between Agent Policy and Optimal Policy\n' +
                    'All trained agents tested across all conditions',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=11, loc='best')
        
        # Add vertical lines to separate games
        for i in [4.5, 9.5]:  # Between PD-SH and SH-HD
            ax.axvline(i, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Set x-axis labels
        ax.set_xticks(range(len(test_condition_labels)))
        ax.set_xticklabels(test_condition_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save
        fname = self.fig_dir / 'kld_by_test_condition_combined.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {fname.name}")
    
    def run(self):
        """Generate all KLD by test condition plots."""
        print("="*60)
        print("KLD BY TEST CONDITION PLOTTER")
        print("="*60)
        
        # Load data
        df = self.load_kld_data()
        
        # Generate plots
        self.plot_kld_by_test_condition_faceted(df)
        self.plot_kld_by_test_condition_combined(df)
        
        print("\n" + "="*60)
        print("PLOTS COMPLETE")
        print("="*60)
        print(f"Figures saved to: {self.fig_dir}")


if __name__ == '__main__':
    plotter = KLDTestConditionPlotter(
        data_dir='experiments/analysis_scripts/output/whole_population_generalization/data',
        output_dir='experiments/analysis_scripts/output/whole_population_generalization'
    )
    plotter.run()
