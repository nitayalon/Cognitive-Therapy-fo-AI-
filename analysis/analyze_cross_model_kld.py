"""
Cross-Model KLD Analysis: Compare policies between models trained on different games.

Instead of comparing each model's train vs test policy (current approach),
this computes KLD between models trained on different games when tested on the same condition.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class CrossModelKLDAnalyzer:
    def __init__(self, kld_csv_path, output_dir):
        self.kld_csv_path = Path(kld_csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir = self.output_dir / 'figures'
        self.fig_dir.mkdir(exist_ok=True)
        
        # Load existing KLD analysis (has train and test cooperation rates)
        self.df = pd.read_csv(kld_csv_path)
        
        self.games = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
        self.game_abbrev = {
            'prisoners-dilemma': 'PD',
            'stag-hunt': 'SH',
            'hawk-dove': 'HD'
        }
        self.opp_ranges = ['very_low', 'low', 'mid', 'high', 'very_high']
        self.opp_range_labels = {
            'very_low': 'Very Low',
            'low': 'Low',
            'mid': 'Mid',
            'high': 'High',
            'very_high': 'Very High'
        }
    
    def binary_kl_divergence(self, p, q, epsilon=0.001):
        """Binary KL divergence: KL(P||Q) for cooperation rates."""
        p = np.clip(p, epsilon, 1 - epsilon)
        q = np.clip(q, epsilon, 1 - epsilon)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    
    def compute_cross_model_kld(self):
        """
        Compute KLD between models trained on different games.
        
        For each test condition (test_game, test_opponent_range):
        - Get all models' test cooperation rates
        - Compute KLD between each pair of training games
        - This shows how differently-trained models behave on the same test
        """
        print("\n" + "="*80)
        print("COMPUTING CROSS-MODEL KLD")
        print("="*80)
        
        cross_kld_data = []
        
        # For each unique test condition
        test_conditions = self.df[['test_game', 'test_opponent_range']].drop_duplicates()
        
        for _, test_cond in test_conditions.iterrows():
            test_game = test_cond['test_game']
            test_opp = test_cond['test_opponent_range']
            
            # Get all models tested on this condition, grouped by training game
            test_data = self.df[
                (self.df['test_game'] == test_game) & 
                (self.df['test_opponent_range'] == test_opp)
            ]
            
            # Compute mean cooperation rate for each training game on this test condition
            train_game_coops = {}
            for train_game in self.games:
                game_data = test_data[test_data['train_game'] == train_game]
                if len(game_data) > 0:
                    train_game_coops[train_game] = game_data['test_coop_rate'].mean()
            
            # Compute KLD between each pair of training games
            for train_game_1 in self.games:
                for train_game_2 in self.games:
                    if train_game_1 == train_game_2:
                        continue
                    
                    if train_game_1 in train_game_coops and train_game_2 in train_game_coops:
                        p = train_game_coops[train_game_1]
                        q = train_game_coops[train_game_2]
                        kld = self.binary_kl_divergence(p, q)
                        
                        cross_kld_data.append({
                            'test_game': test_game,
                            'test_opponent_range': test_opp,
                            'train_game_1': train_game_1,  # Reference distribution P
                            'train_game_2': train_game_2,  # Comparison distribution Q
                            'coop_rate_1': p,
                            'coop_rate_2': q,
                            'coop_diff': p - q,
                            'kld': kld
                        })
        
        self.cross_kld_df = pd.DataFrame(cross_kld_data)
        
        # Save to CSV
        csv_path = self.output_dir / 'cross_model_kld.csv'
        self.cross_kld_df.to_csv(csv_path, index=False)
        print(f"✓ Saved cross-model KLD data: {csv_path}")
        print(f"✓ Total comparisons: {len(self.cross_kld_df)}")
        
        # Print statistics
        print(f"\nCross-Model KLD Statistics:")
        print(f"  Min: {self.cross_kld_df['kld'].min():.6f}")
        print(f"  Max: {self.cross_kld_df['kld'].max():.6f}")
        print(f"  Mean: {self.cross_kld_df['kld'].mean():.6f}")
        print(f"  Median: {self.cross_kld_df['kld'].median():.6f}")
        print(f"  Std: {self.cross_kld_df['kld'].std():.6f}")
        
        # Find most divergent cases
        print(f"\nTop 10 Most Divergent Model Pairs:")
        top_divergent = self.cross_kld_df.nlargest(10, 'kld')
        for idx, row in top_divergent.iterrows():
            print(f"  {self.game_abbrev[row['train_game_1']]} vs {self.game_abbrev[row['train_game_2']]} "
                  f"on {self.game_abbrev[row['test_game']]}-{row['test_opponent_range']}: "
                  f"KLD={row['kld']:.4f} (coop: {row['coop_rate_1']:.3f} vs {row['coop_rate_2']:.3f})")
    
    def compute_training_condition_kld_grid(self):
        """
        Compute average KLD for each training condition (5×3 grid).
        For each training condition, compute average KLD to all other training conditions.
        """
        print("\n" + "="*80)
        print("COMPUTING KLD GRID (5×3) FOR TRAINING CONDITIONS")
        print("="*80)
        
        grid_kld_data = []
        
        # For each training condition (15 total)
        for train_game in self.games:
            for train_opp in self.opp_ranges:
                # Get all KLD values where this condition is train_game_1 (reference)
                condition_kld = self.cross_kld_df[
                    (self.cross_kld_df['train_game_1'] == train_game) &
                    (self.cross_kld_df['train_game_1'] == train_game)  # From this condition
                ]
                
                # Need to filter by opponent range - let's get from original df
                # Get model_ids for this training condition
                models_in_condition = self.df[
                    (self.df['train_game'] == train_game) &
                    (self.df['train_opponent_range'] == train_opp)
                ]['model_id'].unique()
                
                # Compute average cooperation rate for this training condition
                train_coop = self.df[
                    (self.df['train_game'] == train_game) &
                    (self.df['train_opponent_range'] == train_opp)
                ]['train_coop_rate'].mean()
                
                # Compute KLD to all other training conditions
                kld_values = []
                for other_game in self.games:
                    for other_opp in self.opp_ranges:
                        if train_game == other_game and train_opp == other_opp:
                            continue  # Skip self
                        
                        # Get average cooperation for other condition
                        other_coop = self.df[
                            (self.df['train_game'] == other_game) &
                            (self.df['train_opponent_range'] == other_opp)
                        ]['train_coop_rate'].mean()
                        
                        # Compute KLD
                        kld = self.binary_kl_divergence(train_coop, other_coop)
                        kld_values.append(kld)
                
                # Average KLD to all other conditions
                avg_kld = np.mean(kld_values)
                
                grid_kld_data.append({
                    'train_game': train_game,
                    'train_opponent_range': train_opp,
                    'train_coop_rate': train_coop,
                    'avg_kld_to_others': avg_kld,
                    'max_kld_to_others': np.max(kld_values),
                    'min_kld_to_others': np.min(kld_values)
                })
        
        self.grid_kld_df = pd.DataFrame(grid_kld_data)
        
        # Save to CSV
        csv_path = self.output_dir / 'training_condition_kld_grid.csv'
        self.grid_kld_df.to_csv(csv_path, index=False)
        print(f"✓ Saved grid KLD data: {csv_path}")
        print(f"\nGrid KLD Statistics:")
        print(f"  Min avg KLD: {self.grid_kld_df['avg_kld_to_others'].min():.4f}")
        print(f"  Max avg KLD: {self.grid_kld_df['avg_kld_to_others'].max():.4f}")
        print(f"  Mean avg KLD: {self.grid_kld_df['avg_kld_to_others'].mean():.4f}")
    
    def plot_cross_model_kld_heatmap(self):
        """
        Plot 5×3 heatmap showing average KLD for each training condition.
        Rows: training opponent ranges, Columns: training games.
        """
        print("\n" + "="*80)
        print("PLOTTING CROSS-MODEL KLD HEATMAP (5×3 GRID)")
        print("="*80)
        
        # Create pivot table: opponent_range × game
        pivot = self.grid_kld_df.pivot(
            index='train_opponent_range',
            columns='train_game',
            values='avg_kld_to_others'
        )
        pivot = pivot.reindex(index=self.opp_ranges, columns=self.games)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        vmax = max(1.0, pivot.max().max() * 1.1)
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)
        
        # Add text annotations
        for i in range(len(self.opp_ranges)):
            for j in range(len(self.games)):
                value = pivot.iloc[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                           color='black' if value < vmax * 0.5 else 'white',
                           fontsize=12, fontweight='bold')
        
        # Set labels
        ax.set_xticks(range(len(self.games)))
        ax.set_yticks(range(len(self.opp_ranges)))
        ax.set_xticklabels([self.game_abbrev[g] for g in self.games], fontsize=12)
        ax.set_yticklabels([self.opp_range_labels[o] for o in self.opp_ranges], fontsize=12)
        ax.set_xlabel('Training Game', fontsize=13, fontweight='bold')
        ax.set_ylabel('Training Opponent Range', fontsize=13, fontweight='bold')
        ax.set_title('Average Policy Divergence by Training Condition\n' + 
                     'Mean KL(this condition || all other conditions)',
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average KL Divergence (nats)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'cross_model_kld_heatmap_5x3.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: cross_model_kld_heatmap_5x3.png")
        plt.close()
    
    def plot_training_cooperation_and_kld(self):
        """
        Plot side-by-side 5×3 heatmaps showing training cooperation rates and cross-model KLD.
        """
        print("\n" + "="*80)
        print("PLOTTING COOPERATION RATE & KLD SIDE-BY-SIDE")
        print("="*80)
        
        # Create pivot tables
        coop_pivot = self.grid_kld_df.pivot(
            index='train_opponent_range',
            columns='train_game',
            values='train_coop_rate'
        )
        coop_pivot = coop_pivot.reindex(index=self.opp_ranges, columns=self.games)
        
        kld_pivot = self.grid_kld_df.pivot(
            index='train_opponent_range',
            columns='train_game',
            values='avg_kld_to_others'
        )
        kld_pivot = kld_pivot.reindex(index=self.opp_ranges, columns=self.games)
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Training Policy Analysis: Cooperation Rate & Cross-Model KLD',
                     fontsize=15, fontweight='bold', y=0.98)
        
        # Plot 1: Cooperation rates
        ax = axes[0]
        im1 = ax.imshow(coop_pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        for i in range(len(self.opp_ranges)):
            for j in range(len(self.games)):
                value = coop_pivot.iloc[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                           color='black' if 0.3 < value < 0.7 else 'white',
                           fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(self.games)))
        ax.set_yticks(range(len(self.opp_ranges)))
        ax.set_xticklabels([self.game_abbrev[g] for g in self.games], fontsize=11)
        ax.set_yticklabels([self.opp_range_labels[o] for o in self.opp_ranges], fontsize=11)
        ax.set_xlabel('Training Game', fontsize=12, fontweight='bold')
        ax.set_ylabel('Training Opponent Range', fontsize=12, fontweight='bold')
        ax.set_title('Training Cooperation Rate', fontsize=13, fontweight='bold', pad=10)
        cbar1 = plt.colorbar(im1, ax=ax)
        cbar1.set_label('Cooperation Rate', fontsize=11, fontweight='bold')
        
        # Plot 2: Cross-model KLD
        ax = axes[1]
        vmax = max(1.0, kld_pivot.max().max() * 1.1)
        im2 = ax.imshow(kld_pivot.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)
        for i in range(len(self.opp_ranges)):
            for j in range(len(self.games)):
                value = kld_pivot.iloc[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                           color='black' if value < vmax * 0.5 else 'white',
                           fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(self.games)))
        ax.set_yticks(range(len(self.opp_ranges)))
        ax.set_xticklabels([self.game_abbrev[g] for g in self.games], fontsize=11)
        ax.set_yticklabels([self.opp_range_labels[o] for o in self.opp_ranges], fontsize=11)
        ax.set_xlabel('Training Game', fontsize=12, fontweight='bold')
        ax.set_ylabel('Training Opponent Range', fontsize=12, fontweight='bold')
        ax.set_title('Avg KLD to Other Conditions', fontsize=13, fontweight='bold', pad=10)
        cbar2 = plt.colorbar(im2, ax=ax)
        cbar2.set_label('KL Divergence (nats)', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'cooperation_and_kld_comparison.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: cooperation_and_kld_comparison.png")
        plt.close()
    
    def plot_cross_model_kld_game_pairs(self):
        """
        Plot 3×3 heatmap of average KLD between game pairs.
        Shows KLD(train_game_1 || train_game_2) averaged across opponents.
        """
        print("\n" + "="*80)
        print("PLOTTING GAME-PAIR KLD HEATMAP (3×3)")
        print("="*80)
        
        # Average KLD across all test conditions for each train game pair
        avg_kld = self.cross_kld_df.groupby(['train_game_1', 'train_game_2'])['kld'].mean().reset_index()
        
        # Create pivot table: train_game_1 × train_game_2
        pivot = avg_kld.pivot(index='train_game_1', columns='train_game_2', values='kld')
        pivot = pivot.reindex(index=self.games, columns=self.games)
        
        # Fill diagonal with 0 (KLD(P||P) = 0)
        for game in self.games:
            pivot.loc[game, game] = 0.0
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        vmax = max(0.5, pivot.max().max())
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)
        
        # Add text annotations
        for i in range(len(self.games)):
            for j in range(len(self.games)):
                value = pivot.iloc[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                           color='black' if value < vmax * 0.5 else 'white',
                           fontsize=12, fontweight='bold')
        
        # Set labels
        ax.set_xticks(range(len(self.games)))
        ax.set_yticks(range(len(self.games)))
        ax.set_xticklabels([self.game_abbrev[g] for g in self.games], fontsize=12)
        ax.set_yticklabels([self.game_abbrev[g] for g in self.games], fontsize=12)
        ax.set_xlabel('Training Game 2 (Q)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Training Game 1 (P)', fontsize=13, fontweight='bold')
        ax.set_title('Cross-Game KLD: KL(P||Q)\n' + 
                     'Averaged across opponent ranges',
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('KL Divergence (nats)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'cross_game_kld_3x3.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: cross_game_kld_3x3.png")
        plt.close()
    
    def plot_cross_model_kld_by_test_game(self):
        """
        Plot cross-model KLD broken down by test game.
        Shows how model divergence depends on what game they're tested on.
        """
        print("\n" + "="*80)
        print("PLOTTING CROSS-MODEL KLD BY TEST GAME")
        print("="*80)
        
        # Create figure with 3 subplots (one per test game)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Cross-Model KLD When Tested on Different Games\n' + 
                     'KL(P||Q) for models trained on different games',
                     fontsize=14, fontweight='bold', y=1.02)
        
        for idx, test_game in enumerate(self.games):
            ax = axes[idx]
            
            # Filter data for this test game
            test_data = self.cross_kld_df[self.cross_kld_df['test_game'] == test_game]
            
            if len(test_data) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{self.game_abbrev[test_game]}', fontsize=13, fontweight='bold')
                continue
            
            # Average across test opponent ranges
            avg_kld = test_data.groupby(['train_game_1', 'train_game_2'])['kld'].mean().reset_index()
            
            # Create pivot
            pivot = avg_kld.pivot(index='train_game_1', columns='train_game_2', values='kld')
            pivot = pivot.reindex(index=self.games, columns=self.games)
            
            # Fill diagonal
            for game in self.games:
                pivot.loc[game, game] = 0.0
            
            # Plot heatmap
            vmax = max(0.3, pivot.max().max())
            im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)
            
            # Add text
            for i in range(len(self.games)):
                for j in range(len(self.games)):
                    value = pivot.iloc[i, j]
                    if not np.isnan(value):
                        ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                               color='black' if value < vmax * 0.5 else 'white',
                               fontsize=10, fontweight='bold')
            
            # Labels
            ax.set_xticks(range(len(self.games)))
            ax.set_yticks(range(len(self.games)))
            ax.set_xticklabels([self.game_abbrev[g] for g in self.games], fontsize=10)
            if idx == 0:
                ax.set_yticklabels([self.game_abbrev[g] for g in self.games], fontsize=10)
                ax.set_ylabel('Train Game 1 (P)', fontsize=11, fontweight='bold')
            else:
                ax.set_yticklabels([])
            ax.set_xlabel('Train Game 2 (Q)', fontsize=11, fontweight='bold')
            ax.set_title(f'Test: {self.game_abbrev[test_game]}', fontsize=12, fontweight='bold')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            if idx == 2:
                cbar.set_label('KLD (nats)', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'cross_model_kld_by_test_game.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: cross_model_kld_by_test_game.png")
        plt.close()
    
    def plot_cross_model_kld_game_pairs(self):
        """
        Plot 3×3 heatmap of average KLD between game pairs.
        Shows KLD(train_game_1 || train_game_2) averaged across opponents.
        """
        print("\n" + "="*80)
        print("PLOTTING GAME-PAIR KLD HEATMAP (3×3)")
        print("="*80)
        
        # Average KLD across all test conditions for each train game pair
        avg_kld = self.cross_kld_df.groupby(['train_game_1', 'train_game_2'])['kld'].mean().reset_index()
        
        # Create pivot table: train_game_1 × train_game_2
        pivot = avg_kld.pivot(index='train_game_1', columns='train_game_2', values='kld')
        pivot = pivot.reindex(index=self.games, columns=self.games)
        
        # Fill diagonal with 0 (KLD(P||P) = 0)
        for game in self.games:
            pivot.loc[game, game] = 0.0
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        vmax = max(0.5, pivot.max().max())
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)
        
        # Add text annotations
        for i in range(len(self.games)):
            for j in range(len(self.games)):
                value = pivot.iloc[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                           color='black' if value < vmax * 0.5 else 'white',
                           fontsize=12, fontweight='bold')
        
        # Set labels
        ax.set_xticks(range(len(self.games)))
        ax.set_yticks(range(len(self.games)))
        ax.set_xticklabels([self.game_abbrev[g] for g in self.games], fontsize=12)
        ax.set_yticklabels([self.game_abbrev[g] for g in self.games], fontsize=12)
        ax.set_xlabel('Training Game 2 (Q)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Training Game 1 (P)', fontsize=13, fontweight='bold')
        ax.set_title('Cross-Game KLD: KL(P||Q)\n' + 
                     'Averaged across opponent ranges',
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('KL Divergence (nats)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'cross_game_kld_3x3.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: cross_game_kld_3x3.png")
        plt.close()
    
    def run_analysis(self):
        """Run complete cross-model KLD analysis."""
        print("\n" + "="*80)
        print("CROSS-MODEL KLD ANALYSIS")
        print("="*80)
        print(f"Input CSV: {self.kld_csv_path}")
        print(f"Output Dir: {self.output_dir}")
        
        self.compute_cross_model_kld()
        self.compute_training_condition_kld_grid()
        self.plot_cross_model_kld_heatmap()
        self.plot_training_cooperation_and_kld()
        self.plot_cross_model_kld_game_pairs()
        self.plot_cross_model_kld_by_test_game()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Figures saved to: {self.fig_dir}")

def main():
    kld_csv = "experiments/generalization_analysis/data/policy_kld_analysis.csv"
    output_dir = "experiments/cross_model_kld_analysis"
    
    analyzer = CrossModelKLDAnalyzer(kld_csv, output_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
