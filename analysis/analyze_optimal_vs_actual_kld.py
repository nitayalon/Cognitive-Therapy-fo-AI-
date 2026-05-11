"""
Compute KLD between optimal policy (per setup) and actual learned policy (per training condition).

For each training condition, create a 5×3 heatmap showing:
- KLD(optimal_policy_for_test_setup || actual_policy_from_training_setup)
- Optimal policy = average cooperation rate of models trained on that test setup
- Actual policy = cooperation rate of this training condition when tested on that setup
- By definition, KLD = 0 when training setup = test setup
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class OptimalVsActualKLDAnalyzer:
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
            'very_low': 'VL',
            'low': 'L',
            'mid': 'M',
            'high': 'H',
            'very_high': 'VH'
        }
    
    def binary_kl_divergence(self, p, q, epsilon=0.001):
        """Binary KL divergence: KL(P||Q) for cooperation rates."""
        p = np.clip(p, epsilon, 1 - epsilon)
        q = np.clip(q, epsilon, 1 - epsilon)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    
    def compute_optimal_policies(self):
        """
        Compute optimal policy for each setup.
        Optimal = average cooperation rate of models trained on that setup.
        """
        print("\n" + "="*80)
        print("COMPUTING OPTIMAL POLICIES PER SETUP")
        print("="*80)
        
        optimal_policies = {}
        
        for game in self.games:
            for opp_range in self.opp_ranges:
                # Get average training cooperation rate for this setup
                setup_data = self.df[
                    (self.df['train_game'] == game) &
                    (self.df['train_opponent_range'] == opp_range)
                ]
                
                if len(setup_data) > 0:
                    optimal_coop = setup_data['train_coop_rate'].mean()
                    optimal_policies[(game, opp_range)] = optimal_coop
                    print(f"  {self.game_abbrev[game]}-{self.opp_range_labels[opp_range]}: "
                          f"optimal coop = {optimal_coop:.3f}")
        
        self.optimal_policies = optimal_policies
        print(f"\n✓ Computed {len(optimal_policies)} optimal policies")
    
    def compute_optimal_vs_actual_kld(self):
        """
        For each training condition, compute KLD to optimal policy for each test condition.
        """
        print("\n" + "="*80)
        print("COMPUTING OPTIMAL VS ACTUAL KLD")
        print("="*80)
        
        kld_data = []
        
        # For each training condition
        for train_game in self.games:
            for train_opp in self.opp_ranges:
                print(f"\nProcessing: {self.game_abbrev[train_game]}-{self.opp_range_labels[train_opp]}")
                
                # Get this training condition's performance on all test setups
                train_cond_data = self.df[
                    (self.df['train_game'] == train_game) &
                    (self.df['train_opponent_range'] == train_opp)
                ]
                
                # For each test condition
                for test_game in self.games:
                    for test_opp in self.opp_ranges:
                        # Get optimal policy for this test setup
                        optimal_coop = self.optimal_policies.get((test_game, test_opp), np.nan)
                        
                        # Get actual policy: what this training condition does on this test setup
                        test_data = train_cond_data[
                            (train_cond_data['test_game'] == test_game) &
                            (train_cond_data['test_opponent_range'] == test_opp)
                        ]
                        
                        if len(test_data) > 0:
                            actual_coop = test_data['test_coop_rate'].mean()
                            
                            # Compute KLD(optimal || actual)
                            kld = self.binary_kl_divergence(optimal_coop, actual_coop)
                            
                            kld_data.append({
                                'train_game': train_game,
                                'train_opponent_range': train_opp,
                                'test_game': test_game,
                                'test_opponent_range': test_opp,
                                'optimal_coop': optimal_coop,
                                'actual_coop': actual_coop,
                                'kld': kld,
                                'is_same_setup': (train_game == test_game and train_opp == test_opp)
                            })
        
        self.kld_df = pd.DataFrame(kld_data)
        
        # Save to CSV
        csv_path = self.output_dir / 'optimal_vs_actual_kld.csv'
        self.kld_df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved KLD data: {csv_path}")
        print(f"✓ Total comparisons: {len(self.kld_df)}")
        
        # Statistics
        print(f"\nKLD Statistics:")
        print(f"  Min: {self.kld_df['kld'].min():.6f}")
        print(f"  Max: {self.kld_df['kld'].max():.6f}")
        print(f"  Mean: {self.kld_df['kld'].mean():.6f}")
        print(f"  Median: {self.kld_df['kld'].median():.6f}")
        
        # Check diagonal (should be ~0)
        diagonal = self.kld_df[self.kld_df['is_same_setup']]
        print(f"\nDiagonal (same setup) KLD:")
        print(f"  Mean: {diagonal['kld'].mean():.6f}")
        print(f"  Max: {diagonal['kld'].max():.6f}")
    
    def plot_optimal_vs_actual_kld_grid(self):
        """
        Create 15 subplots (5 rows × 3 cols for training conditions).
        Each subplot shows a 5×3 heatmap of KLD values for test conditions.
        """
        print("\n" + "="*80)
        print("PLOTTING OPTIMAL VS ACTUAL KLD GRID (15 SUBPLOTS)")
        print("="*80)
        
        # Create figure with 5 rows × 3 cols
        fig, axes = plt.subplots(5, 3, figsize=(15, 20))
        fig.suptitle('KL Divergence: Optimal Policy vs Actual Policy\n' + 
                     'KL(optimal_for_test || actual_from_training)\n' +
                     'Each subplot = one training condition | Each cell = one test condition',
                     fontsize=14, fontweight='bold', y=0.995)
        
        # Compute global vmax for consistent color scale
        max_kld = self.kld_df['kld'].max()
        vmax = max(0.5, max_kld)
        
        # For each training condition
        for row_idx, train_opp in enumerate(self.opp_ranges):
            for col_idx, train_game in enumerate(self.games):
                ax = axes[row_idx, col_idx]
                
                # Filter data for this training condition
                train_data = self.kld_df[
                    (self.kld_df['train_game'] == train_game) &
                    (self.kld_df['train_opponent_range'] == train_opp)
                ]
                
                if len(train_data) == 0:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                           transform=ax.transAxes, fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                # Create pivot table: test_opponent_range × test_game
                pivot = train_data.pivot_table(
                    values='kld',
                    index='test_opponent_range',
                    columns='test_game',
                    aggfunc='mean'
                )
                pivot = pivot.reindex(index=self.opp_ranges, columns=self.games)
                
                # Plot heatmap
                im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto',
                              vmin=0, vmax=vmax, interpolation='nearest')
                
                # Add text annotations
                for i in range(len(self.opp_ranges)):
                    for j in range(len(self.games)):
                        value = pivot.iloc[i, j]
                        if not np.isnan(value):
                            # Highlight diagonal (same setup) with different format
                            is_diagonal = (self.opp_ranges[i] == train_opp and 
                                         self.games[j] == train_game)
                            
                            text_color = 'black' if value < vmax * 0.5 else 'white'
                            text = f'{value:.3f}' if value >= 0.01 else f'{value:.4f}'
                            
                            if is_diagonal:
                                # Bold and boxed for diagonal
                                ax.add_patch(plt.Rectangle((j-0.4, i-0.4), 0.8, 0.8,
                                           fill=False, edgecolor='blue', linewidth=2))
                            
                            ax.text(j, i, text, ha='center', va='center',
                                   color=text_color, fontsize=7,
                                   fontweight='bold' if is_diagonal else 'normal')
                
                # Set title and labels
                if row_idx == 0:
                    ax.set_title(f'{self.game_abbrev[train_game]}', 
                               fontsize=11, fontweight='bold')
                
                if col_idx == 0:
                    ax.set_ylabel(f'{self.opp_range_labels[train_opp]}', 
                                 fontsize=10, fontweight='bold')
                    ax.set_yticks(range(len(self.opp_ranges)))
                    ax.set_yticklabels([self.opp_range_labels[o] for o in self.opp_ranges],
                                      fontsize=7)
                else:
                    ax.set_yticks([])
                
                if row_idx == 4:
                    ax.set_xticks(range(len(self.games)))
                    ax.set_xticklabels([self.game_abbrev[g] for g in self.games],
                                      fontsize=7)
                else:
                    ax.set_xticks([])
        
        # Add single colorbar for all subplots
        cbar = fig.colorbar(im, ax=axes, orientation='horizontal',
                           pad=0.02, aspect=40, shrink=0.8)
        cbar.set_label('KL Divergence (nats): KL(optimal || actual)', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'optimal_vs_actual_kld_15_subplots.png',
                   dpi=150, bbox_inches='tight')
        print(f"✓ Saved: optimal_vs_actual_kld_15_subplots.png")
        plt.close()
    
    def plot_summary_heatmap(self):
        """
        Create single 5×3 heatmap showing average KLD across all test conditions.
        """
        print("\n" + "="*80)
        print("PLOTTING SUMMARY HEATMAP")
        print("="*80)
        
        # Compute average KLD for each training condition (excluding diagonal)
        summary_data = []
        
        for train_game in self.games:
            for train_opp in self.opp_ranges:
                train_data = self.kld_df[
                    (self.kld_df['train_game'] == train_game) &
                    (self.kld_df['train_opponent_range'] == train_opp) &
                    (~self.kld_df['is_same_setup'])  # Exclude diagonal
                ]
                
                if len(train_data) > 0:
                    avg_kld = train_data['kld'].mean()
                    max_kld = train_data['kld'].max()
                    
                    summary_data.append({
                        'train_game': train_game,
                        'train_opponent_range': train_opp,
                        'avg_kld_to_others': avg_kld,
                        'max_kld_to_others': max_kld
                    })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create pivot
        pivot = summary_df.pivot(
            index='train_opponent_range',
            columns='train_game',
            values='avg_kld_to_others'
        )
        pivot = pivot.reindex(index=self.opp_ranges, columns=self.games)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        vmax = max(1.0, pivot.max().max())
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)
        
        # Add text
        for i in range(len(self.opp_ranges)):
            for j in range(len(self.games)):
                value = pivot.iloc[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                           color='black' if value < vmax * 0.5 else 'white',
                           fontsize=12, fontweight='bold')
        
        # Labels
        ax.set_xticks(range(len(self.games)))
        ax.set_yticks(range(len(self.opp_ranges)))
        ax.set_xticklabels([self.game_abbrev[g] for g in self.games], fontsize=12)
        ax.set_yticklabels([self.opp_range_labels[o] for o in self.opp_ranges], fontsize=12)
        ax.set_xlabel('Training Game', fontsize=13, fontweight='bold')
        ax.set_ylabel('Training Opponent Range', fontsize=13, fontweight='bold')
        ax.set_title('Average Policy Divergence from Optimal\n' +
                     'Mean KL(optimal || actual) across all test conditions',
                     fontsize=14, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average KL Divergence (nats)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'optimal_vs_actual_summary.png',
                   dpi=150, bbox_inches='tight')
        print(f"✓ Saved: optimal_vs_actual_summary.png")
        plt.close()
    
    def run_analysis(self):
        """Run complete optimal vs actual KLD analysis."""
        print("\n" + "="*80)
        print("OPTIMAL VS ACTUAL KLD ANALYSIS")
        print("="*80)
        print(f"Input CSV: {self.kld_csv_path}")
        print(f"Output Dir: {self.output_dir}")
        
        self.compute_optimal_policies()
        self.compute_optimal_vs_actual_kld()
        self.plot_optimal_vs_actual_kld_grid()
        self.plot_summary_heatmap()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Figures saved to: {self.fig_dir}")

def main():
    kld_csv = "experiments/generalization_analysis/data/policy_kld_analysis.csv"
    output_dir = "experiments/optimal_vs_actual_kld_analysis"
    
    analyzer = OptimalVsActualKLDAnalyzer(kld_csv, output_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
