#!/usr/bin/env python3
"""
Comprehensive Training Dynamics Analysis
Job: 888509 (Battle-of-Sexes excluded from experiment)

Generates 3 plots:
1. Training loss by opponent type per game (3 subplots: PD/SH/HD)
2. Cooperation rate by opponent type per game (3 subplots: PD/SH/HD)
3. Policy entropy by opponent type per game (3 subplots: PD/SH/HD)
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

class TrainingDynamicsAnalyzer:
    """Training dynamics analysis with opponent-specific learning curves."""
    
    def __init__(self, training_dir, output_dir):
        self.training_dir = Path(training_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.fig_dir = self.output_dir / 'figures'
        self.data_dir = self.output_dir / 'data'
        
        for dir_path in [self.fig_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Game mappings (battle-of-sexes removed from experiment)
        self.games = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
        self.game_abbrev = {
            'prisoners-dilemma': 'PD',
            'stag-hunt': 'SH',
            'hawk-dove': 'HD'
        }
        
        self.opp_ranges = ['very_low', 'low', 'mid', 'high', 'very_high']
        self.opp_colors = {
            'very_low': '#2E7D32',    # Dark green
            'low': '#66BB6A',          # Light green
            'mid': '#FFA726',          # Orange
            'high': '#EF5350',         # Light red
            'very_high': '#C62828'     # Dark red
        }
        
        # Data storage
        self.training_data = {}
    
    def load_all_training_data(self):
        """Load all training results with epoch-level detail."""
        print("="*80)
        print("LOADING TRAINING DATA WITH EPOCH-LEVEL METRICS")
        print("="*80)
        
        condition_dirs = sorted([d for d in self.training_dir.iterdir() 
                                if d.is_dir() and d.name.startswith('condition_')])
        
        for cond_dir in condition_dirs:
            # Extract condition and seed
            parts = cond_dir.name.split('_')
            try:
                cond_id = int(parts[1])
                seed_id = int(parts[3])
            except (IndexError, ValueError):
                continue  # Skip directories that don't match pattern
            model_id = cond_id * 5 + seed_id
            
            # Find experiment directory
            exp_dirs = list(cond_dir.glob("generalization_matrix_task_*"))
            if not exp_dirs:
                continue
            
            # Load training results
            pkl_files = list((exp_dirs[0] / "results").glob("training_task_*_results.pkl"))
            if not pkl_files:
                continue
            
            try:
                with open(pkl_files[0], 'rb') as f:
                    data = pickle.load(f)
                
                # Extract training condition info
                train_cond = data.get('training_condition', {})
                training_results = data.get('training_results', {})
                
                self.training_data[model_id] = {
                    'condition': cond_dir.name,
                    'cond_id': cond_id,
                    'seed_id': seed_id,
                    'game': train_cond.get('game', 'unknown'),
                    'opponent_range': train_cond.get('opponent_range', 'unknown'),
                    'epoch_results': training_results.get('epoch_results', [])
                }
                
            except Exception as e:
                print(f"  ⚠ Error loading {cond_dir.name}: {e}")
                continue
        
        print(f"\n✓ Loaded {len(self.training_data)} training models")
        
        # Verify data structure
        if self.training_data:
            sample_model = next(iter(self.training_data.values()))
            print(f"  Sample model has {len(sample_model['epoch_results'])} epochs")
            if sample_model['epoch_results']:
                print(f"  Epoch metrics: {list(sample_model['epoch_results'][0].keys())[:5]}...")
    
    def plot_training_loss_by_opponent(self, normalized=False):
        """Plot 1: Training loss by opponent type per game with CIs.
        
        Args:
            normalized (bool): If True, normalize losses within each game to [0, 1] range
        """
        print("\n" + "="*80)
        print(f"PLOT 1: {'NORMALIZED ' if normalized else ''}TRAINING LOSS BY OPPONENT TYPE")
        print("="*80)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        if normalized:
            fig.suptitle('Normalized Training Loss by Opponent Type (Mean ± 95% CI across 5 seeds)', 
                        fontsize=14, y=1.02)
        else:
            fig.suptitle('Training Loss by Opponent Type (Mean ± 95% CI across 5 seeds)', 
                        fontsize=14, y=1.02)
        
        # First pass: collect all losses per game to determine normalization params
        game_losses = {game: [] for game in self.games}
        
        if normalized:
            for game in self.games:
                for opp_range in self.opp_ranges:
                    models = [m for m in self.training_data.values() 
                             if m['game'] == game and m['opponent_range'] == opp_range]
                    for model in models:
                        losses = [e.get('total_loss', np.nan) for e in model['epoch_results']]
                        game_losses[game].extend([l for l in losses if not np.isnan(l)])
            
            # Compute min/max per game for normalization
            game_min_max = {}
            for game in self.games:
                if game_losses[game]:
                    game_min_max[game] = {
                        'min': np.min(game_losses[game]),
                        'max': np.max(game_losses[game])
                    }
                    print(f"  {self.game_abbrev[game]} loss range: [{game_min_max[game]['min']:.3f}, {game_min_max[game]['max']:.3f}]")
                else:
                    game_min_max[game] = {'min': 0, 'max': 1}
        
        # Second pass: plot with normalization
        for game_idx, game in enumerate(self.games):
            ax = axes[game_idx]
            
            # Group by opponent range for this game
            for opp_range in self.opp_ranges:
                # Get all models for this game + opponent range (5 seeds)
                models = [m for m in self.training_data.values() 
                         if m['game'] == game and m['opponent_range'] == opp_range]
                
                if not models:
                    continue
                
                # Collect loss trajectories across seeds
                loss_trajectories = []
                for model in models:
                    losses = [e.get('total_loss', np.nan) for e in model['epoch_results']]
                    
                    # Normalize if requested
                    if normalized:
                        min_val = game_min_max[game]['min']
                        max_val = game_min_max[game]['max']
                        if max_val > min_val:
                            losses = [(l - min_val) / (max_val - min_val) if not np.isnan(l) else np.nan 
                                     for l in losses]
                        else:
                            losses = [0.0 if not np.isnan(l) else np.nan for l in losses]
                    
                    loss_trajectories.append(losses)
                
                if not loss_trajectories:
                    continue
                
                # Convert to array for statistics
                loss_array = np.array(loss_trajectories)  # Shape: (n_seeds, n_epochs)
                epochs = range(loss_array.shape[1])
                
                # Compute mean and 95% CI
                mean_loss = np.nanmean(loss_array, axis=0)
                std_loss = np.nanstd(loss_array, axis=0)
                n_seeds = loss_array.shape[0]
                ci_95 = 1.96 * std_loss / np.sqrt(n_seeds)
                
                # Plot with CI
                color = self.opp_colors[opp_range]
                ax.plot(epochs, mean_loss, color=color, linewidth=2, 
                       label=opp_range.replace('_', ' ').title())
                ax.fill_between(epochs, mean_loss - ci_95, mean_loss + ci_95, 
                               color=color, alpha=0.2)
            
            ax.set_xlabel('Epoch', fontsize=12)
            if normalized:
                ax.set_ylabel('Normalized Loss [0-1]', fontsize=12)
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.set_ylabel('Training Loss', fontsize=12)
            ax.set_title(f'{self.game_abbrev[game]}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')
        
        plt.tight_layout()
        
        if normalized:
            plt.savefig(self.fig_dir / 'plot1_normalized_training_loss_by_opponent.png', dpi=150, bbox_inches='tight')
            print(f"✓ Saved: plot1_normalized_training_loss_by_opponent.png")
        else:
            plt.savefig(self.fig_dir / 'plot1_training_loss_by_opponent.png', dpi=150, bbox_inches='tight')
            print(f"✓ Saved: plot1_training_loss_by_opponent.png")
        plt.close()
    
    def plot_training_reward_by_opponent(self, normalized=False):
        """Plot 1b: Training reward by opponent type per game with CIs.
        
        Args:
            normalized (bool): If True, normalize rewards within each game to [0, 1] range
        """
        print("\n" + "="*80)
        print(f"PLOT 1b: {'NORMALIZED ' if normalized else ''}TRAINING REWARD BY OPPONENT TYPE")
        print("="*80)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        if normalized:
            fig.suptitle('Normalized Training Reward by Opponent Type (Mean ± 95% CI across 5 seeds)', 
                        fontsize=14, y=1.02)
        else:
            fig.suptitle('Training Reward by Opponent Type (Mean ± 95% CI across 5 seeds)', 
                        fontsize=14, y=1.02)
        
        # First pass: collect all rewards per game to determine normalization params
        game_rewards = {game: [] for game in self.games}
        
        if normalized:
            for game in self.games:
                for opp_range in self.opp_ranges:
                    models = [m for m in self.training_data.values() 
                             if m['game'] == game and m['opponent_range'] == opp_range]
                    for model in models:
                        rewards = [e.get('epoch_cumulative_reward', np.nan) for e in model['epoch_results']]
                        game_rewards[game].extend([r for r in rewards if not np.isnan(r)])
            
            # Compute min/max per game for normalization
            game_min_max = {}
            for game in self.games:
                if game_rewards[game]:
                    game_min_max[game] = {
                        'min': np.min(game_rewards[game]),
                        'max': np.max(game_rewards[game])
                    }
                    print(f"  {self.game_abbrev[game]} reward range: [{game_min_max[game]['min']:.3f}, {game_min_max[game]['max']:.3f}]")
                else:
                    game_min_max[game] = {'min': 0, 'max': 1}
        
        # Second pass: plot with normalization
        for game_idx, game in enumerate(self.games):
            ax = axes[game_idx]
            
            # Group by opponent range for this game
            for opp_range in self.opp_ranges:
                # Get all models for this game + opponent range (5 seeds)
                models = [m for m in self.training_data.values() 
                         if m['game'] == game and m['opponent_range'] == opp_range]
                
                if not models:
                    continue
                
                # Collect reward trajectories across seeds
                reward_trajectories = []
                for model in models:
                    rewards = [e.get('epoch_cumulative_reward', np.nan) for e in model['epoch_results']]
                    
                    # Normalize if requested
                    if normalized:
                        min_val = game_min_max[game]['min']
                        max_val = game_min_max[game]['max']
                        if max_val > min_val:
                            rewards = [(r - min_val) / (max_val - min_val) if not np.isnan(r) else np.nan 
                                     for r in rewards]
                        else:
                            rewards = [0.5 if not np.isnan(r) else np.nan for r in rewards]
                    
                    reward_trajectories.append(rewards)
                
                if not reward_trajectories:
                    continue
                
                # Convert to array for statistics
                reward_array = np.array(reward_trajectories)  # Shape: (n_seeds, n_epochs)
                epochs = range(reward_array.shape[1])
                
                # Compute mean and 95% CI
                mean_reward = np.nanmean(reward_array, axis=0)
                std_reward = np.nanstd(reward_array, axis=0)
                n_seeds = reward_array.shape[0]
                ci_95 = 1.96 * std_reward / np.sqrt(n_seeds)
                
                # Plot with CI
                color = self.opp_colors[opp_range]
                ax.plot(epochs, mean_reward, color=color, linewidth=2, 
                       label=opp_range.replace('_', ' ').title())
                ax.fill_between(epochs, mean_reward - ci_95, mean_reward + ci_95, 
                               color=color, alpha=0.2)
            
            ax.set_xlabel('Epoch', fontsize=12)
            if normalized:
                ax.set_ylabel('Normalized Reward [0-1]', fontsize=12)
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.set_ylabel('Cumulative Reward per Epoch', fontsize=12)
            ax.set_title(f'{self.game_abbrev[game]}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')
        
        plt.tight_layout()
        
        if normalized:
            plt.savefig(self.fig_dir / 'plot1b_normalized_training_reward_by_opponent.png', dpi=150, bbox_inches='tight')
            print(f"✓ Saved: plot1b_normalized_training_reward_by_opponent.png")
        else:
            plt.savefig(self.fig_dir / 'plot1b_training_reward_by_opponent.png', dpi=150, bbox_inches='tight')
            print(f"✓ Saved: plot1b_training_reward_by_opponent.png")
        plt.close()
    
    def plot_cooperation_rate_by_opponent(self):
        """Plot 2: Convergence to optimal policy (cooperation rate) by opponent type per game."""
        print("\n" + "="*80)
        print("PLOT 2: COOPERATION RATE BY OPPONENT TYPE")
        print("="*80)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Cooperation Rate by Opponent Type (Mean ± 95% CI across 5 seeds)', 
                     fontsize=14, y=1.02)
        
        for game_idx, game in enumerate(self.games):
            ax = axes[game_idx]
            
            # Group by opponent range for this game
            for opp_range in self.opp_ranges:
                # Get all models for this game + opponent range (5 seeds)
                models = [m for m in self.training_data.values() 
                         if m['game'] == game and m['opponent_range'] == opp_range]
                
                if not models:
                    continue
                
                # Collect cooperation rate trajectories across seeds
                coop_trajectories = []
                for model in models:
                    coop_rates = [e.get('epoch_average_cooperation_rate', np.nan) 
                                 for e in model['epoch_results']]
                    coop_trajectories.append(coop_rates)
                
                if not coop_trajectories:
                    continue
                
                # Convert to array for statistics
                coop_array = np.array(coop_trajectories)  # Shape: (n_seeds, n_epochs)
                epochs = range(coop_array.shape[1])
                
                # Compute mean and 95% CI
                mean_coop = np.nanmean(coop_array, axis=0)
                std_coop = np.nanstd(coop_array, axis=0)
                n_seeds = coop_array.shape[0]
                ci_95 = 1.96 * std_coop / np.sqrt(n_seeds)
                
                # Plot with CI
                color = self.opp_colors[opp_range]
                ax.plot(epochs, mean_coop, color=color, linewidth=2, 
                       label=opp_range.replace('_', ' ').title())
                ax.fill_between(epochs, mean_coop - ci_95, mean_coop + ci_95, 
                               color=color, alpha=0.2)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Cooperation Rate', fontsize=12)
            ax.set_title(f'{self.game_abbrev[game]}', fontsize=13, fontweight='bold')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'plot2_cooperation_rate_by_opponent.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: plot2_cooperation_rate_by_opponent.png")
        plt.close()
    
    def plot_policy_entropy_by_opponent(self):
        """Plot 3: Policy entropy by opponent type per game."""
        print("\n" + "="*80)
        print("PLOT 3: POLICY ENTROPY BY OPPONENT TYPE")
        print("="*80)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Policy Entropy by Opponent Type (Mean ± 95% CI across 5 seeds)', 
                     fontsize=14, y=1.02)
        
        for game_idx, game in enumerate(self.games):
            ax = axes[game_idx]
            
            # Group by opponent range for this game
            for opp_range in self.opp_ranges:
                # Get all models for this game + opponent range (5 seeds)
                models = [m for m in self.training_data.values() 
                         if m['game'] == game and m['opponent_range'] == opp_range]
                
                if not models:
                    continue
                
                # Collect policy entropy trajectories across seeds
                entropy_trajectories = []
                for model in models:
                    # Compute entropy from cooperation rate
                    coop_rates = [e.get('epoch_average_cooperation_rate', np.nan) 
                                 for e in model['epoch_results']]
                    
                    # Binary entropy: H = -p*log(p) - (1-p)*log(1-p)
                    entropies = []
                    for p in coop_rates:
                        if np.isnan(p) or p <= 0 or p >= 1:
                            # Handle edge cases
                            if p == 0 or p == 1:
                                entropies.append(0.0)
                            else:
                                entropies.append(np.nan)
                        else:
                            entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
                            entropies.append(entropy)
                    
                    entropy_trajectories.append(entropies)
                
                if not entropy_trajectories:
                    continue
                
                # Convert to array for statistics
                entropy_array = np.array(entropy_trajectories)  # Shape: (n_seeds, n_epochs)
                epochs = range(entropy_array.shape[1])
                
                # Compute mean and 95% CI
                mean_entropy = np.nanmean(entropy_array, axis=0)
                std_entropy = np.nanstd(entropy_array, axis=0)
                n_seeds = entropy_array.shape[0]
                ci_95 = 1.96 * std_entropy / np.sqrt(n_seeds)
                
                # Plot with CI
                color = self.opp_colors[opp_range]
                ax.plot(epochs, mean_entropy, color=color, linewidth=2, 
                       label=opp_range.replace('_', ' ').title())
                ax.fill_between(epochs, mean_entropy - ci_95, mean_entropy + ci_95, 
                               color=color, alpha=0.2)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Policy Entropy (bits)', fontsize=12)
            ax.set_title(f'{self.game_abbrev[game]}', fontsize=13, fontweight='bold')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'plot3_policy_entropy_by_opponent.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: plot3_policy_entropy_by_opponent.png")
        plt.close()
    
    def run_analysis(self):
        """Run complete training analysis."""
        print("\n" + "="*80)
        print("TRAINING DYNAMICS ANALYSIS - JOB 888509")
        print("="*80)
        print(f"Training Dir: {self.training_dir}")
        print(f"Output Dir: {self.output_dir}")
        
        # Load data
        self.load_all_training_data()
        
        # Generate requested plots
        self.plot_training_loss_by_opponent(normalized=False)  # Original scale
        self.plot_training_loss_by_opponent(normalized=True)   # Normalized [0-1]
        self.plot_training_reward_by_opponent(normalized=False)  # Original scale
        self.plot_training_reward_by_opponent(normalized=True)   # Normalized [0-1]
        self.plot_cooperation_rate_by_opponent()
        self.plot_policy_entropy_by_opponent()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Figures saved to: {self.fig_dir}")

def main():
    training_dir = "experiments/generalization_matrix_train_888509/training"
    output_dir = "experiments/training_analysis_888509"
    
    analyzer = TrainingDynamicsAnalyzer(training_dir, output_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
