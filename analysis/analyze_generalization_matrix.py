"""
Comprehensive Analysis of Generalization Matrix Experiment
Analyzes results from experiments/generalization_matrix_834222/
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class GeneralizationMatrixAnalyzer:
    """Comprehensive analyzer for generalization matrix experiment"""
    
    def __init__(self, experiment_dir, output_dir):
        self.experiment_dir = Path(experiment_dir)
        self.output_dir = Path(output_dir)
        self.tasks_data = {}
        self.games = ['prisoners-dilemma', 'stag-hunt', 'battle-of-sexes', 'hawk-dove']
        self.game_abbrev = {'prisoners-dilemma': 'PD', 'stag-hunt': 'SH', 
                           'battle-of-sexes': 'BoS', 'hawk-dove': 'HD'}
        
        # Create output directories
        self.fig_dir = self.output_dir / 'figures'
        self.table_dir = self.output_dir / 'tables'
        self.data_dir = self.output_dir / 'data'
        
        for dir_path in [self.fig_dir, self.table_dir, self.data_dir,
                        self.fig_dir / 'training', self.fig_dir / 'generalization',
                        self.fig_dir / 'tom_analysis', self.fig_dir / 'policy_adaptation',
                        self.fig_dir / 'statistical']:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_all_tasks(self):
        """Load data from all 16 tasks"""
        print("Loading data from all tasks...")
        task_dirs = sorted(self.experiment_dir.glob('generalization_matrix_task_*'))
        
        for task_dir in task_dirs:
            task_id = int(task_dir.name.split('_')[3])
            results_file = task_dir / 'results' / f'task_{task_id}_results.pkl'
            
            if results_file.exists():
                with open(results_file, 'rb') as f:
                    self.tasks_data[task_id] = pickle.load(f)
                print(f"  Loaded task {task_id}")
            else:
                print(f"  Warning: Missing results for task {task_id}")
        
        print(f"Successfully loaded {len(self.tasks_data)} tasks\n")
        return self.tasks_data
    
    def extract_training_metrics(self):
        """Extract training metrics from all tasks"""
        training_data = []
        
        for task_id, data in self.tasks_data.items():
            training_results = data.get('training_results', {})
            training_condition = data.get('training_condition', {})
            
            # Extract metrics from epoch_results and final_metrics
            epoch_results = training_results.get('epoch_results', [])
            final_metrics = training_results.get('final_metrics', {})
            
            if epoch_results:  # List of per-epoch results
                final_epoch = len(epoch_results) - 1
                last_epoch = epoch_results[-1] if epoch_results else {}
                
                # Get convergence info
                convergence_info = final_metrics.get('convergence_info', {})
                
                metrics = {
                    'task_id': task_id,
                    'train_game': training_condition.get('game', 'unknown'),
                    'train_opponent_range': training_condition.get('opponent_range', 'unknown'),
                    'train_opponent_probs': str(training_condition.get('opponent_probs', [])),
                    'final_epoch': final_metrics.get('total_epochs', len(epoch_results)),
                    'final_total_loss': last_epoch.get('total_loss', np.nan),
                    'final_rl_loss': last_epoch.get('rl_loss', np.nan),
                    'final_tom_loss': last_epoch.get('opponent_policy_loss', np.nan),
                    'converged': convergence_info.get('converged', False),
                    'epochs_trained': convergence_info.get('epochs_trained', len(epoch_results))
                }
                training_data.append(metrics)
        
        return pd.DataFrame(training_data)
    
    def extract_generalization_metrics(self):
        """Extract generalization test results into a matrix format"""
        gen_data = []
        
        # Mapping for opponent ranges to mean probabilities for sorting/comparison
        range_to_prob = {
            'low': 0.2, 'mid_low': 0.4, 'mid': 0.5, 
            'mid_high': 0.6, 'high': 0.8
        }
        
        for task_id, data in self.tasks_data.items():
            training_condition = data.get('training_condition', {})
            eval_results = data.get('evaluation_results', {})
            
            train_game = training_condition.get('game', 'unknown')
            train_opp_range = training_condition.get('opponent_range', 'unknown')
            train_opp_probs = training_condition.get('opponent_probs', [])
            train_opp_mean = np.mean(train_opp_probs) if train_opp_probs else range_to_prob.get(train_opp_range, 0.5)
            
            # Parse evaluation results - each test condition contains nested results
            for test_cond_name, test_cond_data in eval_results.items():
                if not isinstance(test_cond_data, dict):
                    continue
                    
                test_game = test_cond_data.get('game', 'unknown')
                test_opp_range = test_cond_data.get('opponent_range', 'unknown')
                condition_type = test_cond_data.get('condition', 'unknown')
                
                # Get results for each opponent in this test condition
                opp_results = test_cond_data.get('results', {})
                
                # Aggregate metrics across all opponents in this condition
                coop_rates = []
                avg_rewards = []
                
                for opp_key, opp_data in opp_results.items():
                    if isinstance(opp_data, dict):
                        coop_rate = opp_data.get('cooperation_rate')
                        avg_reward = opp_data.get('average_reward')
                        
                        if coop_rate is not None:
                            coop_rates.append(coop_rate)
                        if avg_reward is not None:
                            avg_rewards.append(avg_reward)
                
                # Calculate aggregate metrics
                if coop_rates or avg_rewards:
                    test_opp_mean = range_to_prob.get(test_opp_range, 0.5)
                    
                    metrics = {
                        'task_id': task_id,
                        'train_game': train_game,
                        'train_opponent_range': train_opp_range,
                        'train_opponent_mean': train_opp_mean,
                        'test_condition_name': test_cond_name,
                        'test_game': test_game,
                        'test_opponent_range': test_opp_range,
                        'test_opponent_mean': test_opp_mean,
                        'condition_type': condition_type,
                        'cooperation_rate': np.mean(coop_rates) if coop_rates else np.nan,
                        'avg_reward': np.mean(avg_rewards) if avg_rewards else np.nan,
                        'same_game': train_game == test_game,
                        'same_opponent_range': train_opp_range == test_opp_range
                    }
                    gen_data.append(metrics)
        
        return pd.DataFrame(gen_data)
    
    def plot_training_curves(self, df_training):
        """Plot 1.1: Training curves for all tasks"""
        print("Generating training curves...")
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        
        for task_id, data in sorted(self.tasks_data.items()):
            if task_id >= 16:
                continue
                
            epoch_results = data.get('training_results', {}).get('epoch_results', [])
            train_cond = data.get('training_condition', {})
            
            ax = axes[task_id]
            
            if epoch_results:
                epochs = range(len(epoch_results))
                
                # Extract losses from epoch results
                total_losses = [e.get('total_loss', np.nan) for e in epoch_results]
                rl_losses = [e.get('rl_loss', np.nan) for e in epoch_results]
                tom_losses = [e.get('opponent_policy_loss', np.nan) for e in epoch_results]
                
                # Plot losses
                ax.plot(epochs, rl_losses, label='RL Loss', alpha=0.7)
                ax.plot(epochs, tom_losses, label='ToM Loss', alpha=0.7)
                ax.plot(epochs, total_losses, label='Total Loss', linewidth=2, color='black')
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
            
            game = train_cond.get('game', 'unknown')
            opp_range = train_cond.get('opponent_range', 'unknown')
            ax.set_title(f"Task {task_id}: {self.game_abbrev.get(game, game)}, {opp_range}", 
                        fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'training' / 'training_curves_all_tasks.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved training_curves_all_tasks.png")
    
    def plot_training_comparison(self, df_training):
        """Plot 1.2: Cross-task training comparison"""
        print("Generating training comparison plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Convergence epochs
        ax = axes[0]
        df_sorted = df_training.sort_values('epochs_trained')
        colors = ['green' if c else 'red' for c in df_sorted['converged']]
        ax.barh(range(len(df_sorted)), df_sorted['epochs_trained'], color=colors, alpha=0.6)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels([f"T{t}" for t in df_sorted['task_id']], fontsize=8)
        ax.set_xlabel('Epochs Trained')
        ax.set_title('Training Duration by Task')
        ax.axvline(df_sorted['epochs_trained'].median(), color='black', 
                  linestyle='--', label='Median')
        ax.legend()
        
        # Final losses
        ax = axes[1]
        df_sorted = df_training.sort_values('final_total_loss')
        ax.barh(range(len(df_sorted)), df_sorted['final_total_loss'], alpha=0.6)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels([f"T{t}" for t in df_sorted['task_id']], fontsize=8)
        ax.set_xlabel('Final Total Loss')
        ax.set_title('Final Training Loss by Task')
        
        # Loss components comparison
        ax = axes[2]
        x = np.arange(len(df_training))
        width = 0.35
        ax.bar(x - width/2, df_training['final_rl_loss'], width, label='RL Loss', alpha=0.7)
        ax.bar(x + width/2, df_training['final_tom_loss'], width, label='ToM Loss', alpha=0.7)
        ax.set_xlabel('Task ID')
        ax.set_ylabel('Final Loss Value')
        ax.set_title('RL vs ToM Loss Components')
        ax.set_xticks(x)
        ax.set_xticklabels([f"T{t}" for t in df_training['task_id']], fontsize=8, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'training' / 'training_performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved training_performance_comparison.png")
    
    def plot_generalization_matrix(self, df_gen):
        """Plot 2.1: Generalization matrix heatmaps"""
        print("Generating generalization matrix heatmaps...")
        
        metrics = ['cooperation_rate', 'avg_reward']
        metric_names = ['Cooperation Rate', 'Average Reward']
        
        for metric, metric_name in zip(metrics, metric_names):
            fig, ax = plt.subplots(figsize=(16, 12))
            
            # Create matrix: rows = training conditions, cols = test conditions
            train_conditions = df_gen.groupby(['train_game', 'train_opponent_range']).size().index
            test_conditions = df_gen.groupby(['test_game', 'test_opponent_range']).size().index
            
            matrix = np.zeros((len(train_conditions), len(test_conditions)))
            
            for i, (train_game, train_opp) in enumerate(train_conditions):
                for j, (test_game, test_opp) in enumerate(test_conditions):
                    mask = ((df_gen['train_game'] == train_game) & 
                           (df_gen['train_opponent_range'] == train_opp) &
                           (df_gen['test_game'] == test_game) & 
                           (df_gen['test_opponent_range'] == test_opp))
                    values = df_gen[mask][metric]
                    matrix[i, j] = values.mean() if len(values) > 0 else np.nan
            
            # Plot heatmap
            sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                       vmin=0, vmax=1 if metric != 'avg_reward' else None,
                       ax=ax, cbar_kws={'label': metric_name})
            
            # Labels
            train_labels = [f"{self.game_abbrev.get(g, g)}-{o}" 
                          for g, o in train_conditions]
            test_labels = [f"{self.game_abbrev.get(g, g)}-{o}" 
                         for g, o in test_conditions]
            
            ax.set_xticklabels(test_labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(train_labels, rotation=0, fontsize=8)
            ax.set_xlabel('Test Condition (Game-Opponent Range)', fontsize=11)
            ax.set_ylabel('Training Condition (Game-Opponent Range)', fontsize=11)
            ax.set_title(f'Generalization Matrix: {metric_name}', fontsize=14, pad=20)
            
            plt.tight_layout()
            plt.savefig(self.fig_dir / 'generalization' / f'generalization_matrix_{metric}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved generalization_matrix_{metric}.png")
    
    def plot_within_vs_cross_game(self, df_gen):
        """Plot 2.2: Within-game vs cross-game generalization"""
        print("Generating within vs cross-game comparison...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        metrics = ['cooperation_rate', 'avg_reward']
        titles = ['Cooperation Rate', 'Average Reward']
        
        for ax, metric, title in zip(axes, metrics, titles):
            within_game = df_gen[df_gen['same_game']][metric].dropna()
            cross_game = df_gen[~df_gen['same_game']][metric].dropna()
            
            data = [within_game, cross_game]
            positions = [1, 2]
            
            bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                           showmeans=True, meanline=True)
            
            for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xticklabels(['Within-Game', 'Cross-Game'])
            ax.set_ylabel(title)
            ax.set_title(f'{title}\n(Within vs Cross-Game)')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistical test
            if len(within_game) > 0 and len(cross_game) > 0:
                t_stat, p_val = stats.ttest_ind(within_game, cross_game)
                ax.text(0.5, 0.95, f'p = {p_val:.4f}', transform=ax.transAxes,
                       ha='center', va='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'generalization' / 'within_vs_cross_game_generalization.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved within_vs_cross_game_generalization.png")
        
        # Statistical report
        stats_data = []
        for metric in metrics:
            within = df_gen[df_gen['same_game']][metric].dropna()
            cross = df_gen[~df_gen['same_game']][metric].dropna()
            t_stat, p_val = stats.ttest_ind(within, cross)
            effect_size = (within.mean() - cross.mean()) / np.sqrt((within.std()**2 + cross.std()**2) / 2)
            
            stats_data.append({
                'metric': metric,
                'within_mean': within.mean(),
                'within_std': within.std(),
                'cross_mean': cross.mean(),
                'cross_std': cross.std(),
                't_statistic': t_stat,
                'p_value': p_val,
                'cohens_d': effect_size
            })
        
        df_stats = pd.DataFrame(stats_data)
        df_stats.to_csv(self.table_dir / 'within_vs_cross_game_statistics.csv', index=False)
        print("  Saved within_vs_cross_game_statistics.csv")
    
    def plot_opponent_generalization(self, df_gen):
        """Plot 2.3: Opponent generalization patterns"""
        print("Generating opponent generalization patterns...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, game in enumerate(self.games):
            ax = axes[idx]
            
            # For each training opponent range, plot performance across test opponent ranges
            train_opp_ranges = sorted(df_gen['train_opponent_range'].unique())
            
            for train_opp in train_opp_ranges:
                mask = (df_gen['test_game'] == game) & (df_gen['train_opponent_range'] == train_opp)
                subset = df_gen[mask].groupby('test_opponent_range')['cooperation_rate'].mean()
                
                if len(subset) > 0:
                    # Use numeric values for x-axis
                    test_ranges = subset.index.tolist()
                    x_pos = range(len(test_ranges))
                    ax.plot(x_pos, subset.values, marker='o', 
                           label=f'Train {train_opp}', alpha=0.7)
            
            ax.set_xlabel('Test Opponent Range')
            ax.set_ylabel('Cooperation Rate')
            ax.set_title(f'{self.game_abbrev.get(game, game)} - Opponent Generalization')
            
            # Set x-tick labels
            test_ranges = sorted(df_gen['test_opponent_range'].unique())
            ax.set_xticks(range(len(test_ranges)))
            ax.set_xticklabels(test_ranges, rotation=45, ha='right')
            
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'generalization' / 'opponent_generalization_patterns.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved opponent_generalization_patterns.png")
    
    def plot_tom_analysis(self, df_gen):
        """Plot 3: Theory of Mind analysis (simplified - no ToM metrics available)"""
        print("Skipping ToM analysis - no opponent prediction accuracy metrics found...")
        # ToM metrics not available in this data format
        # Future implementation can add this if ToM tracking is added to experiments
    
    def plot_game_performance(self, df_gen):
        """Plot 5.1: Performance by game type"""
        print("Generating game-specific performance analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 5.1 Performance by game - training
        ax = axes[0, 0]
        game_performance = df_gen.groupby('train_game')['cooperation_rate'].agg(['mean', 'std'])
        game_performance.plot(kind='bar', y='mean', yerr='std', ax=ax, 
                             color='steelblue', alpha=0.7, legend=False)
        ax.set_xlabel('Training Game')
        ax.set_ylabel('Mean Cooperation Rate')
        ax.set_title('Training Performance by Game Type')
        ax.set_xticklabels([self.game_abbrev.get(g, g) for g in game_performance.index], 
                          rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 5.1 Performance by game - testing
        ax = axes[0, 1]
        test_game_performance = df_gen.groupby('test_game')['cooperation_rate'].agg(['mean', 'std'])
        test_game_performance.plot(kind='bar', y='mean', yerr='std', ax=ax, 
                                  color='coral', alpha=0.7, legend=False)
        ax.set_xlabel('Test Game')
        ax.set_ylabel('Mean Cooperation Rate')
        ax.set_title('Test Performance by Game Type')
        ax.set_xticklabels([self.game_abbrev.get(g, g) for g in test_game_performance.index], 
                          rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 5.2 Game transition difficulty
        ax = axes[1, 0]
        transition_matrix = np.zeros((len(self.games), len(self.games)))
        
        for i, train_game in enumerate(self.games):
            for j, test_game in enumerate(self.games):
                mask = (df_gen['train_game'] == train_game) & (df_gen['test_game'] == test_game)
                transition_matrix[i, j] = df_gen[mask]['cooperation_rate'].mean()
        
        sns.heatmap(transition_matrix, annot=True, fmt='.3f', cmap='YlGn', 
                   vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Cooperation Rate'})
        game_labels = [self.game_abbrev.get(g, g) for g in self.games]
        ax.set_xticklabels(game_labels, rotation=0)
        ax.set_yticklabels(game_labels, rotation=0)
        ax.set_xlabel('Test Game')
        ax.set_ylabel('Training Game')
        ax.set_title('Game Transition Matrix\n(Cooperation Rate)')
        
        # 5.3 Reward by game
        ax = axes[1, 1]
        game_rewards = df_gen.groupby(['train_game', 'test_game'])['avg_reward'].mean().unstack()
        game_rewards.plot(kind='bar', ax=ax, alpha=0.7)
        ax.set_xlabel('Training Game')
        ax.set_ylabel('Average Reward')
        ax.set_title('Average Reward by Game Combination')
        ax.set_xticklabels([self.game_abbrev.get(g, g) for g in game_rewards.index], 
                          rotation=45)
        ax.legend(title='Test Game', labels=[self.game_abbrev.get(g, g) for g in game_rewards.columns],
                 fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'generalization' / 'performance_by_game.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved performance_by_game.png")
    
    def plot_transfer_learning_analysis(self, df_gen):
        """Plot 6.1: Transfer learning effects"""
        print("Generating transfer learning analysis...")
        
        # Calculate average generalization performance per training condition
        transfer_scores = df_gen.groupby(['train_game', 'train_opponent_range']).agg({
            'cooperation_rate': 'mean',
            'avg_reward': 'mean'
        }).reset_index()
        
        transfer_scores['condition'] = (transfer_scores['train_game'].apply(lambda x: self.game_abbrev.get(x, x)) + 
                                       '_' + transfer_scores['train_opponent_range'].astype(str))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        metrics = ['cooperation_rate', 'avg_reward']
        titles = ['Cooperation Rate', 'Average Reward']
        
        for ax, metric, title in zip(axes, metrics, titles):
            sorted_data = transfer_scores.sort_values(metric, ascending=False)
            colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_data)))
            
            ax.barh(range(len(sorted_data)), sorted_data[metric], color=colors, alpha=0.7)
            ax.set_yticks(range(len(sorted_data)))
            ax.set_yticklabels(sorted_data['condition'], fontsize=8)
            ax.set_xlabel(f'Mean {title}')
            ax.set_title(f'Transfer Learning Ranking\n(by {title})')
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'generalization' / 'transfer_learning_rankings.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved transfer_learning_rankings.png")
        
        # Save rankings
        transfer_scores.to_csv(self.table_dir / 'transfer_learning_rankings.csv', index=False)
        print("  Saved transfer_learning_rankings.csv")
    
    def statistical_analysis(self, df_training, df_gen):
        """Comprehensive statistical analysis"""
        print("Performing statistical analysis...")
        
        # Descriptive statistics
        desc_stats = []
        
        for metric in ['cooperation_rate', 'avg_reward']:
            stats_dict = {
                'metric': metric,
                'mean': df_gen[metric].mean(),
                'std': df_gen[metric].std(),
                'min': df_gen[metric].min(),
                'max': df_gen[metric].max(),
                'median': df_gen[metric].median(),
                'q25': df_gen[metric].quantile(0.25),
                'q75': df_gen[metric].quantile(0.75)
            }
            desc_stats.append(stats_dict)
        
        df_desc = pd.DataFrame(desc_stats)
        df_desc.to_csv(self.table_dir / 'descriptive_statistics.csv', index=False)
        print("  Saved descriptive_statistics.csv")
        
        # ANOVA by game type
        inferential_stats = []
        
        for metric in ['cooperation_rate', 'avg_reward']:
            groups = [df_gen[df_gen['train_game'] == game][metric].dropna() 
                     for game in self.games]
            # Filter out empty groups
            groups = [g for g in groups if len(g) > 0]
            if len(groups) > 1:
                f_stat, p_val = stats.f_oneway(*groups)
                
                inferential_stats.append({
                    'test': f'ANOVA_{metric}_by_train_game',
                    'f_statistic': f_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                })
        
        df_inf = pd.DataFrame(inferential_stats)
        df_inf.to_csv(self.table_dir / 'inferential_statistics.csv', index=False)
        print("  Saved inferential_statistics.csv")
        
        # PCA visualization
        print("Generating PCA visualization...")
        
        # Prepare feature matrix: average performance per training condition
        train_conditions = df_gen.groupby(['task_id', 'train_game', 'train_opponent_range']).size().index
        
        feature_matrix = []
        labels = []
        
        for task_id, train_game, train_opp_range in train_conditions:
            task_data = df_gen[df_gen['task_id'] == task_id]
            
            # Feature vector: cooperation rate for each test condition
            features = []
            for test_game in self.games:
                for test_opp_range in sorted(df_gen['test_opponent_range'].unique()):
                    mask = (task_data['test_game'] == test_game) & (task_data['test_opponent_range'] == test_opp_range)
                    coop = task_data[mask]['cooperation_rate'].mean()
                    features.append(coop if not np.isnan(coop) else 0)
            
            if len(features) > 0:  # Only add if we have features
                feature_matrix.append(features)
                labels.append(f"{self.game_abbrev.get(train_game, train_game)}-{train_opp_range}")
        
        if len(feature_matrix) > 2:  # Need at least 3 samples for PCA
            # Pad feature vectors to same length
            max_len = max(len(f) for f in feature_matrix)
            feature_matrix = [f + [0]*(max_len - len(f)) for f in feature_matrix]
            feature_matrix = np.array(feature_matrix)
            
            # Perform PCA
            pca = PCA(n_components=min(2, len(feature_matrix)))
            pca_result = pca.fit_transform(feature_matrix)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Color by game
            game_colors = {g: plt.cm.Set3(i) for i, g in enumerate(self.games)}
            colors = [game_colors.get(train_game, 'gray') for _, train_game, _ in train_conditions if len(feature_matrix) > 0]
            
            scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1] if pca_result.shape[1] > 1 else np.zeros(len(pca_result)), 
                               c=colors[:len(pca_result)], s=200, alpha=0.6, edgecolors='black')
            
            # Add labels
            for i, label in enumerate(labels):
                if i < len(pca_result):
                    y_val = pca_result[i, 1] if pca_result.shape[1] > 1 else 0
                    ax.annotate(label, (pca_result[i, 0], y_val), 
                               fontsize=8, ha='center', va='center')
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            if pca.n_components > 1:
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax.set_title('PCA: Training Condition Performance Space')
            ax.grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=game_colors[g], label=self.game_abbrev.get(g, g)) 
                             for g in self.games]
            ax.legend(handles=legend_elements, loc='best')
            
            plt.tight_layout()
            plt.savefig(self.fig_dir / 'statistical' / 'performance_space_pca.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("  Saved performance_space_pca.png")
        else:
            print("  Skipping PCA - insufficient data points")
    
    def generate_executive_summary(self, df_training, df_gen):
        """Generate executive summary report"""
        print("Generating executive summary...")
        
        summary = []
        summary.append("# Generalization Matrix Analysis - Executive Summary\n")
        summary.append(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        summary.append(f"**Experiment Directory:** {self.experiment_dir}\n")
        summary.append(f"**Total Tasks Analyzed:** {len(self.tasks_data)}\n\n")
        
        summary.append("## Key Findings\n\n")
        
        # Training convergence
        converged = df_training['converged'].sum()
        total = len(df_training)
        summary.append(f"### 1. Training Performance\n")
        summary.append(f"- **Convergence Rate:** {converged}/{total} tasks ({100*converged/total:.1f}%)\n")
        summary.append(f"- **Average Training Epochs:** {df_training['epochs_trained'].mean():.1f} ± {df_training['epochs_trained'].std():.1f}\n")
        summary.append(f"- **Average Final Loss:** {df_training['final_total_loss'].mean():.4f}\n\n")
        
        # Generalization performance
        summary.append(f"### 2. Generalization Performance\n")
        summary.append(f"- **Overall Cooperation Rate:** {df_gen['cooperation_rate'].mean():.3f} ± {df_gen['cooperation_rate'].std():.3f}\n")
        summary.append(f"- **Overall Average Reward:** {df_gen['avg_reward'].mean():.3f} ± {df_gen['avg_reward'].std():.3f}\n")
        
        within_coop = df_gen[df_gen['same_game']]['cooperation_rate'].mean()
        cross_coop = df_gen[~df_gen['same_game']]['cooperation_rate'].mean()
        summary.append(f"- **Within-Game Cooperation:** {within_coop:.3f}\n")
        summary.append(f"- **Cross-Game Cooperation:** {cross_coop:.3f}\n")
        summary.append(f"- **Generalization Gap:** {within_coop - cross_coop:.3f}\n\n")
        
        # Game-specific insights
        summary.append(f"### 3. Game-Specific Insights\n")
        for game in self.games:
            game_data = df_gen[df_gen['test_game'] == game]
            if len(game_data) > 0:
                summary.append(f"- **{self.game_abbrev.get(game, game)}:** Avg Coop = {game_data['cooperation_rate'].mean():.3f}, "
                             f"Avg Reward = {game_data['avg_reward'].mean():.3f}\n")
        
        summary.append("\n### 4. Best Performing Conditions\n")
        top_coop = df_gen.groupby(['train_game', 'train_opponent_range'])['cooperation_rate'].mean().nlargest(3)
        summary.append("**Top 3 Training Conditions (by cooperation rate):**\n")
        for (game, opp_range), coop in top_coop.items():
            summary.append(f"- {self.game_abbrev.get(game, game)}, {opp_range}: {coop:.3f}\n")
        
        summary.append("\n## Conclusion\n\n")
        summary.append("This analysis reveals patterns in how ToM-RL agents generalize across different "
                      "mixed-motive games and opponent types. Key insights include the trade-off between "
                      "within-game and cross-game generalization, the role of Theory of Mind in successful "
                      "cooperation, and game-specific differences in learning and transfer.\n")
        
        summary_text = ''.join(summary)
        
        with open(self.output_dir / 'executive_summary.md', 'w') as f:
            f.write(summary_text)
        
        print("  Saved executive_summary.md")
        return summary_text
    
    def export_complete_data(self, df_training, df_gen):
        """Export consolidated data"""
        print("Exporting complete datasets...")
        
        df_training.to_csv(self.data_dir / 'training_results.csv', index=False)
        df_gen.to_csv(self.data_dir / 'generalization_results.csv', index=False)
        
        print("  Saved training_results.csv")
        print("  Saved generalization_results.csv")
    
    def run_complete_analysis(self):
        """Execute all analyses"""
        print("="*70)
        print("GENERALIZATION MATRIX COMPREHENSIVE ANALYSIS")
        print("="*70)
        print()
        
        # Load data
        self.load_all_tasks()
        
        # Extract metrics
        print("Extracting metrics...")
        df_training = self.extract_training_metrics()
        df_gen = self.extract_generalization_metrics()
        print(f"  Training data: {len(df_training)} tasks")
        print(f"  Generalization data: {len(df_gen)} test conditions\n")
        
        # Run all analyses
        self.plot_training_curves(df_training)
        self.plot_training_comparison(df_training)
        self.plot_generalization_matrix(df_gen)
        self.plot_within_vs_cross_game(df_gen)
        self.plot_opponent_generalization(df_gen)
        self.plot_tom_analysis(df_gen)
        self.plot_game_performance(df_gen)
        self.plot_transfer_learning_analysis(df_gen)
        self.statistical_analysis(df_training, df_gen)
        self.export_complete_data(df_training, df_gen)
        summary = self.generate_executive_summary(df_training, df_gen)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nAll results saved to: {self.output_dir}")
        print(f"  - Figures: {self.fig_dir}")
        print(f"  - Tables: {self.table_dir}")
        print(f"  - Data: {self.data_dir}")
        print(f"\nExecutive Summary:\n")
        print(summary)


def main():
    """Main execution"""
    # Paths
    base_dir = Path(__file__).parent.parent.parent  # Go up to project root from experiments/analysis_scripts/
    experiment_dir = base_dir / 'experiments' / 'generalization_matrix_834222'
    output_dir = base_dir / 'experiments' / 'results'
    
    # Create analyzer and run
    analyzer = GeneralizationMatrixAnalyzer(experiment_dir, output_dir)
    analyzer.run_complete_analysis()


if __name__ == '__main__':
    main()
