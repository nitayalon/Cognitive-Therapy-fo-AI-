"""
Task-Centric Generalization Analysis
Analyzes which tasks serve as best "foundation" for generalization
Includes reward-based and behavior prediction metrics
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
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class TaskGeneralizationAnalyzer:
    """Analyzes task-level generalization patterns"""
    
    def __init__(self, experiment_dir, output_dir):
        self.experiment_dir = Path(experiment_dir)
        self.output_dir = Path(output_dir)
        self.tasks_data = {}
        self.games = ['prisoners-dilemma', 'stag-hunt', 'battle-of-sexes', 'hawk-dove']
        self.game_abbrev = {'prisoners-dilemma': 'PD', 'stag-hunt': 'SH', 
                           'battle-of-sexes': 'BoS', 'hawk-dove': 'HD'}
        
        # Opponent range mapping to defection probabilities
        self.opp_range_to_defect_prob = {
            'low': 0.2,      # avg of [0.1, 0.3]
            'mid_low': 0.4,  # avg of [0.3, 0.5]
            'mid': 0.5,      # avg of [0.4, 0.6] or just 0.5
            'mid_high': 0.6, # avg of [0.5, 0.7]
            'high': 0.8      # avg of [0.7, 0.9]
        }
        
        # Create output directories
        self.fig_dir = self.output_dir / 'task_analysis' / 'figures'
        self.table_dir = self.output_dir / 'task_analysis' / 'tables'
        self.data_dir = self.output_dir / 'task_analysis' / 'data'
        
        for dir_path in [self.fig_dir, self.table_dir, self.data_dir]:
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
        
        print(f"Successfully loaded {len(self.tasks_data)} tasks\n")
        return self.tasks_data
    
    def calculate_behavior_misprediction(self, agent_coop_rate, opponent_coop_rate):
        """
        Calculate behavior prediction error
        Agent's expected opponent defection = 1 - opponent_cooperation_rate
        This is a measure of how well agent predicts opponent behavior
        """
        if np.isnan(agent_coop_rate) or np.isnan(opponent_coop_rate):
            return np.nan
        
        # Agent should predict opponent behavior
        # If agent cooperates at rate X, it might be because it predicts 
        # opponent cooperates at some rate
        # Misprediction: difference between what agent expects and what opponent does
        return abs(agent_coop_rate - opponent_coop_rate)
    
    def extract_detailed_metrics(self):
        """Extract detailed metrics for each test condition"""
        all_data = []
        
        for task_id, task_data in self.tasks_data.items():
            training_condition = task_data.get('training_condition', {})
            eval_results = task_data.get('evaluation_results', {})
            
            train_game = training_condition.get('game', 'unknown')
            train_opp_range = training_condition.get('opponent_range', 'unknown')
            train_opp_probs = training_condition.get('opponent_probs', [])
            train_opp_defect_prob = self.opp_range_to_defect_prob.get(train_opp_range, 0.5)
            
            for test_cond_name, test_cond_data in eval_results.items():
                if not isinstance(test_cond_data, dict):
                    continue
                
                test_game = test_cond_data.get('game', 'unknown')
                test_opp_range = test_cond_data.get('opponent_range', 'unknown')
                test_opp_defect_prob = self.opp_range_to_defect_prob.get(test_opp_range, 0.5)
                
                # Get individual opponent results
                opp_results = test_cond_data.get('results', {})
                
                for opp_key, opp_data in opp_results.items():
                    if isinstance(opp_data, dict):
                        agent_coop = opp_data.get('cooperation_rate')
                        opponent_coop = opp_data.get('opponent_cooperation_rate')
                        avg_reward = opp_data.get('average_reward')
                        
                        if agent_coop is not None:
                            # Calculate misprediction
                            misprediction = self.calculate_behavior_misprediction(
                                agent_coop, opponent_coop if opponent_coop is not None else (1 - test_opp_defect_prob)
                            )
                            
                            # Categorize
                            same_task = (train_game == test_game)
                            same_opp_range = (train_opp_range == test_opp_range)
                            
                            if same_task and same_opp_range:
                                category = 'same_task_same_opp'
                            elif same_task:
                                category = 'same_task_new_opp'
                            elif same_opp_range:
                                category = 'new_task_same_opp'
                            else:
                                category = 'new_task_new_opp'
                            
                            all_data.append({
                                'task_id': task_id,
                                'train_game': train_game,
                                'train_opp_range': train_opp_range,
                                'train_opp_defect_prob': train_opp_defect_prob,
                                'test_game': test_game,
                                'test_opp_range': test_opp_range,
                                'test_opp_defect_prob': test_opp_defect_prob,
                                'agent_cooperation_rate': agent_coop,
                                'opponent_cooperation_rate': opponent_coop,
                                'avg_reward': avg_reward,
                                'behavior_misprediction': misprediction,
                                'category': category,
                                'same_task': same_task,
                                'same_opp_range': same_opp_range
                            })
        
        return pd.DataFrame(all_data)
    
    def analyze_task_generalization(self, df):
        """Analyze generalization patterns by training task"""
        print("Analyzing task-level generalization patterns...\n")
        
        task_results = []
        
        for train_game in self.games:
            train_data = df[df['train_game'] == train_game]
            
            if len(train_data) == 0:
                continue
            
            # Calculate metrics for each generalization type
            metrics = {}
            
            for category in ['same_task_same_opp', 'same_task_new_opp', 
                           'new_task_same_opp', 'new_task_new_opp']:
                cat_data = train_data[train_data['category'] == category]
                
                if len(cat_data) > 0:
                    metrics[f'{category}_reward_mean'] = cat_data['avg_reward'].mean()
                    metrics[f'{category}_reward_std'] = cat_data['avg_reward'].std()
                    metrics[f'{category}_mispred_mean'] = cat_data['behavior_misprediction'].mean()
                    metrics[f'{category}_mispred_std'] = cat_data['behavior_misprediction'].std()
                    metrics[f'{category}_n'] = len(cat_data)
                else:
                    metrics[f'{category}_reward_mean'] = np.nan
                    metrics[f'{category}_reward_std'] = np.nan
                    metrics[f'{category}_mispred_mean'] = np.nan
                    metrics[f'{category}_mispred_std'] = np.nan
                    metrics[f'{category}_n'] = 0
            
            # Calculate generalization scores (performance drop from baseline)
            baseline_reward = metrics.get('same_task_same_opp_reward_mean', np.nan)
            baseline_mispred = metrics.get('same_task_same_opp_mispred_mean', np.nan)
            
            result = {
                'train_game': train_game,
                'baseline_reward': baseline_reward,
                'baseline_misprediction': baseline_mispred,
                **metrics
            }
            
            # Calculate generalization difficulty scores
            if not np.isnan(baseline_reward):
                result['same_task_new_opp_reward_drop'] = baseline_reward - metrics.get('same_task_new_opp_reward_mean', baseline_reward)
                result['new_task_same_opp_reward_drop'] = baseline_reward - metrics.get('new_task_same_opp_reward_mean', baseline_reward)
                result['new_task_new_opp_reward_drop'] = baseline_reward - metrics.get('new_task_new_opp_reward_mean', baseline_reward)
            
            if not np.isnan(baseline_mispred):
                result['same_task_new_opp_mispred_increase'] = metrics.get('same_task_new_opp_mispred_mean', baseline_mispred) - baseline_mispred
                result['new_task_same_opp_mispred_increase'] = metrics.get('new_task_same_opp_mispred_mean', baseline_mispred) - baseline_mispred
                result['new_task_new_opp_mispred_increase'] = metrics.get('new_task_new_opp_mispred_mean', baseline_mispred) - baseline_mispred
            
            task_results.append(result)
        
        df_task_results = pd.DataFrame(task_results)
        df_task_results.to_csv(self.table_dir / 'task_generalization_summary.csv', index=False)
        print("  Saved task_generalization_summary.csv")
        
        return df_task_results
    
    def plot_task_generalization_matrix(self, df_task_results):
        """Create visualization of task generalization patterns"""
        print("Creating task generalization visualizations...")
        
        # Plot 1: Reward performance by task and category
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        categories = ['same_task_same_opp', 'same_task_new_opp', 
                     'new_task_same_opp', 'new_task_new_opp']
        category_labels = ['Same Task\nSame Opponent', 'Same Task\nNew Opponent',
                          'New Task\nSame Opponent', 'New Task\nNew Opponent']
        
        for idx, (cat, label) in enumerate(zip(categories, category_labels)):
            ax = axes[idx // 2, idx % 2]
            
            games = []
            rewards = []
            errors = []
            
            for _, row in df_task_results.iterrows():
                reward_mean = row.get(f'{cat}_reward_mean')
                reward_std = row.get(f'{cat}_reward_std')
                
                if not np.isnan(reward_mean):
                    games.append(self.game_abbrev.get(row['train_game'], row['train_game']))
                    rewards.append(reward_mean)
                    errors.append(reward_std if not np.isnan(reward_std) else 0)
            
            if games:
                x_pos = np.arange(len(games))
                bars = ax.bar(x_pos, rewards, yerr=errors, capsize=5, alpha=0.7)
                
                # Color bars by value
                colors = plt.cm.RdYlGn([r/max(rewards) if max(rewards) > 0 else 0.5 for r in rewards])
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(games, fontsize=11)
                ax.set_ylabel('Average Reward', fontsize=11)
                ax.set_title(label, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for i, (r, e) in enumerate(zip(rewards, errors)):
                    ax.text(i, r + e + 0.1, f'{r:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Average Reward by Training Task and Generalization Type', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'task_reward_generalization.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved task_reward_generalization.png")
        
        # Plot 2: Behavior misprediction
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for idx, (cat, label) in enumerate(zip(categories, category_labels)):
            ax = axes[idx // 2, idx % 2]
            
            games = []
            mispreds = []
            errors = []
            
            for _, row in df_task_results.iterrows():
                mispred_mean = row.get(f'{cat}_mispred_mean')
                mispred_std = row.get(f'{cat}_mispred_std')
                
                if not np.isnan(mispred_mean):
                    games.append(self.game_abbrev.get(row['train_game'], row['train_game']))
                    mispreds.append(mispred_mean)
                    errors.append(mispred_std if not np.isnan(mispred_std) else 0)
            
            if games:
                x_pos = np.arange(len(games))
                bars = ax.bar(x_pos, mispreds, yerr=errors, capsize=5, alpha=0.7)
                
                # Color bars (lower is better for misprediction)
                max_val = max(mispreds) if max(mispreds) > 0 else 1
                colors = plt.cm.RdYlGn_r([m/max_val for m in mispreds])
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(games, fontsize=11)
                ax.set_ylabel('Behavior Misprediction', fontsize=11)
                ax.set_title(label, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for i, (m, e) in enumerate(zip(mispreds, errors)):
                    ax.text(i, m + e + 0.01, f'{m:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Behavior Misprediction by Training Task and Generalization Type\n(Lower is Better)', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'task_misprediction_generalization.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved task_misprediction_generalization.png")
    
    def plot_generalization_difficulty_ranking(self, df_task_results):
        """Rank tasks by generalization difficulty"""
        print("Creating generalization difficulty rankings...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Reward-based ranking
        ax = axes[0]
        
        # Calculate average performance drop across all generalization types
        df_task_results['avg_reward_drop'] = df_task_results[[
            'same_task_new_opp_reward_drop',
            'new_task_same_opp_reward_drop',
            'new_task_new_opp_reward_drop'
        ]].mean(axis=1)
        
        df_sorted = df_task_results.sort_values('avg_reward_drop')
        
        games = [self.game_abbrev.get(g, g) for g in df_sorted['train_game']]
        drops = df_sorted['avg_reward_drop'].values
        
        y_pos = np.arange(len(games))
        bars = ax.barh(y_pos, drops, alpha=0.7)
        
        # Color by difficulty (green = easy to generalize from, red = hard)
        colors = plt.cm.RdYlGn_r(drops / max(drops) if max(drops) > 0 else [0.5]*len(drops))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(games, fontsize=11)
        ax.set_xlabel('Average Reward Drop from Baseline', fontsize=11)
        ax.set_title('Task Generalization Difficulty\n(Lower = Easier to Generalize FROM)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, drop in enumerate(drops):
            ax.text(drop + 0.02, i, f'{drop:.3f}', va='center', fontsize=9)
        
        # Misprediction-based ranking
        ax = axes[1]
        
        df_task_results['avg_mispred_increase'] = df_task_results[[
            'same_task_new_opp_mispred_increase',
            'new_task_same_opp_mispred_increase',
            'new_task_new_opp_mispred_increase'
        ]].mean(axis=1)
        
        df_sorted = df_task_results.sort_values('avg_mispred_increase')
        
        games = [self.game_abbrev.get(g, g) for g in df_sorted['train_game']]
        increases = df_sorted['avg_mispred_increase'].values
        
        y_pos = np.arange(len(games))
        bars = ax.barh(y_pos, increases, alpha=0.7)
        
        # Color by difficulty
        max_increase = max(increases) if max(increases) > 0 else 1
        colors = plt.cm.RdYlGn_r(increases / max_increase)
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(games, fontsize=11)
        ax.set_xlabel('Average Misprediction Increase from Baseline', fontsize=11)
        ax.set_title('Behavior Prediction Difficulty\n(Lower = Better Prediction Transfer)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, inc in enumerate(increases):
            ax.text(inc + 0.002, i, f'{inc:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'generalization_difficulty_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved generalization_difficulty_ranking.png")
        
        # Save rankings
        df_task_results[['train_game', 'avg_reward_drop', 'avg_mispred_increase']].to_csv(
            self.table_dir / 'generalization_difficulty_rankings.csv', index=False
        )
        print("  Saved generalization_difficulty_rankings.csv")
    
    def create_detailed_breakdown(self, df):
        """Create detailed breakdown tables"""
        print("Creating detailed breakdown tables...")
        
        # By training task and category
        breakdown = df.groupby(['train_game', 'category']).agg({
            'avg_reward': ['mean', 'std', 'count'],
            'behavior_misprediction': ['mean', 'std'],
            'agent_cooperation_rate': 'mean',
            'opponent_cooperation_rate': 'mean'
        }).round(4)
        
        breakdown.to_csv(self.table_dir / 'detailed_breakdown_by_task_category.csv')
        print("  Saved detailed_breakdown_by_task_category.csv")
        
        # Opponent-specific analysis
        opp_breakdown = df.groupby(['train_game', 'test_opp_range', 'category']).agg({
            'avg_reward': ['mean', 'std', 'count'],
            'behavior_misprediction': ['mean', 'std']
        }).round(4)
        
        opp_breakdown.to_csv(self.table_dir / 'detailed_breakdown_with_opponents.csv')
        print("  Saved detailed_breakdown_with_opponents.csv")
    
    def generate_executive_summary(self, df_task_results, df):
        """Generate executive summary"""
        print("\nGenerating executive summary...")
        
        summary = []
        summary.append("# Task-Centric Generalization Analysis\n")
        summary.append(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        summary.append("## Research Question\n")
        summary.append("For each training task (game), how well do agents generalize to:\n")
        summary.append("1. New opponents in the **same task**\n")
        summary.append("2. Similar opponents in **new tasks**\n")
        summary.append("3. New opponents in **new tasks**\n\n")
        summary.append("Measured by both **reward performance** and **behavior prediction accuracy**\n\n")
        
        summary.append("## Task Generalization Rankings\n\n")
        
        # Easiest to generalize FROM (by reward)
        df_sorted_reward = df_task_results.sort_values('avg_reward_drop')
        summary.append("### Easiest Tasks to Generalize FROM (by reward):\n")
        for idx, row in df_sorted_reward.head(2).iterrows():
            game_name = self.game_abbrev.get(row['train_game'], row['train_game'])
            summary.append(f"{idx+1}. **{game_name}**: Avg reward drop = {row['avg_reward_drop']:.3f}\n")
        
        summary.append("\n### Hardest Tasks to Generalize FROM (by reward):\n")
        for idx, row in df_sorted_reward.tail(2).iterrows():
            game_name = self.game_abbrev.get(row['train_game'], row['train_game'])
            summary.append(f"{idx+1}. **{game_name}**: Avg reward drop = {row['avg_reward_drop']:.3f}\n")
        
        # Best prediction transfer
        df_sorted_mispred = df_task_results.sort_values('avg_mispred_increase')
        summary.append("\n### Best Behavior Prediction Transfer:\n")
        for idx, row in df_sorted_mispred.head(2).iterrows():
            game_name = self.game_abbrev.get(row['train_game'], row['train_game'])
            summary.append(f"{idx+1}. **{game_name}**: Avg misprediction increase = {row['avg_mispred_increase']:.4f}\n")
        
        summary.append("\n### Worst Behavior Prediction Transfer:\n")
        for idx, row in df_sorted_mispred.tail(2).iterrows():
            game_name = self.game_abbrev.get(row['train_game'], row['train_game'])
            summary.append(f"{idx+1}. **{game_name}**: Avg misprediction increase = {row['avg_mispred_increase']:.4f}\n")
        
        summary.append("\n## Detailed Findings by Task\n\n")
        
        for _, row in df_task_results.iterrows():
            game_name = self.game_abbrev.get(row['train_game'], row['train_game'])
            summary.append(f"### {game_name} (Training Task)\n")
            summary.append(f"- **Baseline Reward:** {row['baseline_reward']:.3f}\n")
            summary.append(f"- **Baseline Misprediction:** {row['baseline_misprediction']:.4f}\n\n")
            
            summary.append("**Generalization Performance:**\n")
            summary.append(f"- Same task, new opponents: Reward = {row.get('same_task_new_opp_reward_mean', np.nan):.3f} ")
            summary.append(f"(drop: {row.get('same_task_new_opp_reward_drop', 0):.3f}), ")
            summary.append(f"Misprediction = {row.get('same_task_new_opp_mispred_mean', np.nan):.4f}\n")
            
            summary.append(f"- New tasks, same opponents: Reward = {row.get('new_task_same_opp_reward_mean', np.nan):.3f} ")
            summary.append(f"(drop: {row.get('new_task_same_opp_reward_drop', 0):.3f}), ")
            summary.append(f"Misprediction = {row.get('new_task_same_opp_mispred_mean', np.nan):.4f}\n")
            
            summary.append(f"- New tasks, new opponents: Reward = {row.get('new_task_new_opp_reward_mean', np.nan):.3f} ")
            summary.append(f"(drop: {row.get('new_task_new_opp_reward_drop', 0):.3f}), ")
            summary.append(f"Misprediction = {row.get('new_task_new_opp_mispred_mean', np.nan):.4f}\n\n")
        
        summary.append("## Key Insights\n\n")
        
        # Which type of generalization is harder
        avg_same_task = df[df['category'] == 'same_task_new_opp']['avg_reward'].mean()
        avg_new_task = df[df['category'] == 'new_task_same_opp']['avg_reward'].mean()
        avg_both = df[df['category'] == 'new_task_new_opp']['avg_reward'].mean()
        
        summary.append(f"1. **Average rewards across all agents:**\n")
        summary.append(f"   - Same task, new opponents: {avg_same_task:.3f}\n")
        summary.append(f"   - New tasks, same opponents: {avg_new_task:.3f}\n")
        summary.append(f"   - New tasks, new opponents: {avg_both:.3f}\n\n")
        
        if avg_new_task < avg_same_task:
            summary.append("2. **Task generalization is harder than opponent generalization**\n\n")
        else:
            summary.append("2. **Opponent generalization is harder than task generalization**\n\n")
        
        # Behavior prediction patterns
        avg_mispred_same_task = df[df['category'] == 'same_task_new_opp']['behavior_misprediction'].mean()
        avg_mispred_new_task = df[df['category'] == 'new_task_same_opp']['behavior_misprediction'].mean()
        
        summary.append(f"3. **Behavior prediction accuracy:**\n")
        summary.append(f"   - Misprediction with new opponents (same task): {avg_mispred_same_task:.4f}\n")
        summary.append(f"   - Misprediction with new tasks (same opponents): {avg_mispred_new_task:.4f}\n")
        
        if avg_mispred_new_task > avg_mispred_same_task:
            summary.append("   - **Agents have more difficulty predicting behavior in new tasks**\n\n")
        else:
            summary.append("   - **Agents have more difficulty predicting behavior with new opponents**\n\n")
        
        summary_text = ''.join(summary)
        
        with open(self.output_dir / 'task_analysis' / 'executive_summary.md', 'w') as f:
            f.write(summary_text)
        
        print("  Saved executive_summary.md")
        return summary_text
    
    def run_complete_analysis(self):
        """Execute complete task-centric analysis"""
        print("="*70)
        print("TASK-CENTRIC GENERALIZATION ANALYSIS")
        print("="*70)
        print()
        
        # Load data
        self.load_all_tasks()
        
        # Extract detailed metrics
        print("Extracting detailed metrics...")
        df = self.extract_detailed_metrics()
        df.to_csv(self.data_dir / 'complete_metrics.csv', index=False)
        print(f"  Extracted {len(df)} individual test results\n")
        
        # Analyze by task
        df_task_results = self.analyze_task_generalization(df)
        
        # Create visualizations
        self.plot_task_generalization_matrix(df_task_results)
        self.plot_generalization_difficulty_ranking(df_task_results)
        
        # Detailed breakdowns
        self.create_detailed_breakdown(df)
        
        # Generate summary
        summary = self.generate_executive_summary(df_task_results, df)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: {self.output_dir / 'task_analysis'}")
        print(f"\nExecutive Summary:\n")
        print(summary)


def main():
    """Main execution"""
    base_dir = Path(__file__).parent.parent.parent
    experiment_dir = base_dir / 'experiments' / 'generalization_matrix_834222'
    output_dir = base_dir / 'experiments' / 'results'
    
    analyzer = TaskGeneralizationAnalyzer(experiment_dir, output_dir)
    analyzer.run_complete_analysis()


if __name__ == '__main__':
    main()
