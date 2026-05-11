"""
2×2 Generalization Analysis for Individual Agents
Analyzes how each agent trained on specific (task, opponent) generalizes to:
1. Same task, same opponent (baseline)
2. Same task, different opponent
3. Different task, same opponent  
4. Different task, different opponent
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
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 10

class Agent2x2GeneralizationAnalyzer:
    """Analyzes individual agent generalization in 2×2 framework"""
    
    def __init__(self, experiment_dir, output_dir):
        self.experiment_dir = Path(experiment_dir)
        self.output_dir = Path(output_dir)
        self.tasks_data = {}
        self.games = ['prisoners-dilemma', 'stag-hunt', 'battle-of-sexes', 'hawk-dove']
        self.game_abbrev = {'prisoners-dilemma': 'PD', 'stag-hunt': 'SH', 
                           'battle-of-sexes': 'BoS', 'hawk-dove': 'HD'}
        
        # Create output directories
        self.fig_dir = self.output_dir / '2x2_analysis' / 'figures'
        self.table_dir = self.output_dir / '2x2_analysis' / 'tables'
        self.data_dir = self.output_dir / '2x2_analysis' / 'data'
        
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
            else:
                print(f"  Warning: Missing results for task {task_id}")
        
        print(f"Successfully loaded {len(self.tasks_data)} tasks\n")
        return self.tasks_data
    
    def categorize_test_condition(self, train_game, train_opp_range, test_game, test_opp_range):
        """
        Categorize test condition into 2×2 quadrants:
        1. Same task, same opponent (baseline)
        2. Same task, different opponent
        3. Different task, same opponent
        4. Different task, different opponent
        """
        same_task = (train_game == test_game)
        same_opponent = (train_opp_range == test_opp_range)
        
        if same_task and same_opponent:
            return 'same_task_same_opp'
        elif same_task and not same_opponent:
            return 'same_task_diff_opp'
        elif not same_task and same_opponent:
            return 'diff_task_same_opp'
        else:
            return 'diff_task_diff_opp'
    
    def analyze_agent_2x2(self, task_id, task_data):
        """Analyze single agent's 2×2 generalization pattern"""
        training_condition = task_data.get('training_condition', {})
        eval_results = task_data.get('evaluation_results', {})
        
        train_game = training_condition.get('game', 'unknown')
        train_opp_range = training_condition.get('opponent_range', 'unknown')
        train_opp_probs = training_condition.get('opponent_probs', [])
        
        # Initialize 2×2 results
        results_2x2 = {
            'same_task_same_opp': [],
            'same_task_diff_opp': [],
            'diff_task_same_opp': [],
            'diff_task_diff_opp': []
        }
        
        # Categorize all test conditions
        for test_cond_name, test_cond_data in eval_results.items():
            if not isinstance(test_cond_data, dict):
                continue
            
            test_game = test_cond_data.get('game', 'unknown')
            test_opp_range = test_cond_data.get('opponent_range', 'unknown')
            condition_type = test_cond_data.get('condition', 'unknown')
            
            # Get aggregated metrics for this test condition
            opp_results = test_cond_data.get('results', {})
            
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
            
            if coop_rates or avg_rewards:
                # Categorize into 2×2
                category = self.categorize_test_condition(
                    train_game, train_opp_range, test_game, test_opp_range
                )
                
                result = {
                    'test_condition': test_cond_name,
                    'test_game': test_game,
                    'test_opponent_range': test_opp_range,
                    'condition_type': condition_type,
                    'cooperation_rate': np.mean(coop_rates) if coop_rates else np.nan,
                    'avg_reward': np.mean(avg_rewards) if avg_rewards else np.nan,
                    'n_opponents': len(coop_rates)
                }
                results_2x2[category].append(result)
        
        return {
            'task_id': task_id,
            'train_game': train_game,
            'train_opponent_range': train_opp_range,
            'train_opponent_probs': train_opp_probs,
            'results_2x2': results_2x2
        }
    
    def create_2x2_matrix_plot(self, agent_analysis, metric='cooperation_rate'):
        """Create 2×2 visualization for one agent"""
        task_id = agent_analysis['task_id']
        train_game = agent_analysis['train_game']
        train_opp_range = agent_analysis['train_opponent_range']
        train_opp_probs = agent_analysis['train_opponent_probs']
        results_2x2 = agent_analysis['results_2x2']
        
        # Calculate mean values for each quadrant
        quadrant_values = {}
        for category, results_list in results_2x2.items():
            values = [r[metric] for r in results_list if not np.isnan(r[metric])]
            quadrant_values[category] = {
                'mean': np.mean(values) if values else 0,
                'std': np.std(values) if values else 0,
                'n': len(values)
            }
        
        # Create 2×2 matrix
        matrix = np.array([
            [quadrant_values['same_task_same_opp']['mean'], 
             quadrant_values['same_task_diff_opp']['mean']],
            [quadrant_values['diff_task_same_opp']['mean'], 
             quadrant_values['diff_task_diff_opp']['mean']]
        ])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        metric_label = 'Cooperation Rate' if metric == 'cooperation_rate' else 'Average Reward'
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1 if metric == 'cooperation_rate' else None,
                   ax=ax, cbar_kws={'label': metric_label},
                   linewidths=2, linecolor='black')
        
        # Labels
        ax.set_xticklabels(['Same Opponent\n(Baseline)', 'Different Opponent'], fontsize=11)
        ax.set_yticklabels(['Same Task', 'Different Task'], fontsize=11, rotation=0)
        ax.set_xlabel('Opponent Generalization', fontsize=12, fontweight='bold')
        ax.set_ylabel('Task Generalization', fontsize=12, fontweight='bold')
        
        # Title
        game_name = self.game_abbrev.get(train_game, train_game)
        title = f'Task {task_id}: {game_name}, Opponent Range={train_opp_range}\n'
        title += f'Training Opponents: {train_opp_probs}\n{metric_label}'
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        
        # Add sample sizes as text
        for i in range(2):
            for j in range(2):
                category_map = [
                    ['same_task_same_opp', 'same_task_diff_opp'],
                    ['diff_task_same_opp', 'diff_task_diff_opp']
                ]
                category = category_map[i][j]
                n = quadrant_values[category]['n']
                ax.text(j + 0.5, i + 0.85, f'(n={n})', 
                       ha='center', va='top', fontsize=8, color='gray')
        
        plt.tight_layout()
        return fig, quadrant_values
    
    def analyze_all_agents(self):
        """Run 2×2 analysis for all agents"""
        print("Running 2×2 generalization analysis for all agents...\n")
        
        all_analyses = []
        summary_data = []
        
        for task_id in sorted(self.tasks_data.keys()):
            task_data = self.tasks_data[task_id]
            
            print(f"Analyzing Task {task_id}...")
            agent_analysis = self.analyze_agent_2x2(task_id, task_data)
            all_analyses.append(agent_analysis)
            
            # Create plots for this agent
            for metric in ['cooperation_rate', 'avg_reward']:
                fig, quadrant_values = self.create_2x2_matrix_plot(agent_analysis, metric)
                
                # Save figure
                filename = f'task_{task_id}_{metric}_2x2.png'
                fig.savefig(self.fig_dir / filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved {filename}")
            
            # Extract summary statistics
            results_2x2 = agent_analysis['results_2x2']
            train_game = agent_analysis['train_game']
            train_opp_range = agent_analysis['train_opponent_range']
            
            for category, results_list in results_2x2.items():
                if results_list:
                    for result in results_list:
                        summary_data.append({
                            'task_id': task_id,
                            'train_game': train_game,
                            'train_opponent_range': train_opp_range,
                            'train_opponent_probs': str(agent_analysis['train_opponent_probs']),
                            'generalization_category': category,
                            'test_condition': result['test_condition'],
                            'test_game': result['test_game'],
                            'test_opponent_range': result['test_opponent_range'],
                            'cooperation_rate': result['cooperation_rate'],
                            'avg_reward': result['avg_reward'],
                            'n_opponents': result['n_opponents']
                        })
            
            print()
        
        # Save all analyses
        with open(self.data_dir / 'all_agent_2x2_analyses.pkl', 'wb') as f:
            pickle.dump(all_analyses, f)
        print(f"Saved complete analysis data\n")
        
        # Save summary table
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(self.table_dir / 'complete_2x2_summary.csv', index=False)
        print(f"Saved summary table with {len(df_summary)} entries\n")
        
        return all_analyses, df_summary
    
    def create_aggregate_comparison(self, df_summary):
        """Create aggregate comparisons across all agents"""
        print("Creating aggregate comparisons...")
        
        # Average performance by quadrant
        quadrant_stats = df_summary.groupby('generalization_category').agg({
            'cooperation_rate': ['mean', 'std', 'count'],
            'avg_reward': ['mean', 'std', 'count']
        }).round(3)
        
        quadrant_stats.to_csv(self.table_dir / 'quadrant_statistics.csv')
        print("  Saved quadrant_statistics.csv")
        
        # Performance drop from baseline
        baseline_data = df_summary[df_summary['generalization_category'] == 'same_task_same_opp']
        
        performance_drops = []
        for task_id in df_summary['task_id'].unique():
            task_baseline = baseline_data[baseline_data['task_id'] == task_id]
            task_data = df_summary[df_summary['task_id'] == task_id]
            
            if len(task_baseline) == 0:
                continue
            
            baseline_coop = task_baseline['cooperation_rate'].mean()
            baseline_reward = task_baseline['avg_reward'].mean()
            
            for category in ['same_task_diff_opp', 'diff_task_same_opp', 'diff_task_diff_opp']:
                cat_data = task_data[task_data['generalization_category'] == category]
                if len(cat_data) > 0:
                    performance_drops.append({
                        'task_id': task_id,
                        'train_game': task_data['train_game'].iloc[0],
                        'train_opponent_range': task_data['train_opponent_range'].iloc[0],
                        'category': category,
                        'baseline_coop': baseline_coop,
                        'test_coop': cat_data['cooperation_rate'].mean(),
                        'coop_drop': baseline_coop - cat_data['cooperation_rate'].mean(),
                        'baseline_reward': baseline_reward,
                        'test_reward': cat_data['avg_reward'].mean(),
                        'reward_drop': baseline_reward - cat_data['avg_reward'].mean()
                    })
        
        df_drops = pd.DataFrame(performance_drops)
        df_drops.to_csv(self.table_dir / 'performance_drops_from_baseline.csv', index=False)
        print("  Saved performance_drops_from_baseline.csv")
        
        # Visualization: Average performance by quadrant
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        category_order = ['same_task_same_opp', 'same_task_diff_opp', 
                         'diff_task_same_opp', 'diff_task_diff_opp']
        category_labels = ['Same Task\nSame Opp', 'Same Task\nDiff Opp',
                          'Diff Task\nSame Opp', 'Diff Task\nDiff Opp']
        
        for ax, metric, title in zip(axes, 
                                     ['cooperation_rate', 'avg_reward'],
                                     ['Cooperation Rate', 'Average Reward']):
            means = [df_summary[df_summary['generalization_category'] == cat][metric].mean() 
                    for cat in category_order]
            stds = [df_summary[df_summary['generalization_category'] == cat][metric].std() 
                   for cat in category_order]
            
            bars = ax.bar(range(4), means, yerr=stds, capsize=5, alpha=0.7,
                         color=['green', 'yellow', 'orange', 'red'])
            
            ax.set_xticks(range(4))
            ax.set_xticklabels(category_labels, fontsize=9)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(f'Average {title} by Generalization Category', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (m, s) in enumerate(zip(means, stds)):
                ax.text(i, m + s + 0.02, f'{m:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'aggregate_quadrant_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved aggregate_quadrant_comparison.png")
        
        # Statistical tests between quadrants
        print("\n  Running statistical tests...")
        stat_results = []
        
        for metric in ['cooperation_rate', 'avg_reward']:
            baseline = df_summary[df_summary['generalization_category'] == 'same_task_same_opp'][metric].dropna()
            
            for category in ['same_task_diff_opp', 'diff_task_same_opp', 'diff_task_diff_opp']:
                test_group = df_summary[df_summary['generalization_category'] == category][metric].dropna()
                
                if len(baseline) > 0 and len(test_group) > 0:
                    t_stat, p_val = stats.ttest_ind(baseline, test_group)
                    effect_size = (baseline.mean() - test_group.mean()) / np.sqrt((baseline.std()**2 + test_group.std()**2) / 2)
                    
                    stat_results.append({
                        'metric': metric,
                        'comparison': f'baseline_vs_{category}',
                        'baseline_mean': baseline.mean(),
                        'test_mean': test_group.mean(),
                        'mean_difference': baseline.mean() - test_group.mean(),
                        't_statistic': t_stat,
                        'p_value': p_val,
                        'cohens_d': effect_size,
                        'significant': p_val < 0.05
                    })
        
        df_stats = pd.DataFrame(stat_results)
        df_stats.to_csv(self.table_dir / 'statistical_comparisons.csv', index=False)
        print("  Saved statistical_comparisons.csv")
        
        return df_drops, df_stats
    
    def generate_summary_report(self, df_summary, df_drops, df_stats):
        """Generate executive summary of 2×2 analysis"""
        print("\nGenerating executive summary...")
        
        report = []
        report.append("# 2×2 Generalization Analysis - Executive Summary\n")
        report.append(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        report.append("## Research Question\n")
        report.append("How do agents trained on specific (task, opponent) combinations generalize to:\n")
        report.append("1. **Same task, same opponent** (baseline performance)\n")
        report.append("2. **Same task, different opponent** (opponent generalization)\n")
        report.append("3. **Different task, same opponent** (task generalization)\n")
        report.append("4. **Different task, different opponent** (full generalization)\n\n")
        
        report.append("## Overall Findings\n\n")
        
        # Quadrant performance
        for category in ['same_task_same_opp', 'same_task_diff_opp', 
                        'diff_task_same_opp', 'diff_task_diff_opp']:
            cat_data = df_summary[df_summary['generalization_category'] == category]
            coop_mean = cat_data['cooperation_rate'].mean()
            coop_std = cat_data['cooperation_rate'].std()
            reward_mean = cat_data['avg_reward'].mean()
            reward_std = cat_data['avg_reward'].std()
            n = len(cat_data)
            
            label = category.replace('_', ' ').title()
            report.append(f"### {label}\n")
            report.append(f"- **Cooperation Rate:** {coop_mean:.3f} ± {coop_std:.3f}\n")
            report.append(f"- **Average Reward:** {reward_mean:.3f} ± {reward_std:.3f}\n")
            report.append(f"- **N test conditions:** {n}\n\n")
        
        # Performance drops
        report.append("## Generalization Performance Drops\n\n")
        
        avg_drops = df_drops.groupby('category').agg({
            'coop_drop': ['mean', 'std'],
            'reward_drop': ['mean', 'std']
        }).round(3)
        
        report.append("Average performance drop from baseline:\n\n")
        for category in ['same_task_diff_opp', 'diff_task_same_opp', 'diff_task_diff_opp']:
            if category in avg_drops.index:
                coop_drop = avg_drops.loc[category, ('coop_drop', 'mean')]
                reward_drop = avg_drops.loc[category, ('reward_drop', 'mean')]
                label = category.replace('_', ' ').title()
                report.append(f"**{label}:**\n")
                report.append(f"- Cooperation drop: {coop_drop:.3f}\n")
                report.append(f"- Reward drop: {reward_drop:.3f}\n\n")
        
        # Statistical significance
        report.append("## Statistical Significance\n\n")
        report.append("Comparison vs baseline (same task, same opponent):\n\n")
        
        for _, row in df_stats.iterrows():
            if row['significant']:
                report.append(f"- **{row['comparison']}** ({row['metric']}): ")
                report.append(f"p = {row['p_value']:.4f}, Cohen's d = {row['cohens_d']:.3f}\n")
        
        report.append("\n## Key Insights\n\n")
        
        # Calculate which generalization is harder
        same_task_diff_opp = df_summary[df_summary['generalization_category'] == 'same_task_diff_opp']['cooperation_rate'].mean()
        diff_task_same_opp = df_summary[df_summary['generalization_category'] == 'diff_task_same_opp']['cooperation_rate'].mean()
        diff_task_diff_opp = df_summary[df_summary['generalization_category'] == 'diff_task_diff_opp']['cooperation_rate'].mean()
        
        if diff_task_same_opp < same_task_diff_opp:
            report.append("1. **Task generalization is harder than opponent generalization**\n")
            report.append(f"   - Different task, same opponent: {diff_task_same_opp:.3f}\n")
            report.append(f"   - Same task, different opponent: {same_task_diff_opp:.3f}\n\n")
        else:
            report.append("1. **Opponent generalization is harder than task generalization**\n")
            report.append(f"   - Same task, different opponent: {same_task_diff_opp:.3f}\n")
            report.append(f"   - Different task, same opponent: {diff_task_same_opp:.3f}\n\n")
        
        report.append("2. **Full generalization (different task AND opponent) shows ")
        baseline = df_summary[df_summary['generalization_category'] == 'same_task_same_opp']['cooperation_rate'].mean()
        full_gen_drop = baseline - diff_task_diff_opp
        report.append(f"largest performance drop:** {full_gen_drop:.3f}\n\n")
        
        # Best generalizing agents
        best_agents = df_drops.groupby('task_id').agg({
            'coop_drop': 'mean',
            'train_game': 'first',
            'train_opponent_range': 'first'
        }).sort_values('coop_drop')
        
        report.append("3. **Best generalizing training conditions:**\n")
        for task_id, row in best_agents.head(3).iterrows():
            game_abbrev = self.game_abbrev.get(row['train_game'], row['train_game'])
            report.append(f"   - Task {task_id}: {game_abbrev}, {row['train_opponent_range']} ")
            report.append(f"(avg drop: {row['coop_drop']:.3f})\n")
        
        report_text = ''.join(report)
        
        with open(self.output_dir / '2x2_analysis' / 'executive_summary.md', 'w') as f:
            f.write(report_text)
        
        print("  Saved executive_summary.md")
        return report_text
    
    def run_complete_analysis(self):
        """Execute complete 2×2 analysis"""
        print("="*70)
        print("2×2 GENERALIZATION ANALYSIS - INDIVIDUAL AGENTS")
        print("="*70)
        print()
        
        # Load data
        self.load_all_tasks()
        
        # Analyze all agents
        all_analyses, df_summary = self.analyze_all_agents()
        
        # Create aggregate comparisons
        df_drops, df_stats = self.create_aggregate_comparison(df_summary)
        
        # Generate summary
        summary = self.generate_summary_report(df_summary, df_drops, df_stats)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: {self.output_dir / '2x2_analysis'}")
        print(f"  - Individual agent figures: {self.fig_dir}")
        print(f"  - Summary tables: {self.table_dir}")
        print(f"  - Raw data: {self.data_dir}")
        print(f"\nGenerated {len(all_analyses)} individual agent analyses")
        print(f"\nExecutive Summary:\n")
        print(summary)


def main():
    """Main execution"""
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    experiment_dir = base_dir / 'experiments' / 'generalization_matrix_834222'
    output_dir = base_dir / 'experiments' / 'results'
    
    # Create analyzer and run
    analyzer = Agent2x2GeneralizationAnalyzer(experiment_dir, output_dir)
    analyzer.run_complete_analysis()


if __name__ == '__main__':
    main()
