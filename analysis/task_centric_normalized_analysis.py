"""
Task-Centric Generalization Analysis with Normalized Rewards
Analyzes generalization success FROM each training task TO other tasks/opponents
Includes reward normalization to make cross-game comparisons valid
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

class TaskCentricGeneralizationAnalyzer:
    """Analyzes which tasks are easiest/hardest to generalize FROM"""
    
    def __init__(self, experiment_dir, output_dir):
        self.experiment_dir = Path(experiment_dir)
        self.output_dir = Path(output_dir)
        self.tasks_data = {}
        self.games = ['prisoners-dilemma', 'stag-hunt', 'battle-of-sexes', 'hawk-dove']
        self.game_abbrev = {'prisoners-dilemma': 'PD', 'stag-hunt': 'SH', 
                           'battle-of-sexes': 'BoS', 'hawk-dove': 'HD'}
        
        # Payoff matrices for reward normalization (standard parametrization)
        # Format: [[CC, CD], [DC, DD]] where C=cooperate, D=defect
        self.payoff_matrices = {
            'prisoners-dilemma': {'R': 3, 'S': 0, 'T': 5, 'P': 1},  # CC, CD, DC, DD
            'stag-hunt': {'R': 4, 'S': 0, 'T': 3, 'P': 2},
            'battle-of-sexes': {'R': 3, 'S': 0, 'T': 2, 'P': 1},
            'hawk-dove': {'R': 2, 'S': 1, 'T': 3, 'P': 0}
        }
        
        # Calculate min/max possible rewards for normalization
        self.reward_bounds = {}
        for game, payoffs in self.payoff_matrices.items():
            # Min is worst outcome, max is best outcome
            all_payoffs = [payoffs['R'], payoffs['S'], payoffs['T'], payoffs['P']]
            self.reward_bounds[game] = {
                'min': min(all_payoffs),
                'max': max(all_payoffs),
                'range': max(all_payoffs) - min(all_payoffs)
            }
        
        # Create output directories
        self.fig_dir = self.output_dir / 'task_centric_analysis' / 'figures'
        self.table_dir = self.output_dir / 'task_centric_analysis' / 'tables'
        self.data_dir = self.output_dir / 'task_centric_analysis' / 'data'
        
        for dir_path in [self.fig_dir, self.table_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def normalize_reward(self, reward, game):
        """Normalize reward to [0,1] based on game's payoff structure"""
        if game not in self.reward_bounds:
            return reward  # Return unnormalized if game unknown
        
        bounds = self.reward_bounds[game]
        if bounds['range'] == 0:
            return 0.5  # If all rewards same, return middle value
        
        # Min-max normalization
        normalized = (reward - bounds['min']) / bounds['range']
        return np.clip(normalized, 0, 1)  # Ensure in [0,1] range
    
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
    
    def extract_generalization_by_training_task(self):
        """Extract generalization metrics organized by training task"""
        print("Extracting generalization metrics by training task...")
        
        gen_data = []
        
        for task_id, data in self.tasks_data.items():
            training_condition = data.get('training_condition', {})
            eval_results = data.get('evaluation_results', {})
            
            train_game = training_condition.get('game', 'unknown')
            train_opp_range = training_condition.get('opponent_range', 'unknown')
            train_opp_probs = training_condition.get('opponent_probs', [])
            
            # Process each test condition
            for test_cond_name, test_cond_data in eval_results.items():
                if not isinstance(test_cond_data, dict):
                    continue
                
                test_game = test_cond_data.get('game', 'unknown')
                test_opp_range = test_cond_data.get('opponent_range', 'unknown')
                condition_type = test_cond_data.get('condition', 'unknown')
                
                # Get results for each opponent in this test condition
                opp_results = test_cond_data.get('results', {})
                
                for opp_key, opp_data in opp_results.items():
                    if not isinstance(opp_data, dict):
                        continue
                    
                    # Extract opponent defection probability from key (e.g., "prob_opponent_0_p0.30")
                    try:
                        opp_defect_prob = float(opp_key.split('_p')[-1])
                    except:
                        opp_defect_prob = np.nan
                    
                    coop_rate = opp_data.get('cooperation_rate')
                    avg_reward = opp_data.get('average_reward')
                    opp_coop_rate = opp_data.get('opponent_cooperation_rate')
                    
                    # Normalize reward
                    normalized_reward = self.normalize_reward(avg_reward, test_game) if avg_reward is not None else np.nan
                    
                    # Calculate behavior prediction error
                    # Agent's cooperation vs opponent's cooperation
                    if coop_rate is not None and opp_coop_rate is not None:
                        behavior_misprediction = abs(coop_rate - opp_coop_rate)
                    else:
                        behavior_misprediction = np.nan
                    
                    gen_data.append({
                        'task_id': task_id,
                        'train_game': train_game,
                        'train_opponent_range': train_opp_range,
                        'train_opponent_probs': str(train_opp_probs),
                        'test_condition_name': test_cond_name,
                        'test_game': test_game,
                        'test_opponent_range': test_opp_range,
                        'test_opponent_defect_prob': opp_defect_prob,
                        'condition_type': condition_type,
                        'agent_cooperation_rate': coop_rate,
                        'opponent_cooperation_rate': opp_coop_rate,
                        'behavior_misprediction': behavior_misprediction,
                        'raw_reward': avg_reward,
                        'normalized_reward': normalized_reward,
                        'same_game': train_game == test_game,
                        'same_opponent_range': train_opp_range == test_opp_range
                    })
        
        df = pd.DataFrame(gen_data)
        print(f"  Extracted {len(df)} test condition results\n")
        return df
    
    def analyze_generalization_from_each_task(self, df):
        """Analyze generalization success FROM each training game"""
        print("Analyzing generalization from each training game...\n")
        
        results_by_game = {}
        
        for train_game in self.games:
            game_data = df[df['train_game'] == train_game].copy()
            
            if len(game_data) == 0:
                continue
            
            # Calculate success metrics for different generalization scenarios
            scenarios = {
                'baseline': game_data[game_data['same_game'] & game_data['same_opponent_range']],
                'same_task_new_opp': game_data[game_data['same_game'] & ~game_data['same_opponent_range']],
                'new_task_same_opp': game_data[~game_data['same_game'] & game_data['same_opponent_range']],
                'new_task_new_opp': game_data[~game_data['same_game'] & ~game_data['same_opponent_range']]
            }
            
            metrics = {}
            for scenario_name, scenario_data in scenarios.items():
                if len(scenario_data) > 0:
                    metrics[scenario_name] = {
                        'n': len(scenario_data),
                        'raw_reward_mean': scenario_data['raw_reward'].mean(),
                        'raw_reward_std': scenario_data['raw_reward'].std(),
                        'normalized_reward_mean': scenario_data['normalized_reward'].mean(),
                        'normalized_reward_std': scenario_data['normalized_reward'].std(),
                        'cooperation_mean': scenario_data['agent_cooperation_rate'].mean(),
                        'cooperation_std': scenario_data['agent_cooperation_rate'].std(),
                        'misprediction_mean': scenario_data['behavior_misprediction'].mean(),
                        'misprediction_std': scenario_data['behavior_misprediction'].std()
                    }
                else:
                    metrics[scenario_name] = {
                        'n': 0,
                        'raw_reward_mean': np.nan,
                        'raw_reward_std': np.nan,
                        'normalized_reward_mean': np.nan,
                        'normalized_reward_std': np.nan,
                        'cooperation_mean': np.nan,
                        'cooperation_std': np.nan,
                        'misprediction_mean': np.nan,
                        'misprediction_std': np.nan
                    }
            
            # Calculate generalization ratios (relative to baseline)
            baseline_norm_reward = metrics['baseline']['normalized_reward_mean']
            baseline_mispred = metrics['baseline']['misprediction_mean']
            
            generalization_ratios = {}
            for scenario in ['same_task_new_opp', 'new_task_same_opp', 'new_task_new_opp']:
                if not np.isnan(baseline_norm_reward) and baseline_norm_reward != 0:
                    generalization_ratios[f'{scenario}_reward_ratio'] = \
                        metrics[scenario]['normalized_reward_mean'] / baseline_norm_reward
                else:
                    generalization_ratios[f'{scenario}_reward_ratio'] = np.nan
                
                # Misprediction increase (lower is better)
                if not np.isnan(baseline_mispred):
                    generalization_ratios[f'{scenario}_mispred_increase'] = \
                        metrics[scenario]['misprediction_mean'] - baseline_mispred
                else:
                    generalization_ratios[f'{scenario}_mispred_increase'] = np.nan
            
            results_by_game[train_game] = {
                'metrics': metrics,
                'ratios': generalization_ratios
            }
            
            # Print summary
            game_abbrev = self.game_abbrev[train_game]
            print(f"=== {game_abbrev} ({train_game}) ===")
            print(f"Baseline (same task, same opponent):")
            print(f"  Normalized Reward: {metrics['baseline']['normalized_reward_mean']:.3f} ± {metrics['baseline']['normalized_reward_std']:.3f}")
            print(f"  Behavior Misprediction: {metrics['baseline']['misprediction_mean']:.3f} ± {metrics['baseline']['misprediction_std']:.3f}")
            print(f"\nGeneralization to same task, new opponent:")
            print(f"  Reward Ratio: {generalization_ratios['same_task_new_opp_reward_ratio']:.3f}")
            print(f"  Misprediction Increase: {generalization_ratios['same_task_new_opp_mispred_increase']:.3f}")
            print(f"\nGeneralization to new task, same opponent:")
            print(f"  Reward Ratio: {generalization_ratios['new_task_same_opp_reward_ratio']:.3f}")
            print(f"  Misprediction Increase: {generalization_ratios['new_task_same_opp_mispred_increase']:.3f}")
            print(f"\nGeneralization to new task, new opponent:")
            print(f"  Reward Ratio: {generalization_ratios['new_task_new_opp_reward_ratio']:.3f}")
            print(f"  Misprediction Increase: {generalization_ratios['new_task_new_opp_mispred_increase']:.3f}")
            print()
        
        return results_by_game
    
    def create_task_comparison_plots(self, results_by_game):
        """Create comparison plots across training tasks"""
        print("Creating task comparison visualizations...")
        
        # Prepare data for plotting
        games_order = ['prisoners-dilemma', 'stag-hunt', 'battle-of-sexes', 'hawk-dove']
        games_in_data = [g for g in games_order if g in results_by_game]
        game_labels = [self.game_abbrev[g] for g in games_in_data]
        
        scenarios = ['same_task_new_opp', 'new_task_same_opp', 'new_task_new_opp']
        scenario_labels = ['Same Task\nNew Opponent', 'New Task\nSame Opponent', 'New Task\nNew Opponent']
        
        # Figure 1: Reward Ratios
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Subplot 1: Normalized Reward Ratios
        ax = axes[0]
        x = np.arange(len(games_in_data))
        width = 0.25
        
        for i, (scenario, label) in enumerate(zip(scenarios, scenario_labels)):
            ratios = [results_by_game[g]['ratios'][f'{scenario}_reward_ratio'] for g in games_in_data]
            ax.bar(x + i*width, ratios, width, label=label, alpha=0.8)
        
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Baseline (ratio=1.0)')
        ax.set_xlabel('Training Game', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Reward Ratio\n(Test / Baseline)', fontsize=11)
        ax.set_title('Generalization Success: Normalized Reward Retention', fontsize=13, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(game_labels)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (scenario, label) in enumerate(zip(scenarios, scenario_labels)):
            ratios = [results_by_game[g]['ratios'][f'{scenario}_reward_ratio'] for g in games_in_data]
            for j, ratio in enumerate(ratios):
                if not np.isnan(ratio):
                    ax.text(j + i*width, ratio + 0.02, f'{ratio:.2f}', 
                           ha='center', va='bottom', fontsize=8)
        
        # Subplot 2: Behavior Misprediction Increase
        ax = axes[1]
        
        for i, (scenario, label) in enumerate(zip(scenarios, scenario_labels)):
            mispred_increases = [results_by_game[g]['ratios'][f'{scenario}_mispred_increase'] for g in games_in_data]
            ax.bar(x + i*width, mispred_increases, width, label=label, alpha=0.8)
        
        ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1, label='No increase from baseline')
        ax.set_xlabel('Training Game', fontsize=12, fontweight='bold')
        ax.set_ylabel('Behavior Misprediction Increase\n(Test - Baseline)', fontsize=11)
        ax.set_title('Generalization Challenge: Behavior Prediction Error', fontsize=13, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(game_labels)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (scenario, label) in enumerate(zip(scenarios, scenario_labels)):
            mispred_increases = [results_by_game[g]['ratios'][f'{scenario}_mispred_increase'] for g in games_in_data]
            for j, increase in enumerate(mispred_increases):
                if not np.isnan(increase):
                    ax.text(j + i*width, increase + 0.005, f'{increase:.3f}', 
                           ha='center', va='bottom', fontsize=7, rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'task_generalization_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved task_generalization_comparison.png")
        
        # Figure 2: Ranking tasks by generalization ease
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Calculate overall generalization scores
        gen_scores = {}
        for game in games_in_data:
            # Average reward ratio across all generalization scenarios
            reward_ratios = [results_by_game[game]['ratios'][f'{s}_reward_ratio'] for s in scenarios]
            avg_reward_ratio = np.nanmean(reward_ratios)
            
            # Average misprediction increase
            mispred_increases = [results_by_game[game]['ratios'][f'{s}_mispred_increase'] for s in scenarios]
            avg_mispred_increase = np.nanmean(mispred_increases)
            
            gen_scores[game] = {
                'reward_ratio': avg_reward_ratio,
                'mispred_increase': avg_mispred_increase,
                'combined_score': avg_reward_ratio - avg_mispred_increase  # Higher is better
            }
        
        # Rank by reward retention
        ax = axes[0, 0]
        sorted_by_reward = sorted(gen_scores.items(), key=lambda x: x[1]['reward_ratio'], reverse=True)
        games_sorted = [self.game_abbrev[g] for g, _ in sorted_by_reward]
        scores = [s['reward_ratio'] for _, s in sorted_by_reward]
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(games_sorted)))
        
        ax.barh(range(len(games_sorted)), scores, color=colors, alpha=0.8)
        ax.set_yticks(range(len(games_sorted)))
        ax.set_yticklabels(games_sorted)
        ax.set_xlabel('Average Normalized Reward Ratio', fontsize=11)
        ax.set_title('Tasks Ranked by Reward Retention\n(Easier to Generalize FROM → Higher)', 
                    fontsize=12, fontweight='bold')
        ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, score in enumerate(scores):
            ax.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=10)
        
        # Rank by misprediction (lower is better, so ascending)
        ax = axes[0, 1]
        sorted_by_mispred = sorted(gen_scores.items(), key=lambda x: x[1]['mispred_increase'])
        games_sorted = [self.game_abbrev[g] for g, _ in sorted_by_mispred]
        scores = [s['mispred_increase'] for _, s in sorted_by_mispred]
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(games_sorted)))
        
        ax.barh(range(len(games_sorted)), scores, color=colors, alpha=0.8)
        ax.set_yticks(range(len(games_sorted)))
        ax.set_yticklabels(games_sorted)
        ax.set_xlabel('Average Behavior Misprediction Increase', fontsize=11)
        ax.set_title('Tasks Ranked by Behavior Prediction Accuracy\n(Easier to Generalize FROM → Lower)', 
                    fontsize=12, fontweight='bold')
        ax.axvline(x=0.0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, score in enumerate(scores):
            ax.text(score + 0.002 if score >= 0 else score - 0.002, i, f'{score:.3f}', 
                   va='center', ha='left' if score >= 0 else 'right', fontsize=10)
        
        # Combined ranking
        ax = axes[1, 0]
        sorted_by_combined = sorted(gen_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        games_sorted = [self.game_abbrev[g] for g, _ in sorted_by_combined]
        scores = [s['combined_score'] for _, s in sorted_by_combined]
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(games_sorted)))
        
        ax.barh(range(len(games_sorted)), scores, color=colors, alpha=0.8)
        ax.set_yticks(range(len(games_sorted)))
        ax.set_yticklabels(games_sorted)
        ax.set_xlabel('Combined Generalization Score\n(Reward Ratio - Misprediction)', fontsize=11)
        ax.set_title('Overall Task Generalization Ranking\n(Higher = Easier to Generalize FROM)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, score in enumerate(scores):
            ax.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=10)
        
        # Scatter: Reward vs Misprediction
        ax = axes[1, 1]
        for game in games_in_data:
            ax.scatter(gen_scores[game]['mispred_increase'], 
                      gen_scores[game]['reward_ratio'],
                      s=200, alpha=0.7, label=self.game_abbrev[game])
            ax.annotate(self.game_abbrev[game], 
                       (gen_scores[game]['mispred_increase'], gen_scores[game]['reward_ratio']),
                       xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
        
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Behavior Misprediction Increase (Lower = Better)', fontsize=11)
        ax.set_ylabel('Reward Ratio (Higher = Better)', fontsize=11)
        ax.set_title('Generalization Trade-off Space\n(Top-Left = Best Generalization)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add quadrant labels
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(xlim[0] + (xlim[1]-xlim[0])*0.05, ylim[1] - (ylim[1]-ylim[0])*0.05, 
               'BEST', fontsize=12, fontweight='bold', color='green', alpha=0.5)
        ax.text(xlim[1] - (xlim[1]-xlim[0])*0.08, ylim[0] + (ylim[1]-ylim[0])*0.05, 
               'WORST', fontsize=12, fontweight='bold', color='red', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'task_generalization_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved task_generalization_rankings.png")
        
        return gen_scores
    
    def create_summary_tables(self, results_by_game, gen_scores, df):
        """Create summary tables"""
        print("Creating summary tables...")
        
        # Table 1: Detailed metrics by game and scenario
        table_data = []
        for game in self.games:
            if game not in results_by_game:
                continue
            
            metrics = results_by_game[game]['metrics']
            ratios = results_by_game[game]['ratios']
            
            for scenario in ['baseline', 'same_task_new_opp', 'new_task_same_opp', 'new_task_new_opp']:
                row = {
                    'training_game': game,
                    'game_abbrev': self.game_abbrev[game],
                    'scenario': scenario,
                    'n_conditions': metrics[scenario]['n'],
                    'raw_reward_mean': metrics[scenario]['raw_reward_mean'],
                    'raw_reward_std': metrics[scenario]['raw_reward_std'],
                    'normalized_reward_mean': metrics[scenario]['normalized_reward_mean'],
                    'normalized_reward_std': metrics[scenario]['normalized_reward_std'],
                    'cooperation_mean': metrics[scenario]['cooperation_mean'],
                    'cooperation_std': metrics[scenario]['cooperation_std'],
                    'misprediction_mean': metrics[scenario]['misprediction_mean'],
                    'misprediction_std': metrics[scenario]['misprediction_std']
                }
                
                # Add ratios for non-baseline scenarios
                if scenario != 'baseline':
                    row['reward_ratio'] = ratios[f'{scenario}_reward_ratio']
                    row['mispred_increase'] = ratios[f'{scenario}_mispred_increase']
                
                table_data.append(row)
        
        df_detailed = pd.DataFrame(table_data)
        df_detailed.to_csv(self.table_dir / 'task_generalization_detailed.csv', index=False)
        print("  Saved task_generalization_detailed.csv")
        
        # Table 2: Rankings
        ranking_data = []
        for game, scores in gen_scores.items():
            ranking_data.append({
                'game': game,
                'game_abbrev': self.game_abbrev[game],
                'avg_reward_ratio': scores['reward_ratio'],
                'avg_mispred_increase': scores['mispred_increase'],
                'combined_score': scores['combined_score']
            })
        
        df_rankings = pd.DataFrame(ranking_data)
        df_rankings = df_rankings.sort_values('combined_score', ascending=False)
        df_rankings['rank'] = range(1, len(df_rankings) + 1)
        df_rankings.to_csv(self.table_dir / 'task_generalization_rankings.csv', index=False)
        print("  Saved task_generalization_rankings.csv")
        
        # Table 3: Complete raw data
        df.to_csv(self.table_dir / 'complete_generalization_data.csv', index=False)
        print("  Saved complete_generalization_data.csv")
        
        return df_detailed, df_rankings
    
    def generate_report(self, results_by_game, gen_scores, df_rankings):
        """Generate executive summary"""
        print("\nGenerating executive summary...")
        
        report = []
        report.append("# Task-Centric Generalization Analysis with Normalized Rewards\n")
        report.append(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        report.append("## Research Question\n")
        report.append("From which training tasks is it **easiest** vs **hardest** to generalize?\n")
        report.append("Analysis considers:\n")
        report.append("- **Reward retention** (normalized across games for fair comparison)\n")
        report.append("- **Behavior prediction accuracy** (agent-opponent misprediction)\n\n")
        
        report.append("## Reward Normalization\n")
        report.append("To enable fair comparison across games with different payoff structures:\n")
        for game in self.games:
            if game in self.reward_bounds:
                bounds = self.reward_bounds[game]
                report.append(f"- **{self.game_abbrev[game]}**: min={bounds['min']}, max={bounds['max']}, range={bounds['range']}\n")
        report.append("\nNormalized reward = (raw_reward - min) / range → [0, 1]\n\n")
        
        report.append("## Overall Rankings\n\n")
        report.append("### Easiest to Generalize FROM (Best → Worst):\n")
        for _, row in df_rankings.iterrows():
            report.append(f"{row['rank']}. **{row['game_abbrev']} ({row['game']})**\n")
            report.append(f"   - Avg Reward Ratio: {row['avg_reward_ratio']:.3f}\n")
            report.append(f"   - Avg Misprediction Increase: {row['avg_mispred_increase']:.3f}\n")
            report.append(f"   - Combined Score: {row['combined_score']:.3f}\n\n")
        
        report.append("## Detailed Findings by Training Game\n\n")
        
        for game in self.games:
            if game not in results_by_game:
                continue
            
            game_abbrev = self.game_abbrev[game]
            metrics = results_by_game[game]['metrics']
            ratios = results_by_game[game]['ratios']
            
            report.append(f"### {game_abbrev} ({game})\n\n")
            report.append(f"**Baseline Performance (same task, same opponent):**\n")
            report.append(f"- Normalized Reward: {metrics['baseline']['normalized_reward_mean']:.3f} ± {metrics['baseline']['normalized_reward_std']:.3f}\n")
            report.append(f"- Behavior Misprediction: {metrics['baseline']['misprediction_mean']:.3f} ± {metrics['baseline']['misprediction_std']:.3f}\n\n")
            
            report.append(f"**Generalization to Same Task, New Opponent:**\n")
            report.append(f"- Reward Ratio: {ratios['same_task_new_opp_reward_ratio']:.3f} ({(ratios['same_task_new_opp_reward_ratio']-1)*100:.1f}% change)\n")
            report.append(f"- Misprediction Increase: {ratios['same_task_new_opp_mispred_increase']:.3f}\n\n")
            
            report.append(f"**Generalization to New Task, Same Opponent:**\n")
            report.append(f"- Reward Ratio: {ratios['new_task_same_opp_reward_ratio']:.3f} ({(ratios['new_task_same_opp_reward_ratio']-1)*100:.1f}% change)\n")
            report.append(f"- Misprediction Increase: {ratios['new_task_same_opp_mispred_increase']:.3f}\n\n")
            
            report.append(f"**Generalization to New Task, New Opponent:**\n")
            report.append(f"- Reward Ratio: {ratios['new_task_new_opp_reward_ratio']:.3f} ({(ratios['new_task_new_opp_reward_ratio']-1)*100:.1f}% change)\n")
            report.append(f"- Misprediction Increase: {ratios['new_task_new_opp_mispred_increase']:.3f}\n\n")
            report.append("---\n\n")
        
        report.append("## Key Insights\n\n")
        
        # Best and worst
        best_game = df_rankings.iloc[0]
        worst_game = df_rankings.iloc[-1]
        
        report.append(f"1. **Easiest task to generalize FROM:** {best_game['game_abbrev']} (combined score: {best_game['combined_score']:.3f})\n")
        report.append(f"   - Retains {best_game['avg_reward_ratio']:.1%} of baseline reward\n")
        report.append(f"   - Minimal behavior misprediction increase: {best_game['avg_mispred_increase']:.3f}\n\n")
        
        report.append(f"2. **Hardest task to generalize FROM:** {worst_game['game_abbrev']} (combined score: {worst_game['combined_score']:.3f})\n")
        report.append(f"   - Retains only {worst_game['avg_reward_ratio']:.1%} of baseline reward\n")
        report.append(f"   - Larger behavior misprediction increase: {worst_game['avg_mispred_increase']:.3f}\n\n")
        
        # Compare generalization types
        avg_same_task = np.nanmean([results_by_game[g]['ratios']['same_task_new_opp_reward_ratio'] 
                                     for g in results_by_game.keys()])
        avg_new_task = np.nanmean([results_by_game[g]['ratios']['new_task_same_opp_reward_ratio'] 
                                   for g in results_by_game.keys()])
        
        if avg_new_task < avg_same_task:
            report.append(f"3. **Task generalization is harder than opponent generalization**\n")
            report.append(f"   - New task, same opponent: {avg_new_task:.1%} reward retention\n")
            report.append(f"   - Same task, new opponent: {avg_same_task:.1%} reward retention\n\n")
        else:
            report.append(f"3. **Opponent generalization is harder than task generalization**\n")
            report.append(f"   - Same task, new opponent: {avg_same_task:.1%} reward retention\n")
            report.append(f"   - New task, same opponent: {avg_new_task:.1%} reward retention\n\n")
        
        report_text = ''.join(report)
        
        with open(self.output_dir / 'task_centric_analysis' / 'executive_summary.md', 'w') as f:
            f.write(report_text)
        
        print("  Saved executive_summary.md")
        return report_text
    
    def run_complete_analysis(self):
        """Execute complete task-centric analysis"""
        print("="*70)
        print("TASK-CENTRIC GENERALIZATION ANALYSIS (NORMALIZED REWARDS)")
        print("="*70)
        print()
        
        # Load data
        self.load_all_tasks()
        
        # Extract generalization metrics
        df = self.extract_generalization_by_training_task()
        
        # Analyze generalization from each task
        results_by_game = self.analyze_generalization_from_each_task(df)
        
        # Create visualizations
        gen_scores = self.create_task_comparison_plots(results_by_game)
        
        # Create summary tables
        df_detailed, df_rankings = self.create_summary_tables(results_by_game, gen_scores, df)
        
        # Generate report
        summary = self.generate_report(results_by_game, gen_scores, df_rankings)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: {self.output_dir / 'task_centric_analysis'}")
        print(f"  - Figures: {self.fig_dir}")
        print(f"  - Tables: {self.table_dir}")
        print(f"\nExecutive Summary:\n")
        print(summary)


def main():
    """Main execution"""
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    experiment_dir = base_dir / 'experiments' / 'generalization_matrix_834222'
    output_dir = base_dir / 'experiments' / 'results'
    
    # Create analyzer and run
    analyzer = TaskCentricGeneralizationAnalyzer(experiment_dir, output_dir)
    analyzer.run_complete_analysis()


if __name__ == '__main__':
    main()
