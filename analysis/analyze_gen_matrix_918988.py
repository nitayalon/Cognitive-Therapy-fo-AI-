"""
Comprehensive Analysis of Generalization Matrix Experiment 918988
Analyzes training (918988) and testing (918989) results with 5 seeds
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
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

class GeneralizationMatrixAnalyzer918988:
    """Comprehensive analyzer for experiment 918988/918989"""
    
    def __init__(self, train_dir, test_dir, output_dir):
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.fig_dir = self.output_dir / 'figures'
        self.table_dir = self.output_dir / 'tables'
        self.data_dir = self.output_dir / 'data'
        
        for dir_path in [self.fig_dir, self.table_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.training_data = {}
        self.test_data = {}
        
    def load_training_data(self):
        """Load all training data from condition_X_seed_Y directories"""
        print("="*80)
        print("LOADING TRAINING DATA")
        print("="*80)
        
        training_base = self.train_dir / 'training'
        condition_dirs = sorted(training_base.glob('condition_*_seed_*'))
        
        print(f"Found {len(condition_dirs)} training condition directories")
        
        for cond_dir in condition_dirs:
            # Parse condition and seed from directory name
            parts = cond_dir.name.split('_')
            condition_id = int(parts[1])
            seed_id = int(parts[3])
            
            # Find the experiment directory inside
            exp_dirs = list(cond_dir.glob('generalization_matrix_task_*'))
            if not exp_dirs:
                print(f"  Warning: No experiment directory in {cond_dir.name}")
                continue
            
            exp_dir = exp_dirs[0]
            
            # Load results
            results_dir = exp_dir / 'results'
            results_files = list(results_dir.glob('training_task_*_results.pkl'))
            
            if results_files:
                results_file = results_files[0]
                try:
                    with open(results_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Load JSON report for additional info
                    json_files = list(results_dir.glob('training_task_*_report.json'))
                    if json_files:
                        with open(json_files[0], 'r') as f:
                            report = json.load(f)
                            data['report'] = report
                    
                    # Store with key (condition_id, seed_id)
                    key = (condition_id, seed_id)
                    self.training_data[key] = data
                    
                    if len(self.training_data) % 15 == 0:
                        print(f"  Loaded {len(self.training_data)} training runs...")
                    
                except Exception as e:
                    print(f"  Error loading {results_file}: {e}")
        
        print(f"\nSuccessfully loaded {len(self.training_data)} training runs")
        print(f"Conditions: {sorted(set(k[0] for k in self.training_data.keys()))}")
        print(f"Seeds: {sorted(set(k[1] for k in self.training_data.keys()))}")
        
        return self.training_data
    
    def load_test_data(self):
        """Load all test data from model_X_test_cond_Y directories"""
        print("\n" + "="*80)
        print("LOADING TEST DATA")
        print("="*80)
        
        testing_base = self.test_dir / 'testing'
        test_dirs = sorted(testing_base.glob('model_*_test_cond_*'))
        
        print(f"Found {len(test_dirs)} test directories")
        
        for test_dir in test_dirs:
            # Parse model and test condition from directory name
            parts = test_dir.name.split('_')
            model_id = int(parts[1])
            test_cond = int(parts[4])
            
            # Find experiment directory inside
            exp_dirs = list(test_dir.glob('generalization_matrix_task_*'))
            if not exp_dirs:
                continue
            
            exp_dir = exp_dirs[0]
            
            # Load results
            results_dir = exp_dir / 'results'
            results_files = list(results_dir.glob('eval_model_*_results.pkl'))
            
            if results_files:
                results_file = results_files[0]
                try:
                    with open(results_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Load JSON report
                    json_files = list(results_dir.glob('eval_model_*_report.json'))
                    if json_files:
                        with open(json_files[0], 'r') as f:
                            report = json.load(f)
                            data['report'] = report
                    
                    # Store with key (model_id, test_cond)
                    key = (model_id, test_cond)
                    self.test_data[key] = data
                    
                    if len(self.test_data) % 100 == 0:
                        print(f"  Loaded {len(self.test_data)} test runs...")
                    
                except Exception as e:
                    if len(self.test_data) < 10:  # Only show first few errors
                        print(f"  Error loading {results_file}: {e}")
        
        print(f"\nSuccessfully loaded {len(self.test_data)} test runs")
        
        return self.test_data
    
    def extract_training_summary(self):
        """Create summary DataFrame of training results"""
        print("\n" + "="*80)
        print("EXTRACTING TRAINING SUMMARY")
        print("="*80)
        
        rows = []
        
        for (condition_id, seed_id), data in self.training_data.items():
            # Get training info
            training_info = data.get('training_condition', {})
            training_results = data.get('training_results', {})
            
            # Get final metrics
            epoch_results = training_results.get('epoch_results', [])
            if epoch_results:
                final_epoch = epoch_results[-1]
                final_metrics = training_results.get('final_metrics', {})
                
                row = {
                    'condition_id': condition_id,
                    'seed_id': seed_id,
                    'model_id': condition_id * 5 + seed_id,  # Unique model ID
                    'train_game': training_info.get('game', 'unknown'),
                    'train_opponent_probs': str(training_info.get('opponent_probs', [])),
                    'total_epochs': len(epoch_results),
                    'final_total_loss': final_epoch.get('total_loss', np.nan),
                    'final_rl_loss': final_epoch.get('rl_loss', np.nan),
                    'final_tom_loss': final_epoch.get('opponent_policy_loss', np.nan),
                    'final_cooperation': final_epoch.get('epoch_average_cooperation_rate', np.nan),
                    'converged': final_metrics.get('convergence_info', {}).get('converged', False),
                    'session_stats': str(final_epoch.get('session_stats', {}))
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        print(f"Created training summary with {len(df)} entries")
        print(f"\nColumns: {df.columns.tolist()}")
        
        # Save
        df.to_csv(self.data_dir / 'training_summary.csv', index=False)
        
        return df
    
    def extract_generalization_matrix(self):
        """Create generalization matrix from test results"""
        print("\n" + "="*80)
        print("EXTRACTING GENERALIZATION MATRIX")
        print("="*80)
        
        rows = []
        
        for (model_id, test_cond), data in self.test_data.items():
            # Get evaluation results
            eval_results = data.get('evaluation_results', {})
            
            # The key format is 'condition_X'
            cond_key = f'condition_{test_cond}'
            
            if cond_key in eval_results:
                cond_data = eval_results[cond_key]
                test_info = cond_data.get('test_condition', {})
                results = cond_data.get('results', {})
                
                # Aggregate metrics across all opponents in this test condition
                avg_rewards = []
                coop_rates = []
                
                for opp_key, opp_data in results.items():
                    if isinstance(opp_data, dict):
                        avg_rewards.append(opp_data.get('average_reward', np.nan))
                        coop_rates.append(opp_data.get('cooperation_rate', np.nan))
                
                # Calculate mean metrics
                avg_reward = np.nanmean(avg_rewards) if avg_rewards else np.nan
                cooperation_rate = np.nanmean(coop_rates) if coop_rates else np.nan
                
                row = {
                    'model_id': model_id,
                    'train_condition': model_id // 5,  # Reverse calculation
                    'seed_id': model_id % 5,
                    'test_condition': test_cond,
                    'test_game': test_info.get('game', 'unknown'),
                    'test_opponent_probs': str(test_info.get('opponent_probs', [])),
                    'average_reward': avg_reward,
                    'cooperation_rate': cooperation_rate,
                    'num_opponents_tested': len(avg_rewards)
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        print(f"Created generalization matrix with {len(df)} entries")
        print(f"Unique models: {df['model_id'].nunique()}")
        print(f"Unique test conditions: {df['test_condition'].nunique()}")
        
        # Save
        df.to_csv(self.data_dir / 'generalization_matrix.csv', index=False)
        
        return df
    
    def plot_training_convergence(self, df_training):
        """Plot training convergence statistics"""
        print("\n" + "="*80)
        print("PLOTTING TRAINING CONVERGENCE")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Epochs to convergence by condition
        ax = axes[0, 0]
        summary = df_training.groupby('condition_id')['total_epochs'].agg(['mean', 'std'])
        x = summary.index
        ax.bar(x, summary['mean'], yerr=summary['std'], alpha=0.7, capsize=5)
        ax.set_xlabel('Training Condition')
        ax.set_ylabel('Epochs to Convergence')
        ax.set_title('Training Convergence by Condition')
        ax.grid(True, alpha=0.3)
        
        # 2. Final losses
        ax = axes[0, 1]
        loss_cols = ['final_rl_loss', 'final_tom_loss']
        df_training.groupby('condition_id')[loss_cols].mean().plot(kind='bar', ax=ax, alpha=0.7)
        ax.set_xlabel('Training Condition')
        ax.set_ylabel('Final Loss')
        ax.set_title('Final Training Losses by Condition')
        ax.legend(['RL Loss', 'ToM Loss'])
        ax.grid(True, alpha=0.3)
        
        # 3. Final cooperation rates
        ax = axes[1, 0]
        summary = df_training.groupby('condition_id')['final_cooperation'].agg(['mean', 'std'])
        x = summary.index
        ax.bar(x, summary['mean'], yerr=summary['std'], alpha=0.7, capsize=5)
        ax.set_xlabel('Training Condition')
        ax.set_ylabel('Cooperation Rate')
        ax.set_title('Final Cooperation Rate by Condition')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # 4. Seed consistency
        ax = axes[1, 1]
        pivot = df_training.pivot(index='condition_id', columns='seed_id', values='final_total_loss')
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
        ax.set_title('Final Total Loss: Seeds vs Conditions')
        ax.set_xlabel('Seed ID')
        ax.set_ylabel('Condition ID')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'training_convergence.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {self.fig_dir / 'training_convergence.png'}")
        plt.close()
    
    def plot_generalization_heatmap(self, df_gen):
        """Plot generalization performance as heatmap"""
        print("\n" + "="*80)
        print("PLOTTING GENERALIZATION HEATMAP")
        print("="*80)
        
        # Average across seeds
        summary = df_gen.groupby(['train_condition', 'test_condition']).agg({
            'average_reward': ['mean', 'std'],
            'cooperation_rate': ['mean', 'std']
        }).reset_index()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Average Reward Heatmap
        ax = axes[0]
        pivot_reward = summary.pivot(
            index='train_condition', 
            columns='test_condition', 
            values=('average_reward', 'mean')
        )
        sns.heatmap(pivot_reward, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=pivot_reward.mean().mean(), ax=ax, cbar_kws={'label': 'Average Reward'})
        ax.set_title('Generalization Matrix: Average Reward')
        ax.set_xlabel('Test Condition')
        ax.set_ylabel('Train Condition')
        
        # 2. Cooperation Rate Heatmap
        ax = axes[1]
        pivot_coop = summary.pivot(
            index='train_condition', 
            columns='test_condition', 
            values=('cooperation_rate', 'mean')
        )
        sns.heatmap(pivot_coop, annot=True, fmt='.2f', cmap='Blues', 
                   vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Cooperation Rate'})
        ax.set_title('Generalization Matrix: Cooperation Rate')
        ax.set_xlabel('Test Condition')
        ax.set_ylabel('Train Condition')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'generalization_heatmap.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {self.fig_dir / 'generalization_heatmap.png'}")
        plt.close()
    
    def compute_statistics(self, df_training, df_gen):
        """Compute and save summary statistics"""
        print("\n" + "="*80)
        print("COMPUTING STATISTICS")
        print("="*80)
        
        stats = {}
        
        # Training statistics
        stats['training'] = {
            'num_conditions': df_training['condition_id'].nunique(),
            'num_seeds': df_training['seed_id'].nunique(),
            'total_models': len(df_training),
            'avg_epochs': df_training['total_epochs'].mean(),
            'std_epochs': df_training['total_epochs'].std(),
            'avg_final_loss': df_training['final_total_loss'].mean(),
            'avg_final_cooperation': df_training['final_cooperation'].mean()
        }
        
        # Generalization statistics
        stats['generalization'] = {
            'num_test_samples': len(df_gen),
            'avg_reward': df_gen['average_reward'].mean(),
            'std_reward': df_gen['average_reward'].std(),
            'avg_cooperation': df_gen['cooperation_rate'].mean(),
            'std_cooperation': df_gen['cooperation_rate'].std()
        }
        
        # Same vs different condition performance
        df_gen['same_condition'] = df_gen['train_condition'] == df_gen['test_condition']
        same_cond = df_gen[df_gen['same_condition']]
        diff_cond = df_gen[~df_gen['same_condition']]
        
        stats['generalization']['same_condition_reward'] = same_cond['average_reward'].mean()
        stats['generalization']['diff_condition_reward'] = diff_cond['average_reward'].mean()
        stats['generalization']['generalization_gap'] = (
            same_cond['average_reward'].mean() - diff_cond['average_reward'].mean()
        )
        
        # Save stats
        with open(self.data_dir / 'summary_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print("\nTRAINING:")
        for k, v in stats['training'].items():
            print(f"  {k:30s}: {v}")
        
        print("\nGENERALIZATION:")
        for k, v in stats['generalization'].items():
            print(f"  {k:30s}: {v}")
        
        return stats
    
    def generate_report(self, df_training, df_gen, stats):
        """Generate comprehensive markdown report"""
        print("\n" + "="*80)
        print("GENERATING REPORT")
        print("="*80)
        
        report = f"""# Generalization Matrix Analysis Report
## Experiment 918988 (Training) / 918989 (Testing)

### Experiment Overview
- **Training Conditions**: {stats['training']['num_conditions']}
- **Seeds per Condition**: {stats['training']['num_seeds']}
- **Total Models Trained**: {stats['training']['total_models']}
- **Network Architecture**: Hidden size 128, 2 LSTM layers

### Training Results
- **Average Epochs to Convergence**: {stats['training']['avg_epochs']:.1f} ± {stats['training']['std_epochs']:.1f}
- **Average Final Loss**: {stats['training']['avg_final_loss']:.4f}
- **Average Final Cooperation**: {stats['training']['avg_final_cooperation']:.3f}

### Generalization Results
- **Total Test Samples**: {stats['generalization']['num_test_samples']}
- **Average Test Reward**: {stats['generalization']['avg_reward']:.3f} ± {stats['generalization']['std_reward']:.3f}
- **Average Test Cooperation**: {stats['generalization']['avg_cooperation']:.3f} ± {stats['generalization']['std_cooperation']:.3f}

### Generalization Performance
- **Same Condition Reward**: {stats['generalization']['same_condition_reward']:.3f}
- **Different Condition Reward**: {stats['generalization']['diff_condition_reward']:.3f}
- **Generalization Gap**: {stats['generalization']['generalization_gap']:.3f}

### Generated Outputs
- Training summary: `data/training_summary.csv`
- Generalization matrix: `data/generalization_matrix.csv`
- Statistics: `data/summary_statistics.json`
- Training plots: `figures/training_convergence.png`
- Generalization heatmap: `figures/generalization_heatmap.png`

### Notes
- This experiment used 5 seeds (not 20 as initially planned)
- Network size reduced to 128 hidden units (from 256)
- Input size is 9 elements (updated architecture)

---
*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        report_path = self.output_dir / 'ANALYSIS_REPORT.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved: {report_path}")
        return report
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*80)
        print("STARTING FULL ANALYSIS PIPELINE")
        print("="*80)
        
        # Load data
        self.load_training_data()
        self.load_test_data()
        
        # Extract summaries
        df_training = self.extract_training_summary()
        df_gen = self.extract_generalization_matrix()
        
        # Generate plots
        self.plot_training_convergence(df_training)
        self.plot_generalization_heatmap(df_gen)
        
        # Compute statistics
        stats = self.compute_statistics(df_training, df_gen)
        
        # Generate report
        report = self.generate_report(df_training, df_gen, stats)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {self.output_dir}")
        
        return df_training, df_gen, stats


if __name__ == '__main__':
    # Paths
    TRAIN_DIR = r"c:\Users\User\OneDrive - huji.ac.il\מסמכים\Max_Planck\Cognitive-Therapy-fo-AI-\experiments\generalization_matrix_train_918988"
    TEST_DIR = r"c:\Users\User\OneDrive - huji.ac.il\מסמכים\Max_Planck\Cognitive-Therapy-fo-AI-\experiments\generalization_matrix_test_918989"
    OUTPUT_DIR = r"c:\Users\User\OneDrive - huji.ac.il\מסמכים\Max_Planck\Cognitive-Therapy-fo-AI-\results\gen_matrix_918988_analysis"
    
    # Run analysis
    analyzer = GeneralizationMatrixAnalyzer918988(TRAIN_DIR, TEST_DIR, OUTPUT_DIR)
    df_training, df_gen, stats = analyzer.run_full_analysis()
    
    print("\n✓ Analysis pipeline completed successfully")
