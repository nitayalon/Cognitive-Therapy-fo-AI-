"""
Compare performance between baseline (5-element input) and modified (6-element input with opponent action).

This script analyzes and visualizes the impact of adding opponent's previous action to the network input.

Usage:
    python compare_input_modifications.py \
        --baseline-train <old_train_job_id> \
        --baseline-test <old_test_job_id> \
        --modified-train <new_train_job_id> \
        --modified-test <new_test_job_id> \
        --experiment-type {generalization-matrix|whole-population} \
        --output-dir comparison_results
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class InputModificationComparison:
    """Compare baseline vs modified network architectures."""
    
    def __init__(self, baseline_train_id: str, baseline_test_id: str,
                 modified_train_id: str, modified_test_id: str,
                 experiment_type: str, output_dir: str = "comparison_results"):
        """
        Initialize comparison analysis.
        
        Args:
            baseline_train_id: Job ID for baseline training (5-element input)
            baseline_test_id: Job ID for baseline testing
            modified_train_id: Job ID for modified training (6-element input)
            modified_test_id: Job ID for modified testing
            experiment_type: Type of experiment ('generalization-matrix' or 'whole-population')
            output_dir: Directory to save comparison results
        """
        self.baseline_train_id = baseline_train_id
        self.baseline_test_id = baseline_test_id
        self.modified_train_id = modified_train_id
        self.modified_test_id = modified_test_id
        self.experiment_type = experiment_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths to experiment directories
        self.baseline_train_dir = Path(f"experiments/{self._get_prefix()}_train_{baseline_train_id}")
        self.baseline_test_dir = Path(f"experiments/{self._get_prefix()}_test_{baseline_test_id}")
        self.modified_train_dir = Path(f"experiments/{self._get_prefix()}_train_{modified_train_id}")
        self.modified_test_dir = Path(f"experiments/{self._get_prefix()}_test_{modified_test_id}")
        
        # Verify directories exist
        self._verify_directories()
        
        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def _get_prefix(self) -> str:
        """Get experiment directory prefix."""
        if self.experiment_type == 'generalization-matrix':
            return 'generalization_matrix'
        elif self.experiment_type == 'whole-population':
            return 'whole_population'
        else:
            raise ValueError(f"Unknown experiment type: {self.experiment_type}")
    
    def _verify_directories(self):
        """Verify all required directories exist."""
        dirs = [
            self.baseline_train_dir,
            self.baseline_test_dir,
            self.modified_train_dir,
            self.modified_test_dir
        ]
        
        for d in dirs:
            if not d.exists():
                raise ValueError(f"Directory not found: {d}")
        
        print("✓ All experiment directories found")
    
    def load_training_data(self, train_dir: Path) -> pd.DataFrame:
        """Load training data from experiment directory."""
        training_logs = []
        
        # Search for detailed training logs
        log_pattern = "detailed_training_logs/detailed_training_log.csv"
        
        for condition_dir in train_dir.glob("training/*/"):
            log_file = condition_dir / log_pattern
            if log_file.exists():
                df = pd.read_csv(log_file)
                df['condition_dir'] = condition_dir.name
                training_logs.append(df)
        
        if not training_logs:
            raise ValueError(f"No training logs found in {train_dir}")
        
        return pd.concat(training_logs, ignore_index=True)
    
    def load_testing_data(self, test_dir: Path) -> pd.DataFrame:
        """Load testing data from experiment directory."""
        testing_logs = []
        
        # Search for detailed testing logs
        for test_file in test_dir.rglob("detailed_testing_logs/*/detailed_testing_log.csv"):
            df = pd.read_csv(test_file)
            df['test_file'] = test_file.stem
            testing_logs.append(df)
        
        if not testing_logs:
            raise ValueError(f"No testing logs found in {test_dir}")
        
        return pd.concat(testing_logs, ignore_index=True)
    
    def compare_training_convergence(self):
        """Compare training convergence between baseline and modified."""
        print("\n" + "="*60)
        print("TRAINING CONVERGENCE COMPARISON")
        print("="*60)
        
        # Load training data
        baseline_train = self.load_training_data(self.baseline_train_dir)
        modified_train = self.load_training_data(self.modified_train_dir)
        
        # Plot loss curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        ax = axes[0, 0]
        baseline_train.groupby('epoch')['total_loss'].mean().plot(ax=ax, label='Baseline (5-elem)', linewidth=2)
        modified_train.groupby('epoch')['total_loss'].mean().plot(ax=ax, label='Modified (6-elem)', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Training: Total Loss Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # RL loss
        ax = axes[0, 1]
        baseline_train.groupby('epoch')['rl_loss'].mean().plot(ax=ax, label='Baseline (5-elem)', linewidth=2)
        modified_train.groupby('epoch')['rl_loss'].mean().plot(ax=ax, label='Modified (6-elem)', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RL Loss')
        ax.set_title('Training: RL Loss Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Opponent loss (if available)
        ax = axes[1, 0]
        if 'opponent_loss' in baseline_train.columns and 'opponent_loss' in modified_train.columns:
            baseline_train.groupby('epoch')['opponent_loss'].mean().plot(ax=ax, label='Baseline (5-elem)', linewidth=2)
            modified_train.groupby('epoch')['opponent_loss'].mean().plot(ax=ax, label='Modified (6-elem)', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Opponent Loss')
            ax.set_title('Training: Opponent Prediction Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Opponent loss not available', ha='center', va='center')
        
        # Average reward
        ax = axes[1, 1]
        if 'agent_reward' in baseline_train.columns and 'agent_reward' in modified_train.columns:
            baseline_train.groupby('epoch')['agent_reward'].mean().plot(ax=ax, label='Baseline (5-elem)', linewidth=2)
            modified_train.groupby('epoch')['agent_reward'].mean().plot(ax=ax, label='Modified (6-elem)', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Average Reward')
            ax.set_title('Training: Average Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Reward data not available', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_convergence_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training convergence plot saved to {self.output_dir / 'training_convergence_comparison.png'}")
        
        # Compute final convergence statistics
        final_baseline = baseline_train.groupby('epoch').last().iloc[-10:].mean()
        final_modified = modified_train.groupby('epoch').last().iloc[-10:].mean()
        
        print("\nFinal Training Performance (last 10 epochs avg):")
        print(f"  Baseline Total Loss:  {final_baseline['total_loss']:.4f}")
        print(f"  Modified Total Loss:  {final_modified['total_loss']:.4f}")
        print(f"  Improvement:          {((final_baseline['total_loss'] - final_modified['total_loss']) / final_baseline['total_loss'] * 100):.2f}%")
    
    def compare_test_performance(self):
        """Compare test performance between baseline and modified."""
        print("\n" + "="*60)
        print("TEST PERFORMANCE COMPARISON")
        print("="*60)
        
        # Load testing data
        baseline_test = self.load_testing_data(self.baseline_test_dir)
        modified_test = self.load_testing_data(self.modified_test_dir)
        
        # Add version label
        baseline_test['version'] = 'Baseline (5-elem)'
        modified_test['version'] = 'Modified (6-elem)'
        
        # Combine data
        combined_test = pd.concat([baseline_test, modified_test], ignore_index=True)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Average reward by version
        ax = axes[0, 0]
        if 'agent_reward' in combined_test.columns:
            sns.boxplot(data=combined_test, x='version', y='agent_reward', ax=ax)
            ax.set_ylabel('Average Reward')
            ax.set_xlabel('')
            ax.set_title('Test: Average Reward Distribution')
            ax.grid(True, alpha=0.3)
        
        # Cooperation rate
        ax = axes[0, 1]
        if 'agent_cooperation_rate' in combined_test.columns:
            sns.boxplot(data=combined_test, x='version', y='agent_cooperation_rate', ax=ax)
            ax.set_ylabel('Cooperation Rate')
            ax.set_xlabel('')
            ax.set_title('Test: Agent Cooperation Rate')
            ax.grid(True, alpha=0.3)
        
        # Opponent prediction accuracy
        ax = axes[1, 0]
        if 'opponent_prediction_accuracy' in combined_test.columns:
            sns.boxplot(data=combined_test, x='version', y='opponent_prediction_accuracy', ax=ax)
            ax.set_ylabel('Prediction Accuracy')
            ax.set_xlabel('')
            ax.set_title('Test: Opponent Prediction Accuracy')
            ax.grid(True, alpha=0.3)
        
        # Policy KLD (if available)
        ax = axes[1, 1]
        if 'policy_kld' in combined_test.columns:
            sns.boxplot(data=combined_test, x='version', y='policy_kld', ax=ax)
            ax.set_ylabel('Policy KLD')
            ax.set_xlabel('')
            ax.set_title('Test: Policy Divergence')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'test_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Test performance plot saved to {self.output_dir / 'test_performance_comparison.png'}")
        
        # Statistical comparison
        metrics = ['agent_reward', 'agent_cooperation_rate', 'opponent_prediction_accuracy']
        
        print("\nTest Performance Statistics:")
        for metric in metrics:
            if metric in combined_test.columns:
                baseline_vals = baseline_test[metric].dropna()
                modified_vals = modified_test[metric].dropna()
                
                print(f"\n{metric}:")
                print(f"  Baseline: {baseline_vals.mean():.4f} ± {baseline_vals.std():.4f}")
                print(f"  Modified: {modified_vals.mean():.4f} ± {modified_vals.std():.4f}")
                diff = modified_vals.mean() - baseline_vals.mean()
                pct_change = (diff / baseline_vals.mean()) * 100 if baseline_vals.mean() != 0 else 0
                print(f"  Change:   {diff:+.4f} ({pct_change:+.2f}%)")
    
    def compare_generalization(self):
        """Compare generalization patterns between baseline and modified."""
        print("\n" + "="*60)
        print("GENERALIZATION COMPARISON")
        print("="*60)
        
        # Load testing data
        baseline_test = self.load_testing_data(self.baseline_test_dir)
        modified_test = self.load_testing_data(self.modified_test_dir)
        
        # This is experiment-specific
        if self.experiment_type == 'generalization-matrix':
            self._compare_generalization_matrix(baseline_test, modified_test)
        elif self.experiment_type == 'whole-population':
            self._compare_whole_population_generalization(baseline_test, modified_test)
    
    def _compare_generalization_matrix(self, baseline_df: pd.DataFrame, modified_df: pd.DataFrame):
        """Compare generalization patterns for matrix experiments."""
        # Placeholder - implement based on your specific analysis needs
        print("Generalization matrix comparison - implement based on specific metrics")
    
    def _compare_whole_population_generalization(self, baseline_df: pd.DataFrame, modified_df: pd.DataFrame):
        """Compare generalization patterns for whole population experiments."""
        # Group by game and opponent
        if 'test_game' in baseline_df.columns and 'test_opponent' in baseline_df.columns:
            baseline_grouped = baseline_df.groupby(['test_game', 'test_opponent'])['agent_reward'].mean().reset_index()
            modified_grouped = modified_df.groupby(['test_game', 'test_opponent'])['agent_reward'].mean().reset_index()
            
            # Merge for comparison
            comparison = baseline_grouped.merge(
                modified_grouped,
                on=['test_game', 'test_opponent'],
                suffixes=('_baseline', '_modified')
            )
            
            comparison['improvement'] = comparison['agent_reward_modified'] - comparison['agent_reward_baseline']
            
            print("\nCross-game generalization improvement:")
            print(comparison.to_string(index=False))
            
            # Save to CSV
            comparison.to_csv(self.output_dir / 'generalization_comparison.csv', index=False)
            print(f"\n✓ Detailed comparison saved to {self.output_dir / 'generalization_comparison.csv'}")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n" + "="*60)
        print("GENERATING SUMMARY REPORT")
        print("="*60)
        
        report_path = self.output_dir / 'comparison_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("INPUT MODIFICATION COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Experiment Type: {self.experiment_type}\n")
            f.write(f"Baseline (5-element input):\n")
            f.write(f"  Train Job ID: {self.baseline_train_id}\n")
            f.write(f"  Test Job ID:  {self.baseline_test_id}\n")
            f.write(f"Modified (6-element input with opponent action):\n")
            f.write(f"  Train Job ID: {self.modified_train_id}\n")
            f.write(f"  Test Job ID:  {self.modified_test_id}\n")
            f.write("\n" + "="*80 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("="*80 + "\n\n")
            f.write("See generated plots for detailed comparisons:\n")
            f.write("  - training_convergence_comparison.png\n")
            f.write("  - test_performance_comparison.png\n")
            f.write("  - generalization_comparison.csv\n")
        
        print(f"✓ Summary report saved to {report_path}")
    
    def run_full_comparison(self):
        """Run complete comparison analysis."""
        print("\n" + "="*80)
        print("INPUT MODIFICATION COMPARISON ANALYSIS")
        print("="*80)
        print(f"Baseline: Train {self.baseline_train_id}, Test {self.baseline_test_id}")
        print(f"Modified: Train {self.modified_train_id}, Test {self.modified_test_id}")
        print(f"Experiment: {self.experiment_type}")
        print(f"Output: {self.output_dir}")
        print("="*80)
        
        try:
            self.compare_training_convergence()
            self.compare_test_performance()
            self.compare_generalization()
            self.generate_summary_report()
            
            print("\n" + "="*80)
            print("✓ COMPARISON ANALYSIS COMPLETE")
            print("="*80)
            print(f"Results saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs modified network architectures (5-elem vs 6-elem input)"
    )
    parser.add_argument('--baseline-train', required=True, help='Baseline training job ID')
    parser.add_argument('--baseline-test', required=True, help='Baseline testing job ID')
    parser.add_argument('--modified-train', required=True, help='Modified training job ID')
    parser.add_argument('--modified-test', required=True, help='Modified testing job ID')
    parser.add_argument(
        '--experiment-type',
        required=True,
        choices=['generalization-matrix', 'whole-population'],
        help='Type of experiment'
    )
    parser.add_argument(
        '--output-dir',
        default='experiments/analysis_scripts/comparison_results',
        help='Output directory for comparison results'
    )
    
    args = parser.parse_args()
    
    # Run comparison
    comparator = InputModificationComparison(
        baseline_train_id=args.baseline_train,
        baseline_test_id=args.baseline_test,
        modified_train_id=args.modified_train,
        modified_test_id=args.modified_test,
        experiment_type=args.experiment_type,
        output_dir=args.output_dir
    )
    
    comparator.run_full_comparison()


if __name__ == '__main__':
    main()
