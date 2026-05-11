"""
Vanilla RL Generalization Analysis
Analyzes test results from generalization_matrix_test_893509 and 893510.

Test data structure:
- 75 trained models (15 training conditions × 5 seeds)
- 14 test conditions per model
- Total: 1050 test evaluations

Each result contains:
- Test condition (game, opponent range)
- Performance metrics (reward, cooperation rate)
- Per-opponent breakdowns
"""

import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

class VanillaGeneralizationAnalyzer:
    def __init__(self, test_dir_1, test_dir_2, training_dir, output_dir):
        """
        Args:
            test_dir_1: Path to generalization_matrix_test_893509
            test_dir_2: Path to generalization_matrix_test_893510
            training_dir: Path to generalization_matrix_train_888509
            output_dir: Where to save analysis outputs
        """
        self.test_dir_1 = Path(test_dir_1)
        self.test_dir_2 = Path(test_dir_2)
        self.training_dir = Path(training_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.fig_dir = self.output_dir / 'figures'
        self.data_dir = self.output_dir / 'data'
        for dir_path in [self.fig_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Experiment parameters
        # Note: Training order is PD(0-4), SH(5-9), HD(10-14)
        #       Test order is PD(0-4), HD(5-9), SH(10-14)
        # Using test order for the mini-heatmap rows since that's what gets tested
        self.games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
        self.opp_ranges = ['very_low', 'low', 'mid', 'high', 'very_high']
        
        # Training game display order (for overall grid rows)
        self.training_games = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
        self.num_conditions = 15  # 3 games × 5 opponent ranges
        self.num_seeds = 5
        self.num_test_conditions = 14
        
        # Data storage
        self.test_results = {}  # {model_id: {test_cond_id: results}}
        self.training_conditions = {}  # {model_id: training_condition}
        self.test_conditions = {}  # {test_cond_id: test_condition}
        
    def load_training_conditions(self):
        """Load training condition info from training directory"""
        print("Loading training conditions...")
        
        training_path = self.training_dir / 'training'
        condition_dirs = sorted(training_path.glob('condition_*_seed_*'))
        
        for cond_dir in condition_dirs:
            parts = cond_dir.name.split('_')
            condition_id = int(parts[1])
            seed = int(parts[3])
            
            # Model ID is condition_id * num_seeds + seed
            model_id = condition_id * self.num_seeds + seed
            
            # Determine game and opponent range from condition_id
            # condition_id: 0-4 = PD (very_low to very_high)
            #               5-9 = SH (very_low to very_high)
            #              10-14 = HD (very_low to very_high)
            game_idx = condition_id // 5
            opp_idx = condition_id % 5
            
            game = self.games[game_idx]
            opp_range = self.opp_ranges[opp_idx]
            
            self.training_conditions[model_id] = {
                'model_id': model_id,
                'condition_id': condition_id,
                'seed': seed,
                'game': game,
                'opponent_range': opp_range
            }
        
        print(f"  Loaded {len(self.training_conditions)} training conditions\n")
        return self.training_conditions
    
    def load_test_results(self):
        """Load all test results from both test directories"""
        print("Loading test results...")
        
        # Combine both test directories
        test_dirs = [self.test_dir_1, self.test_dir_2]
        total_loaded = 0
        total_attempted = 0
        
        for test_base_dir in test_dirs:
            testing_path = test_base_dir / 'testing'
            if not testing_path.exists():
                print(f"  Warning: {testing_path} not found")
                continue
            
            model_dirs = sorted(testing_path.glob('model_*_test_cond_*'))
            print(f"  Found {len(model_dirs)} directories in {test_base_dir.name}")
            
            for model_dir in model_dirs:
                total_attempted += 1
                # Parse directory name: model_X_test_cond_Y
                parts = model_dir.name.split('_')
                model_id = int(parts[1])
                test_cond_id = int(parts[4])
                
                # Find the experiment subdirectory (has timestamp)
                exp_dirs = list(model_dir.glob('generalization_matrix_task_*'))
                if not exp_dirs:
                    print(f"  Warning: No experiment dir in {model_dir.name}")
                    continue
                
                exp_dir = exp_dirs[0]  # Should only be one
                
                # Find results pickle file (task ID may not match model ID)
                results_dir = exp_dir / 'results'
                if not results_dir.exists():
                    print(f"  Warning: No results dir in {model_dir.name}")
                    continue
                
                pkl_files = list(results_dir.glob('eval_model_None_task_*_results.pkl'))
                if not pkl_files:
                    print(f"  Warning: No pickle file found for model {model_id}, test_cond {test_cond_id}")
                    continue
                
                results_file = pkl_files[0]  # Should only be one
                
                # Load the pickle file
                try:
                    with open(results_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Extract evaluation results for this test condition
                    eval_results = data.get('evaluation_results', {})
                    cond_key = f'condition_{test_cond_id}'
                    
                    if cond_key in eval_results:
                        cond_data = eval_results[cond_key]
                        
                        # Store test condition info
                        if test_cond_id not in self.test_conditions:
                            test_cond_info = cond_data.get('test_condition', {})
                            self.test_conditions[test_cond_id] = test_cond_info
                        
                        # Store results
                        if model_id not in self.test_results:
                            self.test_results[model_id] = {}
                        
                        self.test_results[model_id][test_cond_id] = cond_data
                        total_loaded += 1
                
                except Exception as e:
                    print(f"  Error loading {results_file}: {e}")
                
                # Progress update every 100 files
                if total_attempted % 100 == 0:
                    print(f"    Progress: {total_attempted} directories checked, {total_loaded} results loaded")
        
        print(f"  Loaded {total_loaded} test results from {total_attempted} directories")
        print(f"  {len(self.test_results)} unique models × {len(self.test_conditions)} test conditions\n")
        return self.test_results
    
    def create_generalization_dataframe(self):
        """Create comprehensive dataframe of all generalization results"""
        print("Creating generalization dataframe...")
        
        rows = []
        
        for model_id, test_conds in self.test_results.items():
            train_info = self.training_conditions.get(model_id, {})
            
            for test_cond_id, test_data in test_conds.items():
                test_info = self.test_conditions.get(test_cond_id, {})
                
                # Aggregate metrics across opponents in this test condition
                results = test_data.get('results', {})
                rewards = []
                coop_rates = []
                
                for opp_key, opp_data in results.items():
                    if isinstance(opp_data, dict):
                        rewards.append(opp_data.get('average_reward', np.nan))
                        coop_rates.append(opp_data.get('cooperation_rate', np.nan))
                
                # Check if same game/opponent range
                train_game = train_info.get('game')
                train_opp_range = train_info.get('opponent_range')
                test_game = test_info.get('game')
                test_opp_range = test_info.get('opponent_range')
                
                row = {
                    'model_id': model_id,
                    'train_condition_id': train_info.get('condition_id'),
                    'seed': train_info.get('seed'),
                    'train_game': train_game,
                    'train_opponent_range': train_opp_range,
                    'test_condition_id': test_cond_id,
                    'test_game': test_game,
                    'test_opponent_range': test_opp_range,
                    'mean_reward': np.mean(rewards) if rewards else np.nan,
                    'std_reward': np.std(rewards) if rewards else np.nan,
                    'mean_cooperation_rate': np.mean(coop_rates) if coop_rates else np.nan,
                    'std_cooperation_rate': np.std(coop_rates) if coop_rates else np.nan,
                    'same_game': train_game == test_game,
                    'same_opponent_range': train_opp_range == test_opp_range,
                    'same_condition': (train_game == test_game) and (train_opp_range == test_opp_range)
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save to CSV
        csv_path = self.data_dir / 'generalization_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"  Saved {len(df)} rows to generalization_results.csv\n")
        
        return df
    
    def plot_generalization_matrix(self, df):
        """Plot full generalization matrix: 15 train conditions × 14 test conditions"""
        print("Generating generalization matrix heatmap...")
        
        # Aggregate across seeds (5 seeds per training condition)
        agg_data = df.groupby(['train_condition_id', 'test_condition_id']).agg({
            'mean_reward': ['mean', 'std'],
            'mean_cooperation_rate': ['mean', 'std']
        }).reset_index()
        
        # Create pivot table for reward
        reward_matrix = agg_data.pivot(
            index='train_condition_id',
            columns='test_condition_id',
            values=('mean_reward', 'mean')
        )
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        sns.heatmap(
            reward_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=reward_matrix.median().median(),
            ax=ax,
            cbar_kws={'label': 'Mean Reward'}
        )
        
        # Add labels
        ax.set_xlabel('Test Condition ID', fontsize=12)
        ax.set_ylabel('Training Condition ID', fontsize=12)
        ax.set_title('Generalization Matrix: Mean Reward\n15 Training Conditions × 14 Test Conditions', 
                     fontsize=14, fontweight='bold')
        
        # Add game boundaries
        for game_idx in range(1, 3):
            ax.axhline(y=game_idx * 5, color='blue', linewidth=2, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'generalization_matrix_reward.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved generalization_matrix_reward.png\n")
    
    def plot_same_vs_cross_game(self, df):
        """Compare performance on same-game vs cross-game generalization"""
        print("Generating same-game vs cross-game comparison...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Panel 1: Reward comparison
        ax = axes[0]
        data_to_plot = [
            df[df['same_game']]['mean_reward'].dropna(),
            df[~df['same_game']]['mean_reward'].dropna()
        ]
        bp = ax.boxplot(data_to_plot, labels=['Same Game', 'Cross Game'], 
                        patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Mean Reward', fontsize=11)
        ax.set_title('Reward: Same vs Cross Game', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Cooperation rate comparison
        ax = axes[1]
        data_to_plot = [
            df[df['same_game']]['mean_cooperation_rate'].dropna(),
            df[~df['same_game']]['mean_cooperation_rate'].dropna()
        ]
        bp = ax.boxplot(data_to_plot, labels=['Same Game', 'Cross Game'],
                        patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Cooperation Rate', fontsize=11)
        ax.set_title('Cooperation: Same vs Cross Game', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Distribution by training game
        ax = axes[2]
        for game in self.games:
            game_df = df[df['train_game'] == game]
            same_game = game_df[game_df['same_game']]['mean_reward'].dropna()
            cross_game = game_df[~game_df['same_game']]['mean_reward'].dropna()
            
            if len(same_game) > 0 and len(cross_game) > 0:
                ax.scatter([game], [same_game.mean()], s=100, 
                          label=f'{game} (same)', alpha=0.7)
                ax.scatter([game], [cross_game.mean()], s=100, marker='x',
                          label=f'{game} (cross)', alpha=0.7)
        
        ax.set_ylabel('Mean Reward', fontsize=11)
        ax.set_xlabel('Training Game', fontsize=11)
        ax.set_title('Generalization by Training Game', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'same_vs_cross_game.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved same_vs_cross_game.png\n")
    
    def plot_by_training_game(self, df):
        """Show generalization performance grouped by training game"""
        print("Generating performance by training game...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, train_game in enumerate(self.games):
            ax = axes[idx]
            game_df = df[df['train_game'] == train_game]
            
            # Group by test game
            test_game_rewards = []
            test_game_labels = []
            
            for test_game in self.games:
                test_df = game_df[game_df['test_game'] == test_game]
                if len(test_df) > 0:
                    test_game_rewards.append(test_df['mean_reward'].dropna())
                    test_game_labels.append(test_game.replace('-', '\n'))
            
            if test_game_rewards:
                bp = ax.boxplot(test_game_rewards, labels=test_game_labels,
                               patch_artist=True, showmeans=True)
                
                # Color same-game differently
                for i, label in enumerate(test_game_labels):
                    if label.replace('\n', '-') == train_game:
                        bp['boxes'][i].set_facecolor('lightgreen')
                    else:
                        bp['boxes'][i].set_facecolor('lightcoral')
            
            ax.set_ylabel('Mean Reward', fontsize=11)
            ax.set_xlabel('Test Game', fontsize=11)
            ax.set_title(f'Trained on: {train_game}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'generalization_by_training_game.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved generalization_by_training_game.png\n")
    
    def plot_opponent_generalization(self, df):
        """Analyze generalization across opponent types"""
        print("Generating opponent generalization analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # For each training game
        for game_idx, train_game in enumerate(self.games):
            # Panel 1: Same game, different opponents
            ax = axes[game_idx * 2]
            game_df = df[(df['train_game'] == train_game) & (df['same_game'] == True)]
            
            # Group by training opponent range
            for opp_range in self.opp_ranges:
                opp_df = game_df[game_df['train_opponent_range'] == opp_range]
                if len(opp_df) > 0:
                    test_opp_ranges = []
                    rewards = []
                    
                    for test_opp_range in self.opp_ranges:
                        test_df = opp_df[opp_df['test_opponent_range'] == test_opp_range]
                        if len(test_df) > 0:
                            test_opp_ranges.append(test_opp_range)
                            rewards.append(test_df['mean_reward'].mean())
                    
                    if rewards:
                        ax.plot(test_opp_ranges, rewards, marker='o', label=opp_range, alpha=0.7)
            
            ax.set_xlabel('Test Opponent Range', fontsize=10)
            ax.set_ylabel('Mean Reward', fontsize=10)
            ax.set_title(f'{train_game}: Same Game\nOpponent Generalization', fontsize=11, fontweight='bold')
            ax.legend(title='Train Opp.', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=30)
            
            # Panel 2: Cross game
            ax = axes[game_idx * 2 + 1]
            game_df = df[(df['train_game'] == train_game) & (df['same_game'] == False)]
            
            # Group by test game
            for test_game in self.games:
                if test_game == train_game:
                    continue
                test_game_df = game_df[game_df['test_game'] == test_game]
                
                if len(test_game_df) > 0:
                    test_opp_ranges = []
                    rewards = []
                    
                    for test_opp_range in self.opp_ranges:
                        test_df = test_game_df[test_game_df['test_opponent_range'] == test_opp_range]
                        if len(test_df) > 0:
                            test_opp_ranges.append(test_opp_range)
                            rewards.append(test_df['mean_reward'].mean())
                    
                    if rewards:
                        ax.plot(test_opp_ranges, rewards, marker='s', label=test_game, alpha=0.7)
            
            ax.set_xlabel('Test Opponent Range', fontsize=10)
            ax.set_ylabel('Mean Reward', fontsize=10)
            ax.set_title(f'{train_game}: Cross Game\nTo Different Games', fontsize=11, fontweight='bold')
            ax.legend(title='Test Game', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=30)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'opponent_generalization.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved opponent_generalization.png\n")
    
    def binary_kl_divergence(self, p, q, epsilon=0.001):
        """Binary KL divergence: KL(P||Q) for cooperation rates."""
        p = np.clip(p, epsilon, 1 - epsilon)
        q = np.clip(q, epsilon, 1 - epsilon)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    
    def get_optimal_policy(self, game, opponent_range):
        """Get optimal cooperation rate for a game/opponent combination."""
        # Define opponent cooperation probabilities
        opp_coop_map = {
            'very_low': 0.85,  # opponents defect at 0.15
            'low': 0.7,        # opponents defect at 0.3
            'mid': 0.5,        # opponents defect at 0.5
            'high': 0.3,       # opponents defect at 0.7
            'very_high': 0.1   # opponents defect at 0.9
        }
        opp_coop = opp_coop_map.get(opponent_range, 0.5)
        
        if game == 'prisoners-dilemma':
            # Dominant strategy: always defect (cooperation rate = 0)
            return 0.0
        
        elif game == 'stag-hunt':
            # Coordination game: match opponent's strategy
            # If opponent cooperates, cooperate; if defects, defect
            return opp_coop
        
        elif game == 'hawk-dove':
            # Anti-coordination game: play opposite of opponent
            # If opponent cooperates (dove), defect (hawk)
            # If opponent defects (hawk), cooperate (dove)
            return 1.0 - opp_coop
        
        return 0.5  # fallback
    
    def load_training_cooperation_rates(self):
        """Load cooperation rates from training for all models."""
        print("Loading training cooperation rates...")
        
        training_coop_rates = {}
        training_path = self.training_dir / 'training'
        
        for model_id, train_info in self.training_conditions.items():
            condition_id = train_info['condition_id']
            seed = train_info['seed']
            
            # Find training directory
            cond_dir = training_path / f"condition_{condition_id}_seed_{seed}"
            if not cond_dir.exists():
                continue
            
            # Find experiment subdirectory
            exp_dirs = list(cond_dir.glob('generalization_matrix_task_*'))
            if not exp_dirs:
                continue
            
            exp_dir = exp_dirs[0]
            # NOTE: results file is named by condition_id, not model_id!
            results_file = exp_dir / 'results' / f'training_task_{condition_id}_results.pkl'
            
            if not results_file.exists():
                continue
            
            try:
                with open(results_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Get final epoch cooperation rate
                epoch_results = data.get('training_results', {}).get('epoch_results', [])
                if epoch_results:
                    final_epoch = epoch_results[-1]
                    train_coop = final_epoch.get('epoch_average_cooperation_rate', np.nan)
                    training_coop_rates[model_id] = train_coop
                    
            except Exception as e:
                print(f"  Error loading training data for model {model_id}: {e}")
        
        print(f"  Loaded training cooperation rates for {len(training_coop_rates)} models\n")
        return training_coop_rates
    
    def plot_training_policy_kld_grid(self, training_coop_rates):
        """Plot 3×5 grid showing KLD between agent's training policy and optimal policy"""
        print("Generating training policy vs optimal KLD grid...")
        
        # Compute KLD for each training condition
        kld_data = []
        
        for model_id, train_coop in training_coop_rates.items():
            train_info = self.training_conditions.get(model_id, {})
            if not train_info:
                continue
            
            game = train_info['game']
            opp_range = train_info['opponent_range']
            
            # Get optimal policy
            optimal_coop = self.get_optimal_policy(game, opp_range)
            
            # Compute KLD
            kld = self.binary_kl_divergence(train_coop, optimal_coop)
            
            kld_data.append({
                'model_id': model_id,
                'condition_id': train_info['condition_id'],
                'seed': train_info['seed'],
                'game': game,
                'opponent_range': opp_range,
                'agent_coop_rate': train_coop,
                'optimal_coop_rate': optimal_coop,
                'kld': kld
            })
        
        kld_df = pd.DataFrame(kld_data)
        
        # Save data
        kld_df.to_csv(self.data_dir / 'training_policy_kld.csv', index=False)
        
        # Create 3×5 grid
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        
        # For each training condition (using training game order)
        for game_idx, train_game in enumerate(self.training_games):
            for opp_idx, train_opp_range in enumerate(self.opp_ranges):
                ax = axes[game_idx, opp_idx]
                
                # Filter data for this training condition
                subset = kld_df[
                    (kld_df['game'] == train_game) &
                    (kld_df['opponent_range'] == train_opp_range)
                ]
                
                if len(subset) == 0:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                           transform=ax.transAxes, fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                # Compute mean KLD across seeds
                mean_kld = subset['kld'].mean()
                std_kld = subset['kld'].std()
                
                # Get agent and optimal cooperation rates
                mean_agent_coop = subset['agent_coop_rate'].mean()
                optimal_coop = subset['optimal_coop_rate'].iloc[0]  # Same for all seeds
                
                # Color based on KLD value
                kld_normalized = np.clip(mean_kld / 3.0, 0, 1)  # Normalize to [0,1] for coloring
                color = plt.cm.YlOrRd(kld_normalized)
                
                # Display KLD value
                ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', lw=1))
                ax.text(0.5, 0.6, f'{mean_kld:.3f}', ha='center', va='center',
                       fontsize=14, fontweight='bold', transform=ax.transAxes)
                
                # Display agent vs optimal
                ax.text(0.5, 0.35, f'Agent: {mean_agent_coop:.2f}', ha='center', va='center',
                       fontsize=8, transform=ax.transAxes)
                ax.text(0.5, 0.2, f'Optimal: {optimal_coop:.2f}', ha='center', va='center',
                       fontsize=8, transform=ax.transAxes, style='italic')
                
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                
                #Title for each subplot
                title = f'{train_game.replace("-", " ").title()}\n{train_opp_range.replace("_", " ").title()}'
                ax.set_title(title, fontsize=9, fontweight='bold')
        
        # Add column labels (opponent ranges) at top
        for opp_idx, opp_range in enumerate(self.opp_ranges):
            axes[0, opp_idx].set_xlabel(opp_range.replace('_', ' ').title(), 
                                       fontsize=10, fontweight='bold')
            axes[0, opp_idx].xaxis.set_label_position('top')
        
        # Add row labels (games) on left - TRAINING game order
        for game_idx, game in enumerate(self.training_games):
            axes[game_idx, 0].set_ylabel(game.replace('-', ' ').title(), 
                                        fontsize=10, fontweight='bold', rotation=0,
                                        ha='right', va='center')
        
        # Add color bar
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.colors import Normalize
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = Normalize(vmin=0, vmax=3)
        cb = ColorbarBase(cbar_ax, cmap=plt.cm.YlOrRd, norm=norm, orientation='vertical')
        cb.set_label('KLD (Agent || Optimal)', fontsize=12, fontweight='bold')
        
        # Add overall title
        fig.suptitle('Training Policy vs Optimal Policy KLD Grid\n' +
                    '(KLD between agent\'s training cooperation rate and optimal policy for that setup)',
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.99])
        plt.savefig(self.fig_dir / 'training_vs_optimal_kld_grid_3x5.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved training_vs_optimal_kld_grid_3x5.png\n")
        
        return kld_df
    
    def plot_test_policy_optimality(self, df):
        """Plot KLD between test policies and optimal policies for all training setups"""
        print("Generating test policy optimality analysis...")
        
        # Compute KLD for each test result
        test_kld_data = []
        
        for _, row in df.iterrows():
            # Get test condition info
            test_game = row['test_game']
            test_opp_range = row['test_opponent_range']
            test_coop_rate = row['mean_cooperation_rate']
            
            # Get training condition info
            train_game = row['train_game']
            train_opp_range = row['train_opponent_range']
            
            # Compute optimal policy for the TEST condition
            optimal_coop = self.get_optimal_policy(test_game, test_opp_range)
            
            # Compute KLD between agent's test policy and optimal policy for test condition
            kld = self.binary_kl_divergence(test_coop_rate, optimal_coop)
            
            test_kld_data.append({
                'train_game': train_game,
                'train_opponent_range': train_opp_range,
                'test_game': test_game,
                'test_opponent_range': test_opp_range,
                'test_coop_rate': test_coop_rate,
                'optimal_coop': optimal_coop,
                'kld': kld
            })
        
        kld_df = pd.DataFrame(test_kld_data)
        
        # Save data
        kld_df.to_csv(self.data_dir / 'test_policy_optimality.csv', index=False)
        
        # Create test condition labels in specified order: PD, SH, HD
        test_condition_order = []
        test_labels = []
        for game in ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']:
            for opp_range in self.opp_ranges:
                test_condition_order.append((game, opp_range))
                game_abbrev = {'prisoners-dilemma': 'PD', 'stag-hunt': 'SH', 'hawk-dove': 'HD'}[game]
                opp_abbrev = {'very_low': 'VL', 'low': 'Lo', 'mid': 'Mi', 'high': 'Hi', 'very_high': 'VH'}[opp_range]
                test_labels.append(f'{game_abbrev}-{opp_abbrev}')
        
        # Colors for training opponent ranges
        colors = {
            'very_low': '#1f77b4',   # blue
            'low': '#ff7f0e',        # orange
            'mid': '#2ca02c',        # green
            'high': '#d62728',       # red
            'very_high': '#9467bd'  # purple
        }
        
        # Create 3 subplots (one per training game)
        fig, axes = plt.subplots(3, 1, figsize=(20, 15))
        
        for ax_idx, train_game in enumerate(self.training_games):
            ax = axes[ax_idx]
            
            # Plot each training opponent range as a separate line
            for train_opp_range in self.opp_ranges:
                # Filter data for this training condition
                train_data = kld_df[
                    (kld_df['train_game'] == train_game) &
                    (kld_df['train_opponent_range'] == train_opp_range)
                ]
                
                if len(train_data) == 0:
                    continue
                
                # Order by test conditions
                kld_values = []
                for test_game, test_opp_range in test_condition_order:
                    test_subset = train_data[
                        (train_data['test_game'] == test_game) &
                        (train_data['test_opponent_range'] == test_opp_range)
                    ]
                    if len(test_subset) > 0:
                        kld_values.append(test_subset['kld'].mean())
                    else:
                        # Fill missing values with 0 for continuity (training condition not tested)
                        kld_values.append(0.0)
                
                # Plot line
                label = train_opp_range.replace('_', ' ').title()
                ax.plot(range(len(test_condition_order)), kld_values,
                       color=colors[train_opp_range],
                       linewidth=2.5,
                       marker='o',
                       markersize=5,
                       label=label,
                       alpha=0.8)
            
            # Formatting
            ax.set_ylabel('KLD (Agent || Optimal)', fontsize=12, fontweight='bold')
            ax.set_title(f'Training Game: {train_game.replace("-", " ").title()}',
                        fontsize=14, fontweight='bold')
            
            # Set x-axis labels
            ax.set_xticks(range(len(test_labels)))
            if ax_idx == 2:  # Only bottom subplot
                ax.set_xticklabels(test_labels, rotation=45, ha='right', fontsize=10)
                ax.set_xlabel('Test Condition', fontsize=12, fontweight='bold')
            else:
                ax.set_xticklabels([])
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add legend
            ax.legend(loc='upper right', fontsize=10, title='Training Opponent', title_fontsize=11)
            
            # Add vertical lines to separate games
            ax.axvline(x=4.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.axvline(x=9.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        
        # Overall title
        fig.suptitle('Test Policy Optimality: KLD Between Agent Test Policy and Optimal Policy\n' +
                    'Each subplot shows one training game with different training opponent ranges (colors)',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(self.fig_dir / 'test_policy_optimality.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved test_policy_optimality.png\n")
        
        return kld_df
    
    def plot_cross_model_kld(self, df):
        """Analyze KLD between models trained on different games tested on same condition"""
        print("Generating cross-model KLD analysis (3×5 grid)...")
        
        cross_kld_data = []
        
        # For each unique test condition
        test_conditions = df[['test_game', 'test_opponent_range']].drop_duplicates()
        
        for _, test_cond in test_conditions.iterrows():
            test_game = test_cond['test_game']
            test_opp = test_cond['test_opponent_range']
            
            # Get all models tested on this condition, grouped by training game
            test_data = df[
                (df['test_game'] == test_game) & 
                (df['test_opponent_range'] == test_opp)
            ]
            
            # Compute mean cooperation rate for each training game on this test condition
            train_game_coops = {}
            for train_game in self.games:
                game_data = test_data[test_data['train_game'] == train_game]
                if len(game_data) > 0:
                    train_game_coops[train_game] = game_data['mean_cooperation_rate'].mean()
            
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
                            'train_game_1': train_game_1,
                            'train_game_2': train_game_2,
                            'coop_rate_1': p,
                            'coop_rate_2': q,
                            'coop_diff': p - q,
                            'kld': kld
                        })
        
        kld_df = pd.DataFrame(cross_kld_data)
        
        # Save data
        kld_df.to_csv(self.data_dir / 'cross_model_kld.csv', index=False)
        
        # Create 3×5 grid of KLD matrices (matching heatmap structure)
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        
        # For each test condition
        for game_idx, test_game in enumerate(self.games):
            for opp_idx, test_opp_range in enumerate(self.opp_ranges):
                ax = axes[game_idx, opp_idx]
                
                # Filter for this specific test condition
                cond_kld = kld_df[
                    (kld_df['test_game'] == test_game) &
                    (kld_df['test_opponent_range'] == test_opp_range)
                ]
                
                if len(cond_kld) == 0:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                           transform=ax.transAxes, fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                # Create KLD matrix (3×3 for training games)
                kld_matrix = np.zeros((len(self.games), len(self.games)))
                
                for i, train_game_1 in enumerate(self.games):
                    for j, train_game_2 in enumerate(self.games):
                        if train_game_1 == train_game_2:
                            kld_matrix[i, j] = 0
                        else:
                            subset = cond_kld[
                                (cond_kld['train_game_1'] == train_game_1) &
                                (cond_kld['train_game_2'] == train_game_2)
                            ]
                            if len(subset) > 0:
                                kld_matrix[i, j] = subset['kld'].mean()
                
                # Create labels for KLD matrix axes (training games)
                game_labels = [g.replace('-', '\n').upper()[:2] for g in self.games]  # PD, SH, HD
                
                # Plot heatmap
                sns.heatmap(kld_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                           ax=ax, cbar=False, vmin=0, vmax=3,
                           xticklabels=game_labels if game_idx == 2 else [],  # Only bottom row
                           yticklabels=game_labels if opp_idx == 0 else [])  # Only left column
                
                # Adjust tick label sizes
                if game_idx == 2:  # Bottom row
                    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=0)
                if opp_idx == 0:  # Left column
                    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6, rotation=0)
                
                # Title for each subplot
                title = f'{test_game.replace("-", " ").title()}\n{test_opp_range.replace("_", " ").title()}'
                ax.set_title(title, fontsize=9, fontweight='bold')
        
        # Add column labels (opponent ranges) at top
        for opp_idx, opp_range in enumerate(self.opp_ranges):
            axes[0, opp_idx].set_xlabel(opp_range.replace('_', ' ').title(), 
                                       fontsize=10, fontweight='bold')
            axes[0, opp_idx].xaxis.set_label_position('top')
        
        # Add row labels (games) on left
        for game_idx, game in enumerate(self.games):
            axes[game_idx, 0].set_ylabel(game.replace('-', ' ').title(), 
                                        fontsize=10, fontweight='bold', rotation=0,
                                        ha='right', va='center')
        
        # Add overall title
        fig.suptitle('Cross-Model KLD Grid: Policy Divergence Between Training Games\n' +
                    '(Each cell: 3×3 matrix showing KLD between PD/SH/HD trained models on that test condition)',
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(self.fig_dir / 'cross_model_kld_grid_3x5.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved cross_model_kld_grid_3x5.png\n")
    
    def plot_generalization_heatmap_grid(self, df):
        """Plot 3×5 grid of heatmaps showing each training condition's generalization (NORMALIZED)"""
        print("Generating 3×5 generalization heatmap grid (normalized rewards)...")
        
        # First pass: compute min/max reward per test game for normalization
        game_reward_ranges = {}
        for test_game in self.games:
            game_rewards = df[df['test_game'] == test_game]['mean_reward'].dropna()
            if len(game_rewards) > 0:
                game_reward_ranges[test_game] = {
                    'min': game_rewards.min(),
                    'max': game_rewards.max()
                }
        
        # Create 3×5 grid (3 games × 5 opponent ranges)
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        
        # For each training condition (using training game order for grid rows)
        for game_idx, train_game in enumerate(self.training_games):
            for opp_idx, train_opp_range in enumerate(self.opp_ranges):
                ax = axes[game_idx, opp_idx]
                
                # Filter data for this training condition
                train_data = df[
                    (df['train_game'] == train_game) &
                    (df['train_opponent_range'] == train_opp_range)
                ]
                
                if len(train_data) == 0:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                           transform=ax.transAxes, fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                # Create matrix: rows = test games, cols = test opponent ranges
                matrix = np.zeros((len(self.games), len(self.opp_ranges)))
                
                for test_game_idx, test_game in enumerate(self.games):
                    for test_opp_idx, test_opp_range in enumerate(self.opp_ranges):
                        subset = train_data[
                            (train_data['test_game'] == test_game) &
                            (train_data['test_opponent_range'] == test_opp_range)
                        ]
                        if len(subset) > 0:
                            raw_reward = subset['mean_reward'].mean()
                            # Normalize per test game
                            if test_game in game_reward_ranges:
                                min_r = game_reward_ranges[test_game]['min']
                                max_r = game_reward_ranges[test_game]['max']
                                if max_r > min_r:
                                    normalized_reward = (raw_reward - min_r) / (max_r - min_r)
                                else:
                                    normalized_reward = 0.5
                                matrix[test_game_idx, test_opp_idx] = normalized_reward
                            else:
                                matrix[test_game_idx, test_opp_idx] = np.nan
                        else:
                            matrix[test_game_idx, test_opp_idx] = np.nan
                
                # Create labels for mini-heatmap axes
                game_labels = [g.replace('-', '\n').upper()[:2] for g in self.games]  # PD, SH, HD
                opp_labels = [o.replace('_', '\n').title()[:4] for o in self.opp_ranges]  # VL, L, M, H, VH
                
                # Plot heatmap (normalized to [0, 1])
                sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                           ax=ax, cbar=False, vmin=0, vmax=1,
                           xticklabels=opp_labels if game_idx == 2 else [],  # Only bottom row
                           yticklabels=game_labels if opp_idx == 0 else [])  # Only left column
                
                # Adjust tick label sizes
                if game_idx == 2:  # Bottom row
                    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=0)
                if opp_idx == 0:  # Left column
                    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6, rotation=0)
                
                # Title for each subplot
                title = f'{train_game.replace("-", " ").title()}\n{train_opp_range.replace("_", " ").title()}'
                ax.set_title(title, fontsize=9, fontweight='bold')
                
                # Highlight same-condition cell
                same_test_game_idx = self.games.index(train_game)
                same_test_opp_idx = self.opp_ranges.index(train_opp_range)
                
                # Debug: Print for a few cells
                if game_idx < 3 and opp_idx == 0:
                    print(f"  Subplot [{game_idx}, {opp_idx}]: train={train_game}/{train_opp_range}")
                    print(f"    Highlighting: row={same_test_game_idx} (test_game={train_game}), col={same_test_opp_idx}")
                
                ax.add_patch(plt.Rectangle((same_test_opp_idx, same_test_game_idx), 1, 1,
                                          fill=False, edgecolor='blue', lw=3))
        
        # Add column labels (opponent ranges) at top
        for opp_idx, opp_range in enumerate(self.opp_ranges):
            axes[0, opp_idx].set_xlabel(opp_range.replace('_', ' ').title(), 
                                       fontsize=10, fontweight='bold')
            axes[0, opp_idx].xaxis.set_label_position('top')
        
        # Add row labels (games) on left - TRAINING game order for overall grid rows
        for game_idx, game in enumerate(self.training_games):
            axes[game_idx, 0].set_ylabel(game.replace('-', ' ').title(), 
                                        fontsize=10, fontweight='bold', rotation=0,
                                        ha='right', va='center')
        
        # Add overall title
        fig.suptitle('Generalization Heatmap Grid: Normalized Reward by Training Condition\n' +
                    '(Rows: Test Games, Cols: Test Opponent Ranges, Blue Box: Same Condition, Normalized per Test Game)',
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(self.fig_dir / 'generalization_heatmap_grid_3x5_normalized.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved generalization_heatmap_grid_3x5_normalized.png\n")
    
    def generate_summary_stats(self, df):
        """Generate summary statistics table"""
        print("Generating summary statistics...")
        
        summary = []
        
        # Overall statistics
        summary.append({
            'metric': 'Overall Mean Reward',
            'value': f"{df['mean_reward'].mean():.3f} ± {df['mean_reward'].std():.3f}"
        })
        
        summary.append({
            'metric': 'Same Game Mean Reward',
            'value': f"{df[df['same_game']]['mean_reward'].mean():.3f}"
        })
        
        summary.append({
            'metric': 'Cross Game Mean Reward',
            'value': f"{df[~df['same_game']]['mean_reward'].mean():.3f}"
        })
        
        summary.append({
            'metric': 'Generalization Gap (Same - Cross)',
            'value': f"{df[df['same_game']]['mean_reward'].mean() - df[~df['same_game']]['mean_reward'].mean():.3f}"
        })
        
        # By training game
        for game in self.games:
            game_df = df[df['train_game'] == game]
            summary.append({
                'metric': f'{game} - Same Game',
                'value': f"{game_df[game_df['same_game']]['mean_reward'].mean():.3f}"
            })
            summary.append({
                'metric': f'{game} - Cross Game',
                'value': f"{game_df[~game_df['same_game']]['mean_reward'].mean():.3f}"
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(self.data_dir / 'summary_statistics.csv', index=False)
        print("  Saved summary_statistics.csv\n")
        
        # Print to console
        print("Summary Statistics:")
        print("=" * 60)
        for _, row in summary_df.iterrows():
            print(f"  {row['metric']}: {row['value']}")
        print("=" * 60 + "\n")
        
        return summary_df
    
    def run_complete_analysis(self):
        """Run all analysis steps"""
        print("="*80)
        print("VANILLA RL GENERALIZATION ANALYSIS")
        print("="*80 + "\n")
        
        # Load data
        self.load_training_conditions()
        self.load_test_results()
        
        # Create dataframe
        df = self.create_generalization_dataframe()
        
        # Generate plots
        self.plot_generalization_matrix(df)
        self.plot_same_vs_cross_game(df)
        self.plot_by_training_game(df)
        self.plot_opponent_generalization(df)
        
        # NEW: Generate KLD and heatmap grid analyses
        self.plot_cross_model_kld(df)
        self.plot_generalization_heatmap_grid(df)
        
        # NEW: Generate training policy optimality analysis
        training_coop_rates = self.load_training_cooperation_rates()
        if training_coop_rates:
            self.plot_training_policy_kld_grid(training_coop_rates)
        
        # NEW: Generate test policy optimality analysis
        self.plot_test_policy_optimality(df)
        
        # Generate summary stats
        self.generate_summary_stats(df)
        
        print("="*80)
        print("ANALYSIS COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("="*80)


if __name__ == '__main__':
    # Set paths
    base_dir = Path(__file__).parent.parent.parent
    test_dir_1 = base_dir / 'experiments' / 'generalization_matrix_test_893509'
    test_dir_2 = base_dir / 'experiments' / 'generalization_matrix_test_893510'
    training_dir = base_dir / 'experiments' / 'generalization_matrix_train_888509'
    output_dir = base_dir / 'experiments' / 'vanilla_generalization_analysis'
    
    # Run analysis
    analyzer = VanillaGeneralizationAnalyzer(test_dir_1, test_dir_2, training_dir, output_dir)
    analyzer.run_complete_analysis()
