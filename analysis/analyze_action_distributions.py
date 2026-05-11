"""
In-Depth Action Distribution Analysis

For each cell in the heatmap (training condition × test condition):
- Show agent cooperation/defection rate
- Show opponent cooperation/defection rate
- Visualize the behavioral patterns underlying the reward differences
"""

import matplotlib
matplotlib.use('Agg')

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
OUTPUT_DIR = Path(__file__).parent / "experiments" / "results" / "vanilla_action_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_from_cached_results():
    """Load data from previously generated all_results.pkl."""
    results_file = OUTPUT_DIR.parent / "vanilla_baseline_analysis" / "all_results.pkl"
    
    print(f"Loading cached results from: {results_file}")
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'rb') as f:
        all_results = pickle.load(f)
    
    print(f"✓ Loaded results for {len(all_results)} tasks")
    print()
    
    # Extract action data
    all_action_data = []
    
    for task_id, task_result in all_results.items():
        train_game = task_result['training_game']
        train_opp = task_result['training_opp_range']
        
        # Get test results (stored as 'test_summaries')
        test_summaries = task_result.get('test_summaries', {})
        
        if not test_summaries:
            print(f"  Task {task_id:2d}: No test results found")
            continue
        
        for condition_name, summary in test_summaries.items():
            # Determine test game and opponent from condition name
            # Conditions: baseline, same_game_low/mid_low/mid_high/high, GAME_same_opponents, GAME_low/high
            
            if 'baseline' in condition_name:
                test_game = train_game
                test_opp = 'baseline'
                test_opp_list = task_result['training_opp_probs']
            elif 'same_game' in condition_name:
                test_game = train_game
                # Extract opponent range from condition name (e.g., same_game_mid_low)
                if '_low' in condition_name and '_mid_low' not in condition_name:
                    test_opp = 'low'
                    test_opp_list = [0.1, 0.2, 0.3, 0.4]
                elif '_mid_low' in condition_name:
                    test_opp = 'mid_low'
                    test_opp_list = [0.3, 0.4, 0.5, 0.6]
                elif '_mid_high' in condition_name:
                    test_opp = 'mid_high'
                    test_opp_list = [0.5, 0.6, 0.7, 0.8]
                elif '_high' in condition_name:
                    test_opp = 'high'
                    test_opp_list = [0.7, 0.8, 0.9, 1.0]
                else:
                    test_opp = 'unknown'
                    test_opp_list = []
            else:
                # New game or cross generalization - extract game name
                for game in ['prisoners-dilemma', 'hawk-dove', 'stag-hunt', 'battle-of-sexes']:
                    if game in condition_name:
                        test_game = game
                        break
                else:
                    test_game = 'unknown'
                
                # Extract opponent range from condition name
                if 'same_opponents' in condition_name:
                    # Same opponents as training
                    test_opp = train_opp
                    test_opp_list = task_result['training_opp_probs']
                elif '_low' in condition_name and '_mid_low' not in condition_name:
                    test_opp = 'low'
                    test_opp_list = [0.1, 0.2, 0.3, 0.4]
                elif '_mid_low' in condition_name:
                    test_opp = 'mid_low'
                    test_opp_list = [0.3, 0.4, 0.5, 0.6]
                elif '_mid_high' in condition_name:
                    test_opp = 'mid_high'
                    test_opp_list = [0.5, 0.6, 0.7, 0.8]
                elif '_high' in condition_name:
                    test_opp = 'high'
                    test_opp_list = [0.7, 0.8, 0.9, 1.0]
                else:
                    test_opp = 'unknown'
                    test_opp_list = []
            
            # Get cooperation rate from summary
            coop_rate = summary.get('coop_rate', 0.0)
            reward = summary.get('reward', 0.0)
            
            # Note: We don't have opponent cooperation in the summaries
            # We'll need to calculate average opponent coop based on test_opp_list
            if test_opp == 'baseline':
                opponent_coop = 1.0 - np.mean(task_result['training_opp_probs'])
            elif test_opp_list:
                opponent_coop = 1.0 - np.mean(test_opp_list)
            else:
                opponent_coop = 0.5  # Unknown
            
            all_action_data.append({
                'task_id': task_id,
                'train_game': train_game,
                'train_opp': train_opp,
                'condition': condition_name,
                'test_game': test_game,
                'test_opp': test_opp,
                'agent_coop_rate': coop_rate,
                'opponent_coop_rate': opponent_coop,
                'agent_defect_rate': 1.0 - coop_rate,
                'opponent_defect_rate': 1.0 - opponent_coop,
                'avg_reward': reward
            })
        
        print(f"  Task {task_id:2d}: {train_game:20s} + {train_opp:8s} ({len(test_summaries)} test conditions)")
    
    return all_action_data

def create_cooperation_heatmaps(all_data):
    """Create heatmaps showing agent and opponent cooperation rates side by side."""
    
    df = pd.DataFrame(all_data)
    
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt', 'battle-of-sexes']
    game_labels = ['PD', 'HD', 'SH', 'BS']
    game_full_names = ['Prisoner\'s Dilemma', 'Hawk-Dove', 'Stag-Hunt', 'Battle of Sexes']
    opp_ranges = ['low', 'mid_low', 'mid_high', 'high']
    opp_labels = ['Low', 'ML', 'MH', 'High']
    
    # Create one plot per training game with agent/opponent side-by-side
    for game_idx, train_game in enumerate(games):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        train_data = df[df['train_game'] == train_game]
        
        for j, train_opp in enumerate(opp_ranges):
            train_opp_data = train_data[train_data['train_opp'] == train_opp]
            
            if len(train_opp_data) > 0:
                # Create matrices for agent and opponent cooperation
                agent_matrix = np.zeros((len(games), len(opp_ranges)))
                opp_matrix = np.zeros((len(games), len(opp_ranges)))
                
                for _, row in train_opp_data.iterrows():
                    try:
                        test_game_idx = games.index(row['test_game'])
                    except ValueError:
                        continue
                    
                    # Map test_opp to index
                    test_opp_str = row['test_opp']
                    for opp_idx, opp_range in enumerate(opp_ranges):
                        if opp_range in test_opp_str:
                            agent_matrix[test_game_idx, opp_idx] = row['agent_coop_rate']
                            opp_matrix[test_game_idx, opp_idx] = row['opponent_coop_rate']
                            break
                
                # Plot agent cooperation
                ax_agent = axes[0, j]
                im_agent = ax_agent.imshow(agent_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
                ax_agent.set_xticks(range(len(opp_ranges)))
                ax_agent.set_yticks(range(len(games)))
                ax_agent.set_xticklabels(opp_labels, fontsize=11)
                ax_agent.set_yticklabels(game_labels, fontsize=11)
                ax_agent.set_title(f"Training: {opp_labels[j]} Opponents", fontsize=12, fontweight='bold')
                
                if j == 0:
                    ax_agent.set_ylabel('Agent Cooperation\n\nTest Game', fontsize=12, fontweight='bold')
                
                # Add values as text
                for i in range(len(games)):
                    for k in range(len(opp_ranges)):
                        val = agent_matrix[i, k]
                        color = 'white' if val < 0.5 else 'black'
                        ax_agent.text(k, i, f'{val:.2f}', ha='center', va='center',
                                    color=color, fontsize=9, fontweight='bold')
                
                # Plot opponent cooperation
                ax_opp = axes[1, j]
                im_opp = ax_opp.imshow(opp_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
                ax_opp.set_xticks(range(len(opp_ranges)))
                ax_opp.set_yticks(range(len(games)))
                ax_opp.set_xticklabels(opp_labels, fontsize=11)
                ax_opp.set_yticklabels(game_labels, fontsize=11)
                ax_opp.set_xlabel('Test Opponent Range', fontsize=11)
                
                if j == 0:
                    ax_opp.set_ylabel('Opponent Cooperation\n\nTest Game', fontsize=12, fontweight='bold')
                
                # Add values as text
                for i in range(len(games)):
                    for k in range(len(opp_ranges)):
                        val = opp_matrix[i, k]
                        color = 'white' if val < 0.5 else 'black'
                        ax_opp.text(k, i, f'{val:.2f}', ha='center', va='center',
                                  color=color, fontsize=9, fontweight='bold')
        
        # Add colorbars
        fig.colorbar(im_agent, ax=axes[0, :], orientation='vertical', 
                    label='Cooperation Rate', pad=0.01, fraction=0.046)
        fig.colorbar(im_opp, ax=axes[1, :], orientation='vertical',
                    label='Cooperation Rate', pad=0.01, fraction=0.046)
        
        plt.suptitle(f'{game_full_names[game_idx]} Agents: Cooperation Rates\n' +
                    'Top: Agent Cooperation | Bottom: Opponent Cooperation',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        filename = f'cooperation_heatmap_{game_labels[game_idx]}.png'
        plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved {filename}")

def main():
    print("="*80)
    print("ACTION DISTRIBUTION ANALYSIS")
    print("="*80)
    print()
    
    all_action_data = load_from_cached_results()
    
    print()
    print(f"✓ Loaded {len(all_action_data)} test conditions")
    print()
    
    # Create visualizations
    print("Creating cooperation heatmaps...")
    create_cooperation_heatmaps(all_action_data)
    print()
    
    # Save raw data
    df = pd.DataFrame(all_action_data)
    df.to_csv(OUTPUT_DIR / 'action_distributions.csv', index=False)
    print(f"✓ Saved action_distributions.csv")
    print()
    
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}/")
    print()
    print("Generated files:")
    print("  - cooperation_heatmap_PD.png (agent/opponent cooperation heatmaps)")
    print("  - cooperation_heatmap_HD.png")
    print("  - cooperation_heatmap_SH.png")
    print("  - cooperation_heatmap_BS.png")
    print("  - action_distributions.csv (raw data)")

if __name__ == "__main__":
    main()
