"""
Compute Game Complexity Metrics M1 and M2.

M1(g): Number of different Best Response (BR) policies against fixed opponents
M2(g): Mean KL divergence between S1 BR and S2 BBR policies

S1 = Task-opponent setup (trained on specific opponent)
S2 = Whole population setup (trained on all opponents)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from collections import defaultdict

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150


def load_final_policy(checkpoint_path, device='cpu'):
    """Load a trained model and extract its final policy distribution."""
    import sys
    from pathlib import Path
    
    # Add src to path
    src_path = Path(__file__).parent.parent / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from cognitive_therapy_ai.network import GameLSTM
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model architecture params
    model_config = checkpoint.get('model_config', {})
    input_size = model_config.get('input_size', 9)
    hidden_size = model_config.get('hidden_size', 128)
    num_layers = model_config.get('num_layers', 2)
    dropout = model_config.get('dropout', 0.1)
    
    # Create and load model
    model = GameLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def extract_policy_distribution(model, game_name, opponent_defect_prob, num_samples=1000, device='cpu'):
    """
    Extract average policy distribution for a game-opponent pair.
    
    Returns: numpy array of shape (2,) representing P(cooperate), P(defect)
    """
    import sys
    from pathlib import Path
    
    # Add src to path
    src_path = Path(__file__).parent.parent / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from cognitive_therapy_ai.games import GameFactory
    
    # Create game
    game = GameFactory.create_game(game_name)
    payoff_matrix = game.get_payoff_matrix().flatten()
    
    # Generate diverse states
    policies = []
    
    for round_num in np.linspace(0, 1, 20):  # Sample across round numbers
        for prev_opp_action in [0, 1]:  # Both actions
            for prev_agent_action in [0, 1]:
                for prev_agent_reward in np.linspace(-5, 5, 5):
                    for prev_opp_reward in np.linspace(-5, 5, 5):
                        # Create state
                        state = torch.tensor([
                            *payoff_matrix,  # 4 elements
                            round_num,       # 1 element
                            prev_opp_action, # 1 element
                            prev_agent_action, # 1 element
                            prev_agent_reward, # 1 element
                            prev_opp_reward    # 1 element
                        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 9)
                        
                        with torch.no_grad():
                            policy_logits, _, _, _ = model(state.to(device))
                            policy_probs = F.softmax(policy_logits, dim=-1).squeeze().cpu().numpy()
                            policies.append(policy_probs)
    
    # Average policy across all states
    avg_policy = np.mean(policies, axis=0)
    
    return avg_policy


def compute_kl_divergence(p, q):
    """
    Compute KL divergence D_KL(p || q).
    
    Args:
        p, q: Probability distributions (arrays of same length)
    
    Returns:
        KL divergence value
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    
    # Normalize to ensure they sum to 1
    p = p / p.sum()
    q = q / q.sum()
    
    return entropy(p, q)


def compute_policy_similarity(policy1, policy2, threshold=0.1):
    """
    Check if two policies are similar (same BR).
    
    Returns: True if policies are similar
    """
    # Policies are similar if they have similar cooperation probabilities
    diff = np.abs(policy1[0] - policy2[0])  # Compare P(cooperate)
    return diff < threshold


def main():
    device = torch.device('cpu')
    
    # Configuration
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    game_display_names = {'prisoners-dilemma': 'PD', 'hawk-dove': 'HD', 'stag-hunt': 'SH'}
    opponent_probs = [0.1, 0.3, 0.5, 0.7, 0.9]  # Defection probabilities
    
    # Find most recent task-opponent training experiment
    experiments_dir = Path('experiments')
    train_dirs = sorted(experiments_dir.glob('generalization_matrix_train_*'))
    
    if not train_dirs:
        print("ERROR: No task-opponent training experiments found")
        return
    
    train_dir = train_dirs[-1] / 'training'
    print(f"Using training data from: {train_dir.parent.name}\n")
    
    # Storage for results
    results = {
        'game': [],
        'M1': [],  # Number of unique BR policies
        'M2': []   # Mean KL divergence
    }
    
    for game in games:
        print(f"{'='*60}")
        print(f"Processing {game_display_names[game]}...")
        print(f"{'='*60}")
        
        # Extract policies for each opponent type (S1 - task-opponent setup)
        s1_policies = {}
        
        for opp_idx, opp_prob in enumerate(opponent_probs):
            # Find condition ID for this game-opponent pair
            condition_id = games.index(game) * 5 + opp_idx
            
            # Find checkpoint (use seed 0 for simplicity)
            checkpoint_pattern = f'condition_{condition_id}_seed_0/generalization_matrix_task_{condition_id}_*/checkpoints/{game}_final_checkpoint.pth'
            checkpoint_files = list(train_dir.glob(checkpoint_pattern))
            
            if not checkpoint_files:
                print(f"  Warning: No checkpoint found for condition {condition_id} (opponent {opp_prob})")
                continue
            
            checkpoint_path = checkpoint_files[0]
            
            # Load model and extract policy
            try:
                model = load_final_policy(checkpoint_path, device)
                policy = extract_policy_distribution(model, game, opp_prob, device=device)
                s1_policies[opp_prob] = policy
                print(f"  Opponent {opp_prob:.1f}: P(coop)={policy[0]:.4f}, P(defect)={policy[1]:.4f}")
            except Exception as e:
                print(f"  Error loading policy for opponent {opp_prob}: {e}")
        
        if len(s1_policies) == 0:
            print(f"  ERROR: No policies loaded for {game}")
            continue
        
        # Compute M1: Number of unique BR policies
        unique_policies = []
        for opp_prob, policy in s1_policies.items():
            is_unique = True
            for unique_policy in unique_policies:
                if compute_policy_similarity(policy, unique_policy):
                    is_unique = False
                    break
            if is_unique:
                unique_policies.append(policy)
        
        m1 = len(unique_policies)
        print(f"\n  M1 (# unique BR policies): {m1}")
        
        # Compute S2 BBR (Bayesian Best Response) - average policy across all opponents
        s2_bbr = np.mean(list(s1_policies.values()), axis=0)
        print(f"  S2 BBR: P(coop)={s2_bbr[0]:.4f}, P(defect)={s2_bbr[1]:.4f}")
        
        # Compute M2: Mean KL divergence between each S1 BR and S2 BBR
        kl_divergences = []
        for opp_prob, s1_policy in s1_policies.items():
            kl_div = compute_kl_divergence(s1_policy, s2_bbr)
            kl_divergences.append(kl_div)
            print(f"  KL(S1_{opp_prob:.1f} || S2): {kl_div:.4f}")
        
        m2 = np.mean(kl_divergences)
        print(f"\n  M2 (mean KL divergence): {m2:.4f}")
        
        # Store results
        results['game'].append(game_display_names[game])
        results['M1'].append(m1)
        results['M2'].append(m2)
        print()
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    print("\n" + "="*60)
    print("SUMMARY: Game Complexity Metrics")
    print("="*60)
    
    if len(df_results) == 0:
        print("ERROR: No results computed. Check that checkpoints exist.")
        return
    
    print(df_results.to_string(index=False))
    print()
    
    # Create scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = {'PD': 'orange', 'HD': 'blue', 'SH': 'green'}
    
    for _, row in df_results.iterrows():
        game = row['game']
        ax.scatter(row['M2'], row['M1'], s=200, c=colors[game], 
                  edgecolors='black', linewidths=2, label=game, alpha=0.7, zorder=3)
        
        # Add game label next to point
        ax.annotate(game, (row['M2'], row['M1']), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=12, fontweight='bold')
    
    ax.set_xlabel('M2', fontsize=12, fontweight='bold')
    ax.set_ylabel('M1', fontsize=12, fontweight='bold')
    ax.set_title('Game Complexity Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    
    # Set reasonable axis limits
    ax.set_xlim(0, max(df_results['M2']) + 0.2)
    ax.set_ylim(0, max(df_results['M1']) * 1.2)
    
    plt.tight_layout()
    
    # Save results
    output_dir = Path('Results/game_complexity')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_results.to_csv(output_dir / 'complexity_metrics.csv', index=False)
    fig.savefig(output_dir / 'complexity_scatter.png', dpi=300, bbox_inches='tight')
    
    print(f"Saved results to: {output_dir}")
    print(f"  - complexity_metrics.csv")
    print(f"  - complexity_scatter.png")
    
    plt.show()


if __name__ == '__main__':
    main()
