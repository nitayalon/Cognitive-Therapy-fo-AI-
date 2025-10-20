"""
Example script demonstrating the ToM-RL (Theory of Mind Reinforcement Learning) architecture.

This script shows how to:
1. Create a ToM-RL network with policy and opponent prediction heads
2. Use the ToM-RL loss function
3. Train on a simple game scenario
4. Analyze the Theory of Mind component contribution

This is a minimal example for understanding the new architecture.
"""

import torch
import numpy as np
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cognitive_therapy_ai.games import GameFactory, Action
from cognitive_therapy_ai.opponent import OpponentFactory
from cognitive_therapy_ai.network import GameLSTM
from cognitive_therapy_ai.tom_rl_loss import ToMRLLoss, LossAnalyzer


def create_tom_rl_network(input_size: int = 5) -> GameLSTM:
    """Create a ToM-RL network."""
    network = GameLSTM(
        input_size=input_size,
        hidden_size=64,  # Smaller for demo
        num_layers=2,
        dropout=0.1,
        num_actions=2
    )
    return network


def simulate_game_episode(network: GameLSTM, game, opponent, num_games: int = 10, training: bool = False):
    """
    Simulate a game episode and collect training data.
    
    Args:
        network: The LSTM network
        game: The game instance
        opponent: The opponent instance
        num_games: Number of games to play
        training: If True, keep gradients for training; if False, disable gradients for evaluation
    
    Returns:
        Dictionary with training data for ToM-RL loss
    """
    # Storage for episode data
    states = []
    actions = []
    rewards = []
    opponent_actions = []
    policy_logits_list = []
    opponent_coop_probs_list = []
    value_estimates_list = []
    
    # Initialize hidden state
    hidden = network.init_hidden(1, torch.device('cpu'))
    game.reset()
    opponent.reset()
    
    for game_num in range(num_games):
        # Get current state
        state_vector = game.get_state_vector()
        state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0).unsqueeze(0)
        
        # Network forward pass - enable gradients if training
        if training:
            # Keep gradients for training
            policy_logits, opponent_coop_prob, value_estimate, new_hidden = network.forward(
                state_tensor, hidden
            )
        else:
            # Disable gradients for evaluation
            with torch.no_grad():
                policy_logits, opponent_coop_prob, value_estimate, new_hidden = network.forward(
                    state_tensor, hidden
                )
        
        # Sample action from policy
        action_probs = torch.softmax(policy_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action_idx = action_dist.sample()
        
        # Convert to Action enum (detach for game interaction)
        player_action = Action.COOPERATE if action_idx.detach().item() == 0 else Action.DEFECT
        
        # Opponent chooses action
        opponent_action = opponent.play_action(game.history, game_num)
        
        # Play the round
        player_reward, opponent_reward = game.play_round(player_action, opponent_action)
        opponent.update_payoff(opponent_reward)
        
        # Store data
        states.append(state_vector)
        actions.append(action_idx.detach().item())  # Detach for storage as Python int
        rewards.append(player_reward)
        opponent_actions.append(1 if opponent_action == Action.COOPERATE else 0)
        
        # Store tensors with proper shapes for later stacking
        # The network outputs should have batch dimension, we need to squeeze sequence dimension only
        policy_logits_list.append(policy_logits.squeeze(0) if policy_logits.dim() > 1 else policy_logits)
        opponent_coop_probs_list.append(opponent_coop_prob.squeeze(0) if opponent_coop_prob.dim() > 1 else opponent_coop_prob)
        value_estimates_list.append(value_estimate.squeeze(0) if value_estimate.dim() > 1 else value_estimate)
        
        # Update hidden state
        hidden = new_hidden
    
    # Convert to tensors - handle different tensor dimensions properly
    training_data = {}
    
    # Stack policy logits (should be (num_games, num_actions))
    if policy_logits_list and policy_logits_list[0].dim() > 0:
        training_data['policy_logits'] = torch.stack(policy_logits_list, dim=0)
    else:
        # If 0-dim tensors, stack and reshape
        training_data['policy_logits'] = torch.stack([p.unsqueeze(0) for p in policy_logits_list], dim=0)
    
    # Stack opponent cooperation probabilities (should be (num_games, 1))  
    if opponent_coop_probs_list and opponent_coop_probs_list[0].dim() > 0:
        training_data['opponent_coop_probs'] = torch.stack(opponent_coop_probs_list, dim=0)
        if training_data['opponent_coop_probs'].dim() == 1:
            training_data['opponent_coop_probs'] = training_data['opponent_coop_probs'].unsqueeze(-1)
    else:
        # If 0-dim tensors, stack and add dimension
        training_data['opponent_coop_probs'] = torch.stack([o.unsqueeze(0) for o in opponent_coop_probs_list], dim=0).unsqueeze(-1)
    
    # Stack value estimates (should be (num_games, 1))
    if value_estimates_list and value_estimates_list[0].dim() > 0:
        training_data['value_estimates'] = torch.stack(value_estimates_list, dim=0)
        if training_data['value_estimates'].dim() == 1:
            training_data['value_estimates'] = training_data['value_estimates'].unsqueeze(-1)
    else:
        # If 0-dim tensors, stack and add dimension
        training_data['value_estimates'] = torch.stack([v.unsqueeze(0) for v in value_estimates_list], dim=0).unsqueeze(-1)
    
    # Simple 1D tensors
    training_data['actions_taken'] = torch.tensor(actions, dtype=torch.long)
    training_data['rewards'] = torch.tensor(rewards, dtype=torch.float32)
    training_data['opponent_actions'] = torch.tensor(opponent_actions, dtype=torch.long)
    
    return training_data


def demo_tom_rl_training():
    """Demonstrate ToM-RL training process."""
    print("="*60)
    print("ToM-RL (Theory of Mind Reinforcement Learning) Demo")
    print("="*60)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create components
    print("\n1. Creating Game...")
    game = GameFactory.create_game('prisoners-dilemma')
    
    print("\n2. Creating ToM-RL Network...")
    network = create_tom_rl_network(input_size=game.get_state_size())
    print(f"   - Network created with {sum(p.numel() for p in network.parameters())} parameters")
    print(f"   - Architecture: LSTM + Policy Head + Opponent Prediction Head + Value Head")
    print(f"   - Input size: {game.get_state_size()} (payoff matrix + metadata)")
    
    print("\n3. Creating Opponents...")
    opponents = OpponentFactory.create_opponent_set([0.3, 0.7])  # Two opponent types
    print(f"   - Game: {game.name}")
    print(f"   - Opponents: {[opp.get_strategy_name() for opp in opponents]}")
    
    print("\n4. Creating ToM-RL Loss Function...")
    loss_fn = ToMRLLoss(alpha=1.0, gamma=0.99, use_gae=True)
    loss_analyzer = LossAnalyzer()
    print(f"   - Alpha (ToM weight): {loss_fn.alpha}")
    print(f"   - Using GAE for advantage estimation")
    
    print("\n5. Running Training Episodes...")
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    
    for epoch in range(5):  # Small number for demo
        network.train()
        epoch_losses = []
        
        for opponent in opponents:
            # Simulate episode with gradients enabled for training
            training_data = simulate_game_episode(network, game, opponent, num_games=20, training=True)
            
            # Calculate loss
            loss_dict = loss_fn(
                training_data['policy_logits'],
                training_data['opponent_coop_probs'], 
                training_data['value_estimates'],
                training_data['actions_taken'],
                training_data['rewards'],
                training_data['opponent_actions']
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss_dict)
        
        # Average losses for epoch
        avg_total_loss = np.mean([l['total_loss'].item() for l in epoch_losses])
        avg_rl_loss = np.mean([l['rl_loss'].item() for l in epoch_losses])
        avg_tom_loss = np.mean([l['opponent_prediction_loss'].item() for l in epoch_losses])
        
        # Record for analysis
        avg_loss_dict = {
            'total_loss': torch.tensor(avg_total_loss),
            'rl_loss': torch.tensor(avg_rl_loss),
            'opponent_prediction_loss': torch.tensor(avg_tom_loss),
            'alpha': loss_fn.alpha
        }
        loss_analyzer.record_loss(avg_loss_dict, epoch)
        
        print(f"   Epoch {epoch+1}: Total={avg_total_loss:.4f}, RL={avg_rl_loss:.4f}, ToM={avg_tom_loss:.4f}")
    
    print("\n6. Training Analysis...")
    loss_balance = loss_analyzer.get_loss_balance_ratio()
    tom_contribution = loss_analyzer.get_tom_contribution()
    
    print(f"   - Loss Balance Ratio (RL/ToM): {loss_balance:.3f}")
    print(f"   - ToM Contribution to Total Loss: {tom_contribution:.1%}")
    
    if 0.5 < loss_balance < 2.0:
        print("Good balance between RL and ToM objectives")
    elif loss_balance > 2.0:
        print("RL loss dominates - consider increasing α")
    else:
        print("ToM loss dominates - consider decreasing α")
    
    print("\n7. Testing Opponent Prediction...")
    network.eval()
    test_opponents = OpponentFactory.create_opponent_set([0.1, 0.9])  # Easy and hard to predict
    
    for test_opponent in test_opponents:
        # Simulate short episode for evaluation (no gradients needed)
        test_data = simulate_game_episode(network, game, test_opponent, num_games=10, training=False)
        
        # Check prediction accuracy
        predicted_coop_probs = test_data['opponent_coop_probs'].detach().numpy().flatten()
        actual_coop_actions = test_data['opponent_actions'].numpy()
        
        # Calculate accuracy (threshold at 0.5)
        predictions = (predicted_coop_probs > 0.5).astype(int)
        accuracy = np.mean(predictions == actual_coop_actions)
        
        # Calculate average predicted vs actual cooperation rate
        avg_predicted = np.mean(predicted_coop_probs)
        avg_actual = np.mean(actual_coop_actions)
        
        print(f"   - {test_opponent.get_strategy_name()}:")
        print(f"     Prediction Accuracy: {accuracy:.1%}")
        print(f"     Predicted Coop Rate: {avg_predicted:.3f}")
        print(f"     Actual Coop Rate: {avg_actual:.3f}")
    
    print("\n" + "="*60)
    print("Demo completed! The network learned to:")
    print("- Maximize rewards through policy learning (RL component)")
    print("- Predict opponent behavior (ToM component)")  
    print("- Balance both objectives for social decision-making")
    print("="*60)


if __name__ == "__main__":
    try:
        demo_tom_rl_training()
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Make sure to install requirements: pip install -r requirements.txt")
        print("And install the package: pip install -e .")