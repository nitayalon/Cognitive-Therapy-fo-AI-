"""
Test script to verify the network input modification (adding opponent's previous action).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from cognitive_therapy_ai import GameFactory, GameLSTM, OpponentFactory

def test_state_vector_size():
    """Test that state vector has correct size (6 elements)."""
    print("=" * 60)
    print("TEST 1: State Vector Size")
    print("=" * 60)
    
    # Create a game
    game = GameFactory.create_game('prisoners-dilemma')
    
    # Check state size
    state_size = game.get_state_size()
    print(f"State size: {state_size}")
    assert state_size == 6, f"Expected state size 6, got {state_size}"
    print("✓ State size is correct (6 elements)")
    
    # Get initial state vector (first trial, no previous opponent action)
    state_vector = game.get_state_vector()
    print(f"Initial state vector shape: {state_vector.shape}")
    print(f"Initial state vector: {state_vector}")
    assert state_vector.shape == (6,), f"Expected shape (6,), got {state_vector.shape}"
    
    # Check that the last element is -1.0 (no previous action)
    assert state_vector[-1] == -1.0, f"Expected opponent_prev_action=-1.0 for first trial, got {state_vector[-1]}"
    print("✓ Initial state has opponent_prev_action = -1.0 (correct)")
    print()

def test_state_after_action():
    """Test that opponent's previous action is correctly recorded."""
    print("=" * 60)
    print("TEST 2: Opponent Previous Action Recording")
    print("=" * 60)
    
    from cognitive_therapy_ai.games import Action
    
    # Create a game
    game = GameFactory.create_game('prisoners-dilemma')
    
    # Play a round with opponent cooperating
    game.play_round(Action.COOPERATE, Action.COOPERATE)
    
    # Get state after first round
    state_vector = game.get_state_vector()
    print(f"State after round 1 (opponent cooperated): {state_vector}")
    
    # Check that opponent's previous action is recorded as 0 (COOPERATE)
    assert state_vector[-1] == 0.0, f"Expected opponent_prev_action=0.0 (COOPERATE), got {state_vector[-1]}"
    print("✓ Opponent's COOPERATE action correctly recorded as 0.0")
    
    # Play another round with opponent defecting
    game.play_round(Action.DEFECT, Action.DEFECT)
    
    # Get state after second round
    state_vector = game.get_state_vector()
    print(f"State after round 2 (opponent defected): {state_vector}")
    
    # Check that opponent's previous action is recorded as 1 (DEFECT)
    assert state_vector[-1] == 1.0, f"Expected opponent_prev_action=1.0 (DEFECT), got {state_vector[-1]}"
    print("✓ Opponent's DEFECT action correctly recorded as 1.0")
    print()

def test_network_input_compatibility():
    """Test that network can process the new 6-element input."""
    print("=" * 60)
    print("TEST 3: Network Input Compatibility")
    print("=" * 60)
    
    # Create network with input_size=6
    network = GameLSTM(input_size=6, hidden_size=64, num_layers=1, dropout=0.0)
    print(f"Network created with input_size=6")
    
    # Create a game and get state
    game = GameFactory.create_game('prisoners-dilemma')
    state_vector = game.get_state_vector()
    
    # Convert to tensor
    state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0).unsqueeze(0)
    print(f"State tensor shape: {state_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        policy_logits, opponent_policy_logits, value_estimate, hidden = network.forward(state_tensor)
    
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Opponent policy logits shape: {opponent_policy_logits.shape}")
    print(f"Value estimate shape: {value_estimate.shape}")
    
    assert policy_logits.shape == (1, 2), f"Expected policy_logits shape (1, 2), got {policy_logits.shape}"
    assert opponent_policy_logits.shape == (1, 2), f"Expected opponent_policy_logits shape (1, 2), got {opponent_policy_logits.shape}"
    assert value_estimate.shape == (1, 1), f"Expected value_estimate shape (1, 1), got {value_estimate.shape}"
    
    print("✓ Network successfully processes 6-element input")
    print()

def test_full_game_sequence():
    """Test a full game sequence with state evolution."""
    print("=" * 60)
    print("TEST 4: Full Game Sequence")
    print("=" * 60)
    
    from cognitive_therapy_ai.games import Action
    
    # Create game and network
    game = GameFactory.create_game('prisoners-dilemma')
    network = GameLSTM(input_size=6, hidden_size=64, num_layers=1, dropout=0.0)
    
    # Create opponent
    opponent = OpponentFactory.create_probabilistic_opponent(defection_probability=0.5)
    
    print("Playing 5 rounds...")
    hidden = None
    
    for round_num in range(5):
        # Get state
        state_vector = game.get_state_vector()
        state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0).unsqueeze(0)
        
        # Network forward pass
        with torch.no_grad():
            policy_logits, _, _, hidden = network.forward(state_tensor, hidden)
        
        # Sample action
        action_probs = torch.softmax(policy_logits, dim=-1)
        action_idx = torch.argmax(action_probs, dim=-1).item()
        player_action = Action(action_idx)
        
        # Opponent action
        opponent_action = opponent.play_action(game.history, round_num)
        
        # Play round
        player_payoff, opponent_payoff = game.play_round(player_action, opponent_action)
        
        # Check state has correct opponent_prev_action
        next_state = game.get_state_vector()
        expected_prev_action = float(opponent_action.value)
        
        print(f"  Round {round_num + 1}: Player={player_action.name}, Opponent={opponent_action.name}, "
              f"Next state opponent_prev_action={next_state[-1]:.1f}")
        
        assert next_state[-1] == expected_prev_action, \
            f"Round {round_num}: Expected opponent_prev_action={expected_prev_action}, got {next_state[-1]}"
    
    print("✓ All 5 rounds completed successfully with correct state evolution")
    print()

def test_all_games():
    """Test that all game types work with the new input."""
    print("=" * 60)
    print("TEST 5: All Game Types")
    print("=" * 60)
    
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    
    for game_name in games:
        game = GameFactory.create_game(game_name)
        state_size = game.get_state_size()
        state_vector = game.get_state_vector()
        
        print(f"  {game_name}: state_size={state_size}, state_shape={state_vector.shape}")
        assert state_size == 6, f"{game_name}: Expected state_size=6, got {state_size}"
        assert state_vector.shape == (6,), f"{game_name}: Expected shape (6,), got {state_vector.shape}"
        assert state_vector[-1] == -1.0, f"{game_name}: Expected opponent_prev_action=-1.0 initially"
    
    print("✓ All game types work correctly with 6-element state")
    print()

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TESTING NETWORK INPUT MODIFICATION")
    print("Adding opponent's previous action to state vector")
    print("=" * 60 + "\n")
    
    try:
        test_state_vector_size()
        test_state_after_action()
        test_network_input_compatibility()
        test_full_game_sequence()
        test_all_games()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nSummary:")
        print("- State vector now has 6 elements (payoff_matrix + round_number + opponent_prev_action)")
        print("- First trial: opponent_prev_action = -1.0 (no previous action)")
        print("- Subsequent trials: opponent_prev_action = 0.0 (COOPERATE) or 1.0 (DEFECT)")
        print("- Network successfully processes 6-element input")
        print("- All game types work correctly")
        print()
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
