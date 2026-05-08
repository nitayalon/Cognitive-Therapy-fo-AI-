#!/usr/bin/env python3
"""
Test script to verify 9-element input modification with separate embedding pathways.

Tests that the network correctly processes the expanded input including:
- Payoff matrix (4 elements) - ENVIRONMENTAL INPUT
- Round number (1 element) - ENVIRONMENTAL INPUT
- Opponent's previous action (1 element) - SOCIAL INPUT
- Agent's previous action (1 element) - SOCIAL INPUT
- Agent's previous reward (1 element) - SOCIAL INPUT
- Opponent's previous reward (1 element) - SOCIAL INPUT
Total: 9 elements

The network uses separate embedding layers for each component, allowing analysis
of which inputs (environmental vs social) the network learns to use.

Author: Research Team
Date: May 2026
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from cognitive_therapy_ai import (
    GameFactory,
    GameLSTM,
    OpponentFactory
)
from cognitive_therapy_ai.config import NetworkConfig
from cognitive_therapy_ai.games import Action


def test_state_size():
    """Test that all games return correct state size."""
    print("Test 1: Verifying state size...")
    
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    for game_name in games:
        game = GameFactory.create_game(game_name)
        state_size = game.get_state_size()
        assert state_size == 9, f"{game_name} state size is {state_size}, expected 9"
        print(f"  ✓ {game_name}: state_size = {state_size}")
    
    print("  ✓ All games return state_size = 9\n")


def test_first_trial_state():
    """Test state vector on first trial (no history)."""
    print("Test 2: First trial state vector (no history)...")
    
    game = GameFactory.create_game('prisoners-dilemma')
    state = game.get_state_vector()
    
    assert len(state) == 9, f"State length is {len(state)}, expected 9"
    
    # First 4 elements: payoff matrix
    payoff_matrix = game.get_payoff_matrix().flatten()
    assert np.allclose(state[:4], payoff_matrix), "Payoff matrix mismatch"
    
    # Element 5: round number (should be 0 normalized)
    assert state[4] == 0.0, f"Round number is {state[4]}, expected 0.0"
    
    # Element 6: opponent_prev_action (should be -1.0 for first trial)
    assert state[5] == -1.0, f"Opponent prev action is {state[5]}, expected -1.0"
    
    # Element 7: agent_prev_action (should be -1.0 for first trial)
    assert state[6] == -1.0, f"Agent prev action is {state[6]}, expected -1.0"
    
    # Element 8: agent_prev_reward (should be 0.0 for first trial)
    assert state[7] == 0.0, f"Agent prev reward is {state[7]}, expected 0.0"
    
    # Element 9: opponent_prev_reward (should be 0.0 for first trial)
    assert state[8] == 0.0, f"Opponent prev reward is {state[8]}, expected 0.0"
    
    print(f"  ✓ State vector: {state}")
    print(f"    - Payoff matrix (4): {state[:4]}")
    print(f"    - Round number (1): {state[4]}")
    print(f"    - Opponent prev action (1): {state[5]}")
    print(f"    - Agent prev action (1): {state[6]}")
    print(f"    - Agent prev reward (1): {state[7]}")
    print(f"    - Opponent prev reward (1): {state[8]}")
    print()


def test_after_actions():
    """Test state vector after playing actions."""
    print("Test 3: State vector after playing actions...")
    
    game = GameFactory.create_game('prisoners-dilemma')
    
    # Play first round: both cooperate
    agent_action = Action.COOPERATE
    opponent_action = Action.COOPERATE
    agent_reward, opponent_reward = game.play_round(agent_action, opponent_action)
    
    state = game.get_state_vector()
    
    assert len(state) == 9, f"State length is {len(state)}, expected 9"
    
    # Round number should be 1 (normalized to 0.01)
    assert state[4] == 0.01, f"Round number is {state[4]}, expected 0.01"
    
    # Opponent prev action should be 0 (COOPERATE)
    assert state[5] == 0.0, f"Opponent prev action is {state[5]}, expected 0.0"
    
    # Agent prev action should be 0 (COOPERATE)
    assert state[6] == 0.0, f"Agent prev action is {state[6]}, expected 0.0"
    
    # Agent prev reward should match the actual reward
    assert state[7] == agent_reward, f"Agent prev reward is {state[7]}, expected {agent_reward}"
    
    # Opponent prev reward should match the actual reward
    assert state[8] == opponent_reward, f"Opponent prev reward is {state[8]}, expected {opponent_reward}"
    
    print(f"  ✓ After round 1 (both cooperate):")
    print(f"    - Agent reward: {agent_reward}, Opponent reward: {opponent_reward}")
    print(f"    - State[5] (opp prev action): {state[5]} (COOPERATE=0)")
    print(f"    - State[6] (agent prev action): {state[6]} (COOPERATE=0)")
    print(f"    - State[7] (agent prev reward): {state[7]}")
    print(f"    - State[8] (opponent prev reward): {state[8]}")
    
    # Play second round: agent defects, opponent cooperates
    agent_action = Action.DEFECT
    opponent_action = Action.COOPERATE
    agent_reward2, opponent_reward2 = game.play_round(agent_action, opponent_action)
    
    state2 = game.get_state_vector()
    
    # Round number should be 2 (normalized to 0.02)
    assert state2[4] == 0.02, f"Round number is {state2[4]}, expected 0.02"
    
    # Opponent prev action should be 0 (COOPERATE from round 2, the most recent round)
    assert state2[5] == 0.0, f"Opponent prev action is {state2[5]}, expected 0.0 (COOPERATE from round 2)"
    
    # Agent prev action should be 1 (DEFECT from round 2, the most recent round)
    assert state2[6] == 1.0, f"Agent prev action is {state2[6]}, expected 1.0 (DEFECT from round 2)"
    
    # Agent prev reward should be from round 2
    assert state2[7] == agent_reward2, f"Agent prev reward is {state2[7]}, expected {agent_reward2} (from round 2)"
    
    # Opponent prev reward should be from round 2
    assert state2[8] == opponent_reward2, f"Opponent prev reward is {state2[8]}, expected {opponent_reward2} (from round 2)"
    
    print(f"\n  ✓ After round 2 (agent defects, opponent cooperates):")
    print(f"    - Agent reward: {agent_reward2}, Opponent reward: {opponent_reward2}")
    print(f"    - State[5] (opp prev action): {state2[5]} (from round 2: COOPERATE=0)")
    print(f"    - State[6] (agent prev action): {state2[6]} (from round 2: DEFECT=1)")
    print(f"    - State[7] (agent prev reward): {state2[7]} (from round 2)")
    print(f"    - State[8] (opponent prev reward): {state2[8]} (from round 2)")
    print()


def test_network_compatibility():
    """Test that network accepts 9-element input."""
    print("Test 4: Network compatibility with 9-element input...")
    
    # Create network with 9-element input
    network = GameLSTM(input_size=9, hidden_size=64, num_layers=2)
    
    # Create a batch of states
    batch_size = 4
    game = GameFactory.create_game('prisoners-dilemma')
    
    # Get state from game (should be 9 elements)
    state = game.get_state_vector()
    assert len(state) == 9, f"State length is {len(state)}, expected 9"
    
    # Create batch tensor - need to add sequence dimension
    # Shape: (batch_size, seq_len=1, input_size=9)
    states_batch = torch.FloatTensor([[state] for _ in range(batch_size)])
    
    # Forward pass
    policy_logits, opponent_coop_probs, value_estimates, _ = network(states_batch)
    
    assert policy_logits.shape == (batch_size, 2), f"Policy shape is {policy_logits.shape}"
    assert opponent_coop_probs.shape == (batch_size, 2), f"Opponent prob shape is {opponent_coop_probs.shape} (2 logits: defect/cooperate)"
    assert value_estimates.shape == (batch_size, 1), f"Value shape is {value_estimates.shape}"
    
    print(f"  ✓ Network successfully processes 9-element input with separate embeddings")
    print(f"    - Input shape: {states_batch.shape}")
    print(f"    - Policy logits shape: {policy_logits.shape}")
    print(f"    - Opponent cooperation probs shape: {opponent_coop_probs.shape}")
    print(f"    - Value estimates shape: {value_estimates.shape}")
    print(f"    - Embedding layers: payoff_matrix, round_number, opp_action, agent_action, agent_reward, opp_reward")
    print()


def test_full_game_sequence():
    """Test a full game sequence tracking all state elements."""
    print("Test 5: Full game sequence with state tracking...")
    
    game = GameFactory.create_game('hawk-dove')
    opponent = OpponentFactory.create_probabilistic_opponent(defection_probability=0.3)
    
    print(f"  Playing 5 rounds of Hawk-Dove against 0.3-defector:")
    
    for round_num in range(5):
        # Get state before action
        state = game.get_state_vector()
        
        # Choose action (just cooperate for simplicity)
        agent_action = Action.COOPERATE
        opponent_action = opponent.play_action(game.history, round_num)
        
        # Play round
        agent_reward, opponent_reward = game.play_round(agent_action, opponent_action)
        
        print(f"\n    Round {round_num + 1}:")
        print(f"      Agent: {agent_action.name}, Opponent: {opponent_action.name}")
        print(f"      Rewards: Agent={agent_reward:.2f}, Opponent={opponent_reward:.2f}")
        
        # Get state after action
        state_after = game.get_state_vector()
        print(f"      Next state preview:")
        print(f"        - Round number: {state_after[4]:.2f}")
        print(f"        - Opp prev action: {state_after[5]:.1f} ({opponent_action.name})")
        print(f"        - Agent prev action: {state_after[6]:.1f} ({agent_action.name})")
        print(f"        - Agent prev reward: {state_after[7]:.2f}")
        print(f"        - Opp prev reward: {state_after[8]:.2f}")
    
    print("\n  ✓ Full game sequence completed successfully")
    print()


def test_all_game_types():
    """Test all three game types with 9-element input."""
    print("Test 6: Testing all game types...")
    
    games = {
        'prisoners-dilemma': 'Prisoner\'s Dilemma',
        'hawk-dove': 'Hawk-Dove',
        'stag-hunt': 'Stag-Hunt'
    }
    
    for game_id, game_name in games.items():
        game = GameFactory.create_game(game_id)
        
        # First trial
        state = game.get_state_vector()
        assert len(state) == 9, f"{game_name} state length is {len(state)}"
        
        # Play a round
        game.play_round(Action.COOPERATE, Action.DEFECT)
        state = game.get_state_vector()
        assert len(state) == 9, f"{game_name} state length after action is {len(state)}"
        
        # Check that previous actions/rewards are recorded
        assert state[5] == 1.0, f"{game_name} opponent prev action incorrect"  # DEFECT
        assert state[6] == 0.0, f"{game_name} agent prev action incorrect"  # COOPERATE
        assert state[7] != 0.0 or state[8] != 0.0, f"{game_name} rewards not recorded"
        
        print(f"  ✓ {game_name}: state_size=9, history tracking works")
    
    print()


def test_config_files():
    """Test that config files have been updated."""
    print("Test 7: Verifying config files...")
    
    import json
    config_files = [
        'config/default_config.json',
        'config/quick_test_config.json',
        'config/generalization_matrix_config.json',
        'config/whole_population_config.json'
    ]
    
    for config_path in config_files:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        input_size = config['network_config']['input_size']
        assert input_size == 9, f"{config_path}: input_size is {input_size}, expected 9"
        print(f"  ✓ {config_path}: input_size = {input_size}")
    
    print()


def test_network_config_default():
    """Test that NetworkConfig has correct default."""
    print("Test 8: Verifying NetworkConfig default...")
    
    config = NetworkConfig()
    assert config.input_size == 9, f"NetworkConfig.input_size is {config.input_size}, expected 9"
    
    print(f"  ✓ NetworkConfig default input_size = {config.input_size}")
    print()


def main():
    """Run all tests."""
    print("="*80)
    print("TESTING 9-element INPUT MODIFICATION")
    print("="*80)
    print()
    
    try:
        test_state_size()
        test_first_trial_state()
        test_after_actions()
        test_network_compatibility()
        test_full_game_sequence()
        test_all_game_types()
        test_config_files()
        test_network_config_default()
        
        print("="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print()
        print("Summary:")
        print("  - State vector now includes 9 elements:")
        print("    1-4: Payoff matrix (flattened)")
        print("    5: Round number (normalized)")
        print("    6: Opponent's previous action (-1 for first trial, 0=COOPERATE, 1=DEFECT)")
        print("    7: Agent's previous action (-1 for first trial, 0=COOPERATE, 1=DEFECT)")
        print("    8: Agent's previous reward (0 for first trial)")
        print("    9: Opponent's previous reward (0 for first trial)")
        print("  - All games, network, and config files updated correctly")
        print()
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
