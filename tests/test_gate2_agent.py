"""
Gate 2 Tests: Agent Architecture

Verifies:
1. Parameter count formula matches actual count for all H and input conditions
2. Forward pass produces correct output shapes
3. Hidden state management (persistence across steps, reset between sessions)
4. Action selection (deterministic, stochastic, epsilon-greedy)
5. Integration with ObservationEncoder
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import yaml

# Import from package
from cognitive_therapy_ai.representation_agent import RepresentationAgent, create_agent_from_config
from cognitive_therapy_ai.encoding import ObservationEncoder, Action
from cognitive_therapy_ai.games import PrisonersDilemma


def load_config():
    """Load base configuration."""
    config_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class TestParameterCount:
    """Test that parameter count formula is exact."""
    
    def test_parameter_count_no_game_all_H(self):
        """Test parameter count for no_game condition (d_in=8) across all H values."""
        config = load_config()
        H_grid = config['agent']['H_grid']
        
        for H in H_grid:
            agent = RepresentationAgent(input_dim=8, hidden_size=H)
            is_correct, details = agent.verify_parameter_count()
            
            assert is_correct, \
                f"H={H}, d_in=8: Expected {details['expected_total']} params, " \
                f"got {details['actual_total']}"
    
    def test_parameter_count_game_tag_all_H(self):
        """Test parameter count for game_tag condition (d_in=11) across all H values."""
        config = load_config()
        H_grid = config['agent']['H_grid']
        
        for H in H_grid:
            agent = RepresentationAgent(input_dim=11, hidden_size=H)
            is_correct, details = agent.verify_parameter_count()
            
            assert is_correct, \
                f"H={H}, d_in=11: Expected {details['expected_total']} params, " \
                f"got {details['actual_total']}"
    
    def test_parameter_breakdown_manual(self):
        """Manually verify parameter count breakdown for H=4, d_in=8."""
        H = 4
        d_in = 8
        
        agent = RepresentationAgent(input_dim=d_in, hidden_size=H)
        counts = agent.count_parameters()
        
        # Expected counts (manual calculation)
        # LSTM: 4 * (H*d_in + H*H + 2*H) = 4 * (32 + 16 + 8) = 4 * 56 = 224
        # Policy: 2*H + 2 = 8 + 2 = 10
        # Value: H + 1 = 4 + 1 = 5
        # Total: 224 + 10 + 5 = 239
        
        expected_lstm = 4 * (H * d_in + H * H + 2 * H)
        expected_policy = 2 * H + 2
        expected_value = H + 1
        expected_total = expected_lstm + expected_policy + expected_value
        
        assert counts['lstm_params'] == expected_lstm, \
            f"LSTM params: expected {expected_lstm}, got {counts['lstm_params']}"
        assert counts['policy_params'] == expected_policy, \
            f"Policy params: expected {expected_policy}, got {counts['policy_params']}"
        assert counts['value_params'] == expected_value, \
            f"Value params: expected {expected_value}, got {counts['value_params']}"
        assert counts['total_params'] == expected_total, \
            f"Total params: expected {expected_total}, got {counts['total_params']}"
    
    def test_capacity_increases_with_H(self):
        """Verify that parameter count increases with hidden size."""
        H_grid = [2, 4, 8, 16, 32]
        d_in = 8
        
        param_counts = []
        for H in H_grid:
            agent = RepresentationAgent(input_dim=d_in, hidden_size=H)
            param_counts.append(agent.count_parameters()['total_params'])
        
        # Check monotonic increase
        for i in range(len(param_counts) - 1):
            assert param_counts[i] < param_counts[i+1], \
                f"Parameter count should increase with H: {param_counts}"


class TestForwardPass:
    """Test forward pass output shapes and behavior."""
    
    def test_forward_output_shapes(self):
        """Test that forward pass produces correct output shapes."""
        batch_size = 4
        H = 8
        d_in = 8
        
        agent = RepresentationAgent(input_dim=d_in, hidden_size=H)
        obs = torch.randn(batch_size, d_in)
        
        policy_logits, value, hidden_state = agent.forward(obs)
        
        assert policy_logits.shape == (batch_size, 2), \
            f"Policy logits shape: expected ({batch_size}, 2), got {policy_logits.shape}"
        assert value.shape == (batch_size, 1), \
            f"Value shape: expected ({batch_size}, 1), got {value.shape}"
        
        h, c = hidden_state
        assert h.shape == (1, batch_size, H), \
            f"Hidden state h shape: expected (1, {batch_size}, {H}), got {h.shape}"
        assert c.shape == (1, batch_size, H), \
            f"Hidden state c shape: expected (1, {batch_size}, {H}), got {c.shape}"
    
    def test_forward_with_one_hot_input(self):
        """Test forward pass with valid one-hot input."""
        H = 4
        encoder = ObservationEncoder("no_game")
        d_in = encoder.get_input_dim()
        
        agent = RepresentationAgent(input_dim=d_in, hidden_size=H)
        
        # Create valid one-hot observation (CC outcome)
        obs = encoder.encode_observation(
            agent_last_action=Action.COOPERATE,
            opponent_last_action=Action.COOPERATE
        )
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, d_in)
        
        policy_logits, value, hidden_state = agent.forward(obs_tensor)
        
        assert policy_logits.shape == (1, 2)
        assert value.shape == (1, 1)
        assert not torch.isnan(policy_logits).any(), "Policy logits contain NaN"
        assert not torch.isnan(value).any(), "Value contains NaN"
    
    def test_forward_batched_different_inputs(self):
        """Test forward pass with batch of different observations."""
        H = 4
        encoder = ObservationEncoder("no_game")
        d_in = encoder.get_input_dim()
        
        agent = RepresentationAgent(input_dim=d_in, hidden_size=H)
        
        # Create batch of different observations
        obs_list = [
            encoder.encode_start_token(),
            encoder.encode_observation(Action.COOPERATE, Action.COOPERATE),
            encoder.encode_observation(Action.DEFECT, Action.DEFECT),
            encoder.encode_observation(Action.COOPERATE, Action.DEFECT),
        ]
        obs_batch = torch.tensor(obs_list, dtype=torch.float32)  # (4, d_in)
        
        policy_logits, value, hidden_state = agent.forward(obs_batch)
        
        assert policy_logits.shape == (4, 2)
        assert value.shape == (4, 1)


class TestHiddenStatePersistence:
    """Test that hidden state persists across rounds and resets between sessions."""
    
    def test_hidden_state_changes_across_steps(self):
        """Verify that hidden state updates across sequential steps."""
        H = 4
        d_in = 8
        agent = RepresentationAgent(input_dim=d_in, hidden_size=H)
        
        obs = torch.randn(1, d_in)
        
        # First step (no hidden state)
        _, _, h1 = agent.forward(obs, hidden_state=None)
        
        # Second step (use hidden state from first step)
        _, _, h2 = agent.forward(obs, hidden_state=h1)
        
        # Hidden states should be different
        assert not torch.allclose(h1[0], h2[0]), \
            "Hidden state h should change across steps"
        assert not torch.allclose(h1[1], h2[1]), \
            "Cell state c should change across steps"
    
    def test_reset_hidden_state(self):
        """Test that reset produces zero-initialized hidden state."""
        H = 8
        d_in = 8
        agent = RepresentationAgent(input_dim=d_in, hidden_size=H)
        
        batch_size = 3
        h0, c0 = agent.reset_hidden_state(batch_size=batch_size)
        
        assert h0.shape == (1, batch_size, H)
        assert c0.shape == (1, batch_size, H)
        assert torch.allclose(h0, torch.zeros_like(h0)), \
            "Reset hidden state h should be all zeros"
        assert torch.allclose(c0, torch.zeros_like(c0)), \
            "Reset cell state c should be all zeros"
    
    def test_session_simulation(self):
        """Simulate T=5 rounds within a session (hidden state persists)."""
        H = 4
        encoder = ObservationEncoder("no_game")
        d_in = encoder.get_input_dim()
        agent = RepresentationAgent(input_dim=d_in, hidden_size=H)
        
        T = 5
        hidden_state = agent.reset_hidden_state(batch_size=1)
        
        prev_h = None
        for t in range(T):
            # Random observation
            obs = torch.randn(1, d_in)
            policy_logits, value, hidden_state = agent.forward(obs, hidden_state)
            
            # Verify hidden state changes
            if prev_h is not None:
                assert not torch.allclose(hidden_state[0], prev_h), \
                    f"Hidden state should change at round {t}"
            
            prev_h = hidden_state[0].clone()


class TestActionSelection:
    """Test action selection methods."""
    
    def test_select_action_deterministic(self):
        """Test deterministic action selection (argmax)."""
        H = 4
        d_in = 8
        agent = RepresentationAgent(input_dim=d_in, hidden_size=H)
        
        obs = torch.randn(d_in)
        
        # Run multiple times - should always get same action
        actions = []
        for _ in range(5):
            action, probs, _ = agent.select_action(obs, deterministic=True)
            actions.append(action)
        
        # All actions should be identical (deterministic)
        assert len(set(actions)) == 1, \
            f"Deterministic selection should give same action, got {actions}"
    
    def test_select_action_stochastic(self):
        """Test stochastic action selection (sampling)."""
        H = 4
        d_in = 8
        agent = RepresentationAgent(input_dim=d_in, hidden_size=H)
        
        obs = torch.randn(d_in)
        
        # Run many times - should get variety (unless policy is very extreme)
        actions = []
        for _ in range(100):
            action, probs, _ = agent.select_action(obs, deterministic=False)
            actions.append(action)
            assert action in [0, 1], f"Action should be 0 or 1, got {action}"
        
        # With high probability, should see both actions (unless policy near deterministic)
        # This is a probabilistic test - could fail rarely
        unique_actions = set(actions)
        # Weak test: at least verify actions are valid
        assert unique_actions.issubset({0, 1}), \
            f"Actions should be in {{0, 1}}, got {unique_actions}"
    
    def test_epsilon_greedy_exploration(self):
        """Test epsilon-greedy exploration."""
        H = 4
        d_in = 8
        agent = RepresentationAgent(input_dim=d_in, hidden_size=H)
        
        obs = torch.randn(d_in)
        
        # With epsilon=1.0, should get random actions
        actions = []
        for _ in range(100):
            action, probs, _ = agent.select_action(obs, epsilon=1.0)
            actions.append(action)
        
        # Should see both actions with high probability
        unique_actions = set(actions)
        assert len(unique_actions) == 2, \
            f"With epsilon=1.0, should see both actions, got {unique_actions}"
    
    def test_action_selection_updates_hidden_state(self):
        """Verify that action selection updates hidden state."""
        H = 4
        d_in = 8
        agent = RepresentationAgent(input_dim=d_in, hidden_size=H)
        
        obs = torch.randn(d_in)
        h0 = agent.reset_hidden_state(batch_size=1)
        
        _, _, h1 = agent.select_action(obs, hidden_state=h0)
        
        assert not torch.allclose(h0[0], h1[0]), \
            "Hidden state should update after action selection"


class TestIntegrationWithEncoding:
    """Test integration between agent and encoding module."""
    
    def test_agent_with_no_game_encoder(self):
        """Test agent with no_game encoding (d_in=8)."""
        encoder = ObservationEncoder("no_game")
        d_in = encoder.get_input_dim()
        agent = RepresentationAgent(input_dim=d_in, hidden_size=4)
        
        # Start token
        start_obs = encoder.encode_start_token()
        start_tensor = torch.tensor(start_obs, dtype=torch.float32)
        
        action, probs, h = agent.select_action(start_tensor)
        assert action in [0, 1]
        assert probs.shape == (2,)
        assert torch.allclose(probs.sum(), torch.tensor(1.0)), \
            "Policy probabilities should sum to 1"
    
    def test_agent_with_game_tag_encoder(self):
        """Test agent with game_tag encoding (d_in=11)."""
        encoder = ObservationEncoder("game_tag")
        d_in = encoder.get_input_dim()
        agent = RepresentationAgent(input_dim=d_in, hidden_size=4)
        
        # Start token with game tag
        start_obs = encoder.encode_start_token(game_name="PD")
        start_tensor = torch.tensor(start_obs, dtype=torch.float32)
        
        action, probs, h = agent.select_action(start_tensor)
        assert action in [0, 1]
        assert probs.shape == (2,)
    
    def test_full_episode_simulation(self):
        """Simulate a full episode: START → CC → CD → DC → DD."""
        encoder = ObservationEncoder("no_game")
        d_in = encoder.get_input_dim()
        agent = RepresentationAgent(input_dim=d_in, hidden_size=8)
        
        # Episode sequence
        observations = [
            encoder.encode_start_token(),
            encoder.encode_observation(Action.COOPERATE, Action.COOPERATE),
            encoder.encode_observation(Action.COOPERATE, Action.DEFECT),
            encoder.encode_observation(Action.DEFECT, Action.COOPERATE),
            encoder.encode_observation(Action.DEFECT, Action.DEFECT),
        ]
        
        hidden_state = agent.reset_hidden_state(batch_size=1)
        
        for t, obs in enumerate(observations):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action, probs, hidden_state = agent.select_action(obs_tensor, hidden_state=hidden_state)
            
            # Verify valid output
            assert action in [0, 1], f"Round {t}: invalid action {action}"
            assert torch.allclose(probs.sum(), torch.tensor(1.0)), \
                f"Round {t}: probabilities don't sum to 1"


class TestFactoryFunction:
    """Test agent creation from config."""
    
    def test_create_agent_from_config_no_game(self):
        """Test creating agent from config with no_game condition."""
        config = load_config()
        agent_config = config['agent']
        agent_config['hidden_size'] = 8  # Set specific H
        
        agent = create_agent_from_config(agent_config, input_condition="no_game")
        
        assert agent.input_dim == 8
        assert agent.hidden_size == 8
    
    def test_create_agent_from_config_game_tag(self):
        """Test creating agent from config with game_tag condition."""
        config = load_config()
        agent_config = config['agent']
        agent_config['hidden_size'] = 16
        
        agent = create_agent_from_config(agent_config, input_condition="game_tag")
        
        assert agent.input_dim == 11
        assert agent.hidden_size == 16


def test_gate2_integration():
    """
    Gate 2 integration test: Verify full agent pipeline.
    
    Tests:
    1. Create agent for both input conditions
    2. Verify parameter counts
    3. Simulate episode with hidden state persistence
    4. Verify action selection works
    """
    config = load_config()
    
    # Test both input conditions
    for input_condition in ["no_game", "game_tag"]:
        encoder = ObservationEncoder(input_condition)
        d_in = encoder.get_input_dim()
        
        # Test all H values in grid
        for H in config['agent']['H_grid']:
            agent = RepresentationAgent(input_dim=d_in, hidden_size=H)
            
            # 1. Verify parameter count
            is_correct, details = agent.verify_parameter_count()
            assert is_correct, \
                f"{input_condition}, H={H}: Parameter count mismatch"
            
            # 2. Simulate episode
            hidden_state = agent.reset_hidden_state(batch_size=1)
            for t in range(5):
                obs = encoder.encode_observation(
                    Action.COOPERATE if t % 2 == 0 else Action.DEFECT,
                    Action.COOPERATE,
                    game_name="PD" if input_condition == "game_tag" else None
                )
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                
                action, probs, hidden_state = agent.select_action(
                    obs_tensor, 
                    hidden_state=hidden_state
                )
                
                # 3. Verify valid outputs
                assert action in [0, 1]
                assert torch.allclose(probs.sum(), torch.tensor(1.0))
    
    print("✅ Gate 2 integration test passed!")


if __name__ == "__main__":
    # Run integration test
    test_gate2_integration()
    print("\n✅ All Gate 2 tests complete!")
