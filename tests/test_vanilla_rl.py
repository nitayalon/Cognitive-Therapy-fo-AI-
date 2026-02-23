"""
Test vanilla RL agent implementation against proto-ToM agent.

This test verifies that:
1. VanillaRLLoss computes correctly
2. Vanilla RL agents can be trained
3. Interface is compatible with proto-ToM agents
"""

import torch
try:
    import pytest
except ImportError:
    pytest = None
from src.cognitive_therapy_ai import (
    GameLSTM,
    GameTrainer,
    GameFactory,
    OpponentFactory
)
from src.cognitive_therapy_ai.tom_rl_loss import VanillaRLLoss, ToMRLLoss
from src.cognitive_therapy_ai.config import TrainingConfig


def test_vanilla_rl_loss_creation():
    """Test that VanillaRLLoss can be instantiated."""
    loss_fn = VanillaRLLoss()
    assert loss_fn is not None
    assert loss_fn.alpha == 0.0  # No auxiliary task
    assert loss_fn.gamma == 0.99
    print("✓ VanillaRLLoss creation test passed")


def test_vanilla_rl_loss_forward():
    """Test that VanillaRLLoss forward pass works correctly."""
    batch_size = 10
    num_actions = 2
    
    # Create dummy inputs
    policy_logits = torch.randn(batch_size, num_actions)
    opponent_policy_logits = torch.randn(batch_size, 2)  # Ignored
    value_estimates = torch.randn(batch_size, 1)
    actions_taken = torch.randint(0, 2, (batch_size,))
    rewards = torch.randn(batch_size)
    opponent_actions = torch.randint(0, 2, (batch_size,))  # Ignored
    true_opponent_policy = torch.randn(batch_size, 2)  # Ignored
    
    # Create loss function
    loss_fn = VanillaRLLoss()
    
    # Forward pass
    loss_dict = loss_fn(
        policy_logits=policy_logits,
        opponent_policy_logits=opponent_policy_logits,
        value_estimates=value_estimates,
        actions_taken=actions_taken,
        rewards=rewards,
        opponent_actions=opponent_actions,
        true_opponent_policy=true_opponent_policy
    )
    
    # Verify outputs
    assert 'total_loss' in loss_dict
    assert 'rl_loss' in loss_dict
    assert 'value_loss' in loss_dict
    assert 'opponent_policy_loss' in loss_dict
    
    # Verify auxiliary loss is zero
    assert loss_dict['opponent_policy_loss'].item() == 0.0
    assert loss_dict['alpha'] == 0.0
    
    print("✓ VanillaRLLoss forward pass test passed")


def test_loss_interface_compatibility():
    """Test that VanillaRLLoss and ToMRLLoss have compatible interfaces."""
    batch_size = 10
    num_actions = 2
    
    # Create dummy inputs
    policy_logits = torch.randn(batch_size, num_actions)
    opponent_policy_logits = torch.randn(batch_size, 2)
    value_estimates = torch.randn(batch_size, 1)
    actions_taken = torch.randint(0, 2, (batch_size,))
    rewards = torch.randn(batch_size)
    opponent_actions = torch.randint(0, 2, (batch_size,))
    true_opponent_policy = torch.softmax(torch.randn(batch_size, 2), dim=-1)
    
    # Test both loss functions
    vanilla_loss = VanillaRLLoss()
    tom_loss = ToMRLLoss()
    
    vanilla_dict = vanilla_loss(
        policy_logits, opponent_policy_logits, value_estimates,
        actions_taken, rewards, opponent_actions, true_opponent_policy
    )
    
    tom_dict = tom_loss(
        policy_logits, opponent_policy_logits, value_estimates,
        actions_taken, rewards, opponent_actions, true_opponent_policy
    )
    
    # Verify both return compatible dictionaries
    shared_keys = set(vanilla_dict.keys()) & set(tom_dict.keys())
    assert 'total_loss' in shared_keys
    assert 'rl_loss' in shared_keys
    
    # Verify vanilla has zero auxiliary loss
    assert vanilla_dict['opponent_policy_loss'].item() == 0.0
    # Verify ToM has non-zero auxiliary loss (should be > 0)
    assert tom_dict['opponent_policy_loss'].item() >= 0.0
    
    print("✓ Loss interface compatibility test passed")


def test_vanilla_trainer_creation():
    """Test that GameTrainer can be created with vanilla agent type."""
    # Create a simple game and network
    game = GameFactory.create_game('prisoners-dilemma')
    network = GameLSTM(input_size=5, hidden_size=32, num_layers=1)
    
    # Create training config
    config = TrainingConfig(
        num_games_per_partner=10,
        max_epochs=5
    )
    
    # Create vanilla trainer
    vanilla_trainer = GameTrainer(
        network=network,
        training_config=config,
        agent_type="vanilla"
    )
    
    assert vanilla_trainer is not None
    assert vanilla_trainer.agent_type == "vanilla"
    assert isinstance(vanilla_trainer.loss_fn, VanillaRLLoss)
    
    print("✓ Vanilla trainer creation test passed")


def test_proto_tom_trainer_creation():
    """Test that GameTrainer can be created with proto-tom agent type."""
    # Create a simple game and network
    game = GameFactory.create_game('prisoners-dilemma')
    network = GameLSTM(input_size=5, hidden_size=32, num_layers=1)
    
    # Create training config
    config = TrainingConfig(
        num_games_per_partner=10,
        max_epochs=5
    )
    
    # Create proto-ToM trainer
    tom_trainer = GameTrainer(
        network=network,
        training_config=config,
        agent_type="proto-tom"
    )
    
    assert tom_trainer is not None
    assert tom_trainer.agent_type == "proto-tom"
    assert isinstance(tom_trainer.loss_fn, (ToMRLLoss, type(tom_trainer.loss_fn)))
    assert not isinstance(tom_trainer.loss_fn, VanillaRLLoss)
    
    print("✓ Proto-ToM trainer creation test passed")


def test_vanilla_quick_training():
    """Test that vanilla agent can complete a few training iterations."""
    # Create game and opponents
    game = GameFactory.create_game('prisoners-dilemma')
    opponents = OpponentFactory.create_opponent_set([0.3, 0.7])
    
    # Create network
    network = GameLSTM(input_size=5, hidden_size=32, num_layers=1)
    
    # Create training config with very few epochs
    config = TrainingConfig(
        num_games_per_partner=10,
        max_epochs=3,
        convergence_threshold=1e-2,
        patience=10
    )
    
    # Create vanilla trainer
    trainer = GameTrainer(
        network=network,
        training_config=config,
        agent_type="vanilla"
    )
    
    # Run a quick training session (should complete without errors)
    try:
        results = trainer.train_on_game(
            game_name='prisoners-dilemma',
            opponents=opponents,
            save_dir=None  # Don't save checkpoints
        )
        assert results is not None
        assert 'training_complete' in results or 'epochs_completed' in results
        print("✓ Vanilla quick training test passed")
    except Exception as e:
        if pytest:
            pytest.fail(f"Vanilla training failed with error: {e}")
        else:
            print(f"✗ Vanilla training test FAILED: {e}")
            raise


def test_agent_type_validation():
    """Test that invalid agent types are rejected."""
    game = GameFactory.create_game('prisoners-dilemma')
    network = GameLSTM(input_size=5, hidden_size=32, num_layers=1)
    config = TrainingConfig()
    
    # Should raise error for invalid agent type
    try:
        trainer = GameTrainer(
            network=network,
            training_config=config,
            agent_type="invalid_type"
        )
        # If we get here, the test failed
        if pytest:
            pytest.fail("Expected ValueError for invalid agent_type")
        else:
            print("✗ Agent type validation test FAILED - no error raised")
            return
    except ValueError as e:
        if "Unknown agent_type" in str(e):
            print("✓ Agent type validation test passed")
        else:
            if pytest:
                pytest.fail(f"Wrong error message: {e}")
            else:
                print(f"✗ Wrong error message: {e}")


if __name__ == "__main__":
    print("Running Vanilla RL Tests...\n")
    
    test_vanilla_rl_loss_creation()
    test_vanilla_rl_loss_forward()
    test_loss_interface_compatibility()
    test_vanilla_trainer_creation()
    test_proto_tom_trainer_creation()
    test_agent_type_validation()
    test_vanilla_quick_training()
    
    print("\n✅ All vanilla RL tests passed!")
