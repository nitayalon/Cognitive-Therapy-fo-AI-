"""
Simple test script to verify the framework installation and basic functionality.

This script performs basic tests of all major components to ensure
the framework is properly installed and working correctly.
"""

import sys
import os
import torch
import numpy as np

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from cognitive_therapy_ai.games import HawkDove, PrisonersDilemma, BattleOfSexes, StagHunt, GameFactory
        from cognitive_therapy_ai.opponent import Opponent, OpponentFactory
        from cognitive_therapy_ai.network import GameLSTM, NetworkManager
        from cognitive_therapy_ai.trainer import GameTrainer, GameSession
        from cognitive_therapy_ai.loss import CompositeLoss
        from cognitive_therapy_ai.config import NetworkConfig, TrainingConfig
        from cognitive_therapy_ai.utils import set_random_seeds, MetricsTracker
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_games():
    """Test game implementations."""
    print("\nTesting games...")
    
    try:
        from cognitive_therapy_ai.games import GameFactory
        
        # Test game creation
        games = [
            GameFactory.create_game('hawk-dove'),
            GameFactory.create_game('prisoners-dilemma'),  
            GameFactory.create_game('battle-of-sexes'),
            GameFactory.create_game('stag-hunt')
        ]
        
        for game in games:
            # Test basic game functionality
            from cognitive_therapy_ai.games import Action
            
            reward1, reward2 = game.play_round(Action.COOPERATE, Action.DEFECT)
            assert isinstance(reward1, (int, float))
            assert isinstance(reward2, (int, float))
            
            state = game.get_state_vector()
            assert isinstance(state, np.ndarray)
            assert len(state) == 20  # 5 history * 4 features
            
            game.reset()
            assert len(game.history) == 0
        
        print("✓ Game tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Game test failed: {e}")
        return False


def test_opponents():
    """Test opponent implementations."""
    print("\nTesting opponents...")
    
    try:
        from cognitive_therapy_ai.opponent import OpponentFactory
        from cognitive_therapy_ai.games import Action
        
        # Test probabilistic opponents
        opponents = OpponentFactory.create_opponent_set([0.1, 0.5, 0.9])
        assert len(opponents) == 3
        
        for opponent in opponents:
            # Test action selection
            action = opponent.play_action([], 0)
            assert action in [Action.COOPERATE, Action.DEFECT]
            
            # Test statistics
            opponent.update_payoff(1.0)
            assert opponent.get_average_payoff() == 1.0
            
            opponent.reset()
            assert opponent.games_played == 0
        
        print("✓ Opponent tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Opponent test failed: {e}")
        return False


def test_network():
    """Test LSTM network."""
    print("\nTesting network...")
    
    try:
        from cognitive_therapy_ai.network import GameLSTM, NetworkManager
        
        # Create network
        network = GameLSTM(
            input_size=5,  # Updated for simplified payoff matrix + round number representation
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            num_actions=2
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 3
        x = torch.randn(batch_size, seq_len, 20)
        
        policy_logits, opponent_logits, type_pred, hidden = network.forward(x)
        
        assert policy_logits.shape == (batch_size, 2)
        assert opponent_logits.shape == (batch_size, 2)
        assert type_pred.shape == (batch_size, 1)
        assert len(hidden) == 2  # h and c
        
        # Test action sampling
        action, log_prob = network.sample_action(x[:1])  # Single batch
        from cognitive_therapy_ai.games import Action
        assert action in [Action.COOPERATE, Action.DEFECT]
        assert isinstance(log_prob, torch.Tensor)
        
        # Test network manager
        manager = NetworkManager(network)
        summary = manager.get_network_summary()
        assert "GameLSTM Network Summary" in summary
        
        print("✓ Network tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Network test failed: {e}")
        return False


def test_loss():
    """Test loss functions."""
    print("\nTesting loss functions...")
    
    try:
        from cognitive_therapy_ai.loss import CompositeLoss
        
        loss_fn = CompositeLoss()
        
        # Create dummy data
        batch_size = 4
        policy_logits = torch.randn(batch_size, 2)
        opponent_action_logits = torch.randn(batch_size, 2)
        opponent_type_preds = torch.rand(batch_size, 1)
        actions_taken = torch.randint(0, 2, (batch_size,))
        rewards = torch.randn(batch_size)
        opponent_actions = torch.randint(0, 2, (batch_size,))
        opponent_type_true = torch.rand(batch_size, 1)
        
        # Test loss calculation
        loss_dict = loss_fn(
            policy_logits,
            opponent_action_logits,
            opponent_type_preds,
            actions_taken,
            rewards,
            opponent_actions,
            opponent_type_true
        )
        
        assert 'total_loss' in loss_dict
        assert 'policy_loss' in loss_dict
        assert 'action_prediction_loss' in loss_dict
        assert 'type_prediction_loss' in loss_dict
        
        # Test that loss is a tensor
        assert isinstance(loss_dict['total_loss'], torch.Tensor)
        assert loss_dict['total_loss'].requires_grad
        
        print("✓ Loss function tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        return False


def test_integration():
    """Test basic integration between components."""
    print("\nTesting component integration...")
    
    try:
        from cognitive_therapy_ai.games import GameFactory
        from cognitive_therapy_ai.opponent import OpponentFactory
        from cognitive_therapy_ai.network import GameLSTM
        from cognitive_therapy_ai.trainer import GameSession
        
        # Create components
        game = GameFactory.create_game('prisoners-dilemma')
        opponents = OpponentFactory.create_opponent_set([0.5])
        network = GameLSTM(input_size=game.get_state_size(), hidden_size=16, num_layers=1)
        
        # Test game session
        session = GameSession(
            game=game,
            opponent=opponents[0],
            network=network,
            num_games=5
        )
        
        device = torch.device('cpu')
        results = session.play_session(device)
        
        assert 'training_data' in results
        assert 'session_stats' in results
        assert results['session_stats']['num_games'] == 5
        
        print("✓ Integration tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("COGNITIVE THERAPY AI FRAMEWORK - TEST SUITE")
    print("="*60)
    
    # Set random seed for reproducible tests
    torch.manual_seed(42)
    np.random.seed(42)
    
    tests = [
        test_imports,
        test_games,
        test_opponents,
        test_network,
        test_loss,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! Framework is ready to use.")
        sys.exit(0)
    else:
        print("Some tests failed. Please check the installation.")
        sys.exit(1)


if __name__ == "__main__":
    main()