"""
Demo script showing the new detailed training monitoring capabilities.

This script demonstrates how to use the enhanced training monitor that logs:
- Loss components for each iteration
- Network head outputs (policy, opponent policy, value)
- Actions and rewards for both parties
- Real-time table updates

Usage:
    python detailed_training_demo.py
"""

import os
from cognitive_therapy_ai import (
    GameFactory, 
    OpponentFactory, 
    GameLSTM, 
    GameTrainer
)
from cognitive_therapy_ai.config import NetworkConfig, TrainingConfig


def run_detailed_training_demo():
    """Run a small training demo with detailed logging."""
    print("🎯 Detailed Training Monitor Demo")
    print("=" * 50)
    
    # Create a simple game and opponents
    game = GameFactory.create_game('prisoners-dilemma')
    opponents = OpponentFactory.create_opponent_set([0.3, 0.7])  # Just 2 opponents for demo
    
    # Create network with small configuration for fast demo
    network_config = NetworkConfig(
        hidden_size=64,  # Smaller for demo
        num_layers=2,    # Fewer layers
        dropout=0.1
    )
    
    network = GameLSTM(
        input_size=game.get_state_size(),
        hidden_size=network_config.hidden_size,
        num_layers=network_config.num_layers,
        dropout=network_config.dropout,
        num_actions=2
    )
    
    # Create training configuration for short demo
    training_config = TrainingConfig(
        num_games_per_partner=20,  # Short sessions
        max_epochs=10,             # Few epochs for demo
        learning_rate=0.001
    )
    
    # Create trainer
    trainer = GameTrainer(
        network=network,
        training_config=training_config,
        use_adaptive_loss=False
    )
    
    # Run training with detailed logging
    print(f"🚀 Starting training with detailed monitoring...")
    print(f"📊 Watch for CSV/Excel files in output directory")
    
    results = trainer.train_on_game(
        game_name='prisoners-dilemma',
        opponents=opponents,
        save_dir='demo_detailed_training'
    )
    
    print(f"\n✅ Training completed!")
    print(f"📁 Detailed logs saved to: demo_detailed_training/detailed_training_logs/")
    print(f"📈 Check the CSV and Excel files for complete training data")
    
    # Print summary
    final_metrics = results.get('final_metrics', {})
    print(f"\n📊 Final Results:")
    print(f"   Epochs: {len(results.get('epoch_results', []))}")
    print(f"   Best Loss: {final_metrics.get('best_loss', 'N/A')}")
    print(f"   Converged: {final_metrics.get('converged', False)}")
    
    return results


def run_multi_game_detailed_demo():
    """Run a multi-game training demo with detailed logging."""
    print("\n🎯 Multi-Game Detailed Training Monitor Demo")
    print("=" * 50)
    
    # Create multiple games
    game_configs = [
        {'name': 'prisoners-dilemma', 'weight': 1.0},
        {'name': 'hawk-dove', 'weight': 1.0}
    ]
    
    opponents = OpponentFactory.create_opponent_set([0.2, 0.8])  # Just 2 opponents
    
    # Create network
    sample_game = GameFactory.create_game('prisoners-dilemma')
    network = GameLSTM(
        input_size=sample_game.get_state_size(),
        hidden_size=64,  # Small for demo
        num_layers=2,
        dropout=0.1,
        num_actions=2
    )
    
    # Create training config
    training_config = TrainingConfig(
        num_games_per_partner=15,  # Short sessions
        max_epochs=8,              # Few epochs
        learning_rate=0.001
    )
    
    # Create trainer
    trainer = GameTrainer(
        network=network,
        training_config=training_config,
        use_adaptive_loss=False
    )
    
    print(f"🚀 Starting multi-game training with detailed monitoring...")
    print(f"📊 Training on: {[config['name'] for config in game_configs]}")
    
    results = trainer.train_on_multiple_games(
        game_configs=game_configs,
        opponents=opponents,
        save_dir='demo_multi_game_detailed'
    )
    
    print(f"\n✅ Multi-game training completed!")
    print(f"📁 Detailed logs saved to: demo_multi_game_detailed/detailed_training_logs/")
    print(f"📈 Multi-game data shows training across different strategic contexts")
    
    return results


if __name__ == "__main__":
    # Run single game demo
    single_results = run_detailed_training_demo()
    
    # Run multi-game demo  
    multi_results = run_multi_game_detailed_demo()
    
    print(f"\n🎉 Demo completed!")
    print(f"💡 Check the generated CSV/Excel files to see:")
    print(f"   - Loss components for every training step")
    print(f"   - Network outputs (policy, opponent policy, value)")
    print(f"   - Actions chosen by agent and opponent")
    print(f"   - Rewards received by both parties")
    print(f"   - Gradient norms and other training metrics")
    print(f"\n📚 Files are human-readable and Excel-compatible for analysis!")