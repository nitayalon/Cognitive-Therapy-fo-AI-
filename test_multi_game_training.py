#!/usr/bin/env python3
"""
Test script for multi-game simultaneous training functionality.
"""

import sys
sys.path.append('src')

import torch
import numpy as np
from cognitive_therapy_ai.trainer import GameTrainer
from cognitive_therapy_ai.network import GameLSTM
from cognitive_therapy_ai.opponent import OpponentFactory
from cognitive_therapy_ai.config import TrainingConfig, NetworkConfig

def test_multi_game_training():
    """Test simultaneous multi-game training."""
    print("Testing Multi-Game Simultaneous Training")
    print("=" * 50)
    
    # Create network configuration
    network_config = NetworkConfig(
        input_size=5,  # Standard state size for mixed-motive games
        hidden_size=64,
        num_layers=2,
        output_size=2,  # Cooperate/Defect
        dropout_rate=0.1
    )
    
    # Create training configuration
    training_config = TrainingConfig(
        max_epochs=5,  # Small for testing
        num_games_per_partner=10,
        learning_rate=0.001,
        patience=50,
        convergence_threshold=1e-6
    )
    
    print(f"Network config: {network_config.hidden_size} hidden units, {network_config.num_layers} layers")
    print(f"Training config: {training_config.max_epochs} epochs, {training_config.num_games_per_partner} games per partner")
    
    # Create network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = GameLSTM(network_config)
    
    print(f"Using device: {device}")
    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
    
    # Create trainer
    trainer = GameTrainer(
        network=network,
        training_config=training_config,
        device=device,
        use_adaptive_loss=True
    )
    
    # Create opponents
    opponent_probs = [0.3, 0.7]
    opponents = []
    
    for prob in opponent_probs:
        opponent = OpponentFactory.create_opponent('probabilistic', cooperation_probability=prob)
        opponents.append(opponent)
    
    print(f"Created {len(opponents)} opponents with cooperation probabilities: {opponent_probs}")
    
    # Test 1: Single game training (for comparison)
    print(f"\n1. Testing single game training (Prisoner's Dilemma):")
    single_results = trainer.train_on_game(
        game_name='prisoners-dilemma',
        opponents=opponents
    )
    
    print(f"  Single game training completed:")
    print(f"  - Epochs: {single_results['final_metrics']['total_epochs']}")
    print(f"  - Best loss: {single_results['final_metrics']['best_loss']:.6f}")
    
    # Reset trainer state for multi-game test
    trainer.current_epoch = 0
    trainer.best_loss = float('inf')
    trainer.patience_counter = 0
    
    # Test 2: Multi-game training
    print(f"\n2. Testing multi-game simultaneous training:")
    
    # Create game configurations
    game_configs = [
        GameTrainer.create_game_config('prisoners-dilemma', weight=1.0),
        GameTrainer.create_game_config('hawk-dove', weight=1.0, resource_value=6.0, cost_of_conflict=10.0),
        GameTrainer.create_game_config('battle-of-sexes', weight=0.5)  # Lower weight
    ]
    
    print(f"  Game configurations:")
    for config in game_configs:
        print(f"    - {config['name']}: weight={config['weight']}")
    
    # Run multi-game training
    multi_results = trainer.train_on_multiple_games(
        game_configs=game_configs,
        opponents=opponents
    )
    
    print(f"\n  Multi-game training completed:")
    print(f"  - Games: {', '.join(multi_results['games'])}")
    print(f"  - Epochs: {multi_results['final_metrics']['total_epochs']}")
    print(f"  - Best loss: {multi_results['final_metrics']['best_loss']:.6f}")
    
    # Show per-game breakdown
    if multi_results['epoch_results']:
        last_epoch = multi_results['epoch_results'][-1]
        if 'per_game_losses' in last_epoch:
            print(f"  - Final per-game losses:")
            for game, losses in last_epoch['per_game_losses'].items():
                print(f"    * {game}: {losses.get('total_loss', 'N/A'):.6f}")
        
        print(f"  - Total sessions per game: {last_epoch['num_sessions_per_game']}")
    
    # Test 3: Multi-game evaluation
    print(f"\n3. Testing multi-game evaluation:")
    
    from cognitive_therapy_ai.games import GameFactory
    
    eval_games = {
        'prisoners-dilemma': GameFactory.create_game('prisoners-dilemma'),
        'hawk-dove': GameFactory.create_game('hawk-dove')
    }
    
    eval_results = trainer.evaluate_on_multiple_games(
        games=eval_games,
        opponents=opponents[:1],  # Just test with one opponent
        num_sessions=3
    )
    
    print(f"  Evaluation results:")
    for game_name, game_results in eval_results.items():
        print(f"    {game_name}:")
        for opponent_name, metrics in game_results.items():
            avg_reward = metrics.get('average_reward', 0)
            cooperation_rate = metrics.get('cooperation_rate', 0)
            print(f"      vs {opponent_name}: reward={avg_reward:.3f}, coop_rate={cooperation_rate:.3f}")
    
    # Test 4: Utility methods
    print(f"\n4. Testing utility methods:")
    
    # Test balanced configs
    balanced_configs = GameTrainer.create_balanced_game_configs([
        'prisoners-dilemma', 'hawk-dove', 'stag-hunt'
    ])
    
    print(f"  Balanced game configs:")
    for config in balanced_configs:
        print(f"    - {config['name']}: weight={config['weight']}")
    
    # Test summary generation
    summary = trainer.get_multi_game_training_summary(multi_results)
    print(f"\n  Training summary:")
    print(f"    {summary}")
    
    print(f"\n✅ All multi-game training tests completed successfully!")

if __name__ == "__main__":
    test_multi_game_training()