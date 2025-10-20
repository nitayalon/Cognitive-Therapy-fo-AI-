#!/usr/bin/env python3
"""
Test script for normalized ToM-RL loss functions.
"""

import sys
sys.path.append('src')

import torch
import numpy as np
from cognitive_therapy_ai.tom_rl_loss import (
    ToMRLLoss, 
    AdaptiveToMRLLoss, 
    create_normalized_loss,
    estimate_game_reward_scale,
    LossAnalyzer
)

def test_normalized_loss():
    """Test normalized loss functionality."""
    print("Testing Normalized ToM-RL Loss Functions")
    print("=" * 50)
    
    # Test reward scale estimation
    print("\n1. Testing reward scale estimation:")
    games = ['prisoners-dilemma', 'hawk-dove', 'battle-of-sexes', 'stag-hunt']
    for game in games:
        scale = estimate_game_reward_scale(game)
        print(f"  {game}: {scale}")
    
    # Create sample data
    batch_size = 32
    num_actions = 2
    
    # Sample tensors
    policy_logits = torch.randn(batch_size, num_actions)
    opponent_coop_probs = torch.sigmoid(torch.randn(batch_size, 1))
    value_estimates = torch.randn(batch_size, 1)
    actions_taken = torch.randint(0, num_actions, (batch_size,))
    rewards = torch.randn(batch_size) * 3.0  # Scale rewards to ~3
    opponent_actions = torch.randint(0, 2, (batch_size,))
    
    print(f"\n2. Sample data shapes:")
    print(f"  Policy logits: {policy_logits.shape}")
    print(f"  Opponent coop probs: {opponent_coop_probs.shape}")
    print(f"  Rewards range: [{rewards.min():.2f}, {rewards.max():.2f}]")
    
    # Test standard loss
    print(f"\n3. Testing standard ToM-RL loss:")
    standard_loss = ToMRLLoss(alpha=1.0, normalize_losses=False)
    standard_result = standard_loss(
        policy_logits, opponent_coop_probs, value_estimates,
        actions_taken, rewards, opponent_actions
    )
    
    print(f"  Standard RL loss: {standard_result['rl_loss'].item():.4f}")
    print(f"  Standard Op loss: {standard_result['opponent_prediction_loss'].item():.4f}")
    print(f"  Standard Total: {standard_result['total_loss'].item():.4f}")
    
    # Test normalized loss
    print(f"\n4. Testing normalized ToM-RL loss:")
    normalized_loss = ToMRLLoss(alpha=1.0, normalize_losses=True, reward_scale=3.0)
    
    # Run a few iterations to build up statistics
    for i in range(5):
        norm_result = normalized_loss(
            policy_logits, opponent_coop_probs, value_estimates,
            actions_taken, rewards, opponent_actions
        )
    
    print(f"  Normalized RL loss: {norm_result['rl_loss_normalized'].item():.4f}")
    print(f"  Normalized Op loss: {norm_result['opponent_prediction_loss_normalized'].item():.4f}")
    print(f"  Normalized Total: {norm_result['total_loss'].item():.4f}")
    print(f"  Normalization stats: {norm_result['normalization_stats']}")
    
    # Test factory function
    print(f"\n5. Testing factory function:")
    factory_loss = create_normalized_loss('prisoners-dilemma', alpha=0.5, adaptive=True)
    print(f"  Created: {type(factory_loss).__name__}")
    print(f"  Alpha: {factory_loss.alpha}")
    print(f"  Normalize losses: {factory_loss.normalize_losses}")
    print(f"  Reward scale: {factory_loss.reward_scale}")
    
    # Test loss analyzer
    print(f"\n6. Testing loss analyzer:")
    analyzer = LossAnalyzer()
    
    # Record some losses
    for epoch in range(10):
        result = normalized_loss(
            policy_logits, opponent_coop_probs, value_estimates,
            actions_taken, rewards, opponent_actions
        )
        analyzer.record_loss(result, epoch)
    
    balance_ratio = analyzer.get_loss_balance_ratio(window_size=5)
    tom_contribution = analyzer.get_tom_contribution(window_size=5)
    
    print(f"  Loss balance ratio (last 5): {balance_ratio:.3f}")
    print(f"  ToM contribution: {tom_contribution:.3f}")
    print(f"  Recorded epochs: {len(analyzer.loss_history['epoch'])}")
    
    print(f"\n✅ All tests completed successfully!")

if __name__ == "__main__":
    test_normalized_loss()