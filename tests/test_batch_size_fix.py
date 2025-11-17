"""
Test script to verify the batch size alignment fix in CompositeLoss.
"""

import torch
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cognitive_therapy_ai.loss import CompositeLoss

def test_batch_size_mismatch():
    """Test that CompositeLoss handles batch size mismatches correctly."""
    
    print("Testing batch size mismatch handling...")
    
    # Create loss function
    loss_fn = CompositeLoss()
    
    # Create tensors with mismatched batch sizes (simulating multi-game training)
    batch_size_1 = 2  # smaller batch
    batch_size_2 = 40  # larger batch
    num_actions = 2
    
    # Tensors with larger batch size
    policy_logits = torch.randn(batch_size_2, num_actions)
    opponent_action_logits = torch.randn(batch_size_2, num_actions)
    opponent_type_preds = torch.randn(batch_size_2, 1)
    
    # Tensors with smaller batch size (this was causing the original error)
    actions_taken = torch.randint(0, num_actions, (batch_size_1,))
    rewards = torch.randn(batch_size_1)
    opponent_actions = torch.randint(0, num_actions, (batch_size_1,))
    opponent_type_true = torch.randn(batch_size_1, 1)
    
    print(f"Input shapes:")
    print(f"  policy_logits: {policy_logits.shape}")
    print(f"  opponent_action_logits: {opponent_action_logits.shape}")
    print(f"  opponent_type_preds: {opponent_type_preds.shape}")
    print(f"  actions_taken: {actions_taken.shape}")
    print(f"  rewards: {rewards.shape}")
    print(f"  opponent_actions: {opponent_actions.shape}")
    print(f"  opponent_type_true: {opponent_type_true.shape}")
    
    try:
        # This should work now with the batch size alignment fix
        loss_dict = loss_fn(
            policy_logits=policy_logits,
            opponent_action_logits=opponent_action_logits,
            opponent_type_preds=opponent_type_preds,
            actions_taken=actions_taken,
            rewards=rewards,
            opponent_actions=opponent_actions,
            opponent_type_true=opponent_type_true
        )
        
        print("\n✅ SUCCESS: Loss calculation completed without errors!")
        print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
        print(f"Policy loss: {loss_dict['policy_loss'].item():.4f}")
        print(f"Action prediction loss: {loss_dict['action_prediction_loss'].item():.4f}")
        print(f"Type prediction loss: {loss_dict['type_prediction_loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_batch_size_mismatch()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")