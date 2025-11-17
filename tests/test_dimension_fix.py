#!/usr/bin/env python3
"""
Test script to verify the dimension mismatch fix.
"""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cognitive_therapy_ai.network import GameLSTM

def test_dimension_handling():
    """Test that the network handles various input dimensions correctly."""
    print("Testing GameLSTM dimension handling...")
    
    # Create network
    network = GameLSTM(input_size=5, hidden_size=32, num_layers=2)
    device = torch.device('cpu')
    
    # Test 1: 2D input (sequence_length, input_size) - unbatched
    print("\nTest 1: 2D input (unbatched)")
    x_2d = torch.randn(10, 5)  # (seq_len=10, input_size=5)
    print(f"Input shape: {x_2d.shape}")
    
    try:
        hidden = network.init_hidden(1, device)  # This creates (num_layers, batch_size=1, hidden_size)
        policy_logits, opponent_logits, value_est, new_hidden = network.forward(x_2d, hidden)
        print(f"✅ Success! Policy shape: {policy_logits.shape}, Opponent: {opponent_logits.shape}, Value: {value_est.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: 3D input (batch_size, sequence_length, input_size) - batched
    print("\nTest 2: 3D input (batched)")
    x_3d = torch.randn(2, 10, 5)  # (batch_size=2, seq_len=10, input_size=5)
    print(f"Input shape: {x_3d.shape}")
    
    try:
        hidden = network.init_hidden(2, device)  # This creates (num_layers, batch_size=2, hidden_size)
        policy_logits, opponent_logits, value_est, new_hidden = network.forward(x_3d, hidden)
        print(f"✅ Success! Policy shape: {policy_logits.shape}, Opponent: {opponent_logits.shape}, Value: {value_est.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Single state vector (like from game.get_state_vector())
    print("\nTest 3: Single state vector (like from game)")
    state_vector = np.random.randn(5)  # (input_size,)
    state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 5)
    print(f"State tensor shape: {state_tensor.shape}")
    
    try:
        hidden = network.init_hidden(1, device)
        policy_logits, opponent_logits, value_est, new_hidden = network.forward(state_tensor, hidden)
        print(f"✅ Success! Policy shape: {policy_logits.shape}, Opponent: {opponent_logits.shape}, Value: {value_est.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Problematic case - 2D input with batch dimension but no sequence dimension
    print("\nTest 4: 2D input with batch dimension (should be handled)")
    x_batch_only = torch.randn(1, 5)  # (batch_size=1, input_size=5) - missing sequence dimension
    print(f"Input shape: {x_batch_only.shape}")
    
    try:
        hidden = network.init_hidden(1, device)
        policy_logits, opponent_logits, value_est, new_hidden = network.forward(x_batch_only, hidden)
        print(f"✅ Success! Policy shape: {policy_logits.shape}, Opponent: {opponent_logits.shape}, Value: {value_est.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎉 All dimension handling tests completed!")

if __name__ == "__main__":
    test_dimension_handling()