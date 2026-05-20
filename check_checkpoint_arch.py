#!/usr/bin/env python3
"""Quick script to check checkpoint architecture."""
import torch
from pathlib import Path

# Check experiment 916788
train_dir = Path('experiments/generalization_matrix_train_916788/training/condition_0_seed_0')
task_dirs = list(train_dir.glob('generalization_matrix_task_*'))

if task_dirs:
    checkpoint_path = task_dirs[0] / 'checkpoints' / 'prisoners-dilemma_final_checkpoint.pth'
    
    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        state_dict = ckpt.get('model_state_dict', ckpt)
        
        # Check key shapes
        print("\nKey architecture parameters:")
        if 'lstm.weight_ih_l0' in state_dict:
            shape = state_dict['lstm.weight_ih_l0'].shape
            print(f"  lstm.weight_ih_l0: {shape}")
            # Shape should be [4*hidden_size, input_size]
            hidden_size = shape[0] // 4
            input_size = shape[1]
            print(f"  Inferred hidden_size: {hidden_size}")
            print(f"  Inferred input_size: {input_size}")
        
        if 'lstm.weight_ih_l1' in state_dict:
            print(f"  Has layer 1: YES (2-layer LSTM)")
        else:
            print(f"  Has layer 1: NO (1-layer LSTM)")
        
        if 'payoff_matrix_embed.0.weight' in state_dict:
            embed_shape = state_dict['payoff_matrix_embed.0.weight'].shape
            print(f"  Embedding size: {embed_shape[0]}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
else:
    print(f"No task directories found in {train_dir}")
