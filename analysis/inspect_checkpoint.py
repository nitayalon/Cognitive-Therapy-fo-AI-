import torch
from pathlib import Path

# Load a sample checkpoint to inspect structure
checkpoint_path = "experiments/generalization_matrix_train_888509/training/condition_0_seed_0/generalization_matrix_task_0_20260324_131903/checkpoints/prisoners-dilemma_final_checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("Checkpoint keys:")
for key in checkpoint.keys():
    print(f"  {key}")

print("\nModel state dict keys:")
for key in checkpoint['model_state_dict'].keys():
    print(f"  {key}")

print(f"\nHidden size: {checkpoint.get('hidden_size', 'NOT FOUND')}")
print(f"\nNum layers in LSTM:")
lstm_layers = [k for k in checkpoint['model_state_dict'].keys() if 'lstm.weight' in k]
print(f"  LSTM weight keys: {lstm_layers}")
