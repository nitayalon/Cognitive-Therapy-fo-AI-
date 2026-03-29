"""
Test script to verify eval-only mode accepts arbitrary task_id values.

This tests the fix for the issue where task_id validation rejected
large SLURM_ARRAY_TASK_ID values in eval-only mode.
"""

import os
import sys
import json
import tempfile
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cognitive_therapy_ai import GameFactory, GameLSTM
from cognitive_therapy_ai.config import NetworkConfig, TrainingConfig

def test_eval_only_with_large_task_id():
    """Test that eval-only mode accepts task_id > num_training_conditions"""
    
    print("Testing eval-only mode with task_id=998...")
    
    # Create a temporary checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pth')
        
        # Create a simple network and save it
        game = GameFactory.create_game('prisoners-dilemma')
        network_config = NetworkConfig()
        network = GameLSTM(
            input_size=5,
            hidden_size=network_config.hidden_size,
            num_layers=network_config.num_layers,
            dropout=network_config.dropout,
            num_actions=2
        )
        
        checkpoint = {
            'model_state_dict': network.state_dict(),
            'game_name': 'prisoners-dilemma',
            'task_id': 0,
            'epoch': 100
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Now test the main_experiment script with task_id=998
        # We'll import and call the function directly
        from main_experiment import run_generalization_matrix_experiment
        
        network_config = NetworkConfig()
        training_config = TrainingConfig(max_epochs=5, num_games=10)
        device = torch.device('cpu')
        
        output_dir = os.path.join(tmpdir, 'test_output')
        os.makedirs(output_dir)
        
        output_dirs = {
            'base': output_dir,
            'checkpoints': os.path.join(output_dir, 'checkpoints'),
            'logs': os.path.join(output_dir, 'logs'),
            'plots': os.path.join(output_dir, 'plots'),
            'results': os.path.join(output_dir, 'results')
        }
        
        for dir_path in output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Use the default matrix config
        matrix_config_path = 'config/generalization_matrix_config.json'
        
        try:
            # This should NOT raise ValueError about task_id=998
            result = run_generalization_matrix_experiment(
                task_id=998,  # Large task_id that would fail before the fix
                matrix_config_path=matrix_config_path,
                network_config=network_config,
                training_config=training_config,
                device=device,
                output_dirs=output_dirs,
                use_adaptive_loss=False,
                agent_type='vanilla',
                mode='eval-only',
                checkpoint_path=checkpoint_path,
                test_condition_ids=[0]  # Test on condition 0
            )
            
            print("✓ SUCCESS: eval-only mode accepted task_id=998")
            print(f"  Result keys: {list(result.keys())}")
            return True
            
        except ValueError as e:
            if "Invalid task_id" in str(e):
                print(f"✗ FAILED: Still rejecting task_id=998: {e}")
                return False
            else:
                raise
        except Exception as e:
            print(f"✗ FAILED with unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    success = test_eval_only_with_large_task_id()
    sys.exit(0 if success else 1)
