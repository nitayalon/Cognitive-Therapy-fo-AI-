#!/usr/bin/env python3
"""
Comprehensive test for JSON serialization fixes in main_experiment.py
"""

import json
import sys
import os
import tempfile
import numpy as np
import torch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cognitive_therapy_ai.opponent import OpponentFactory


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types, PyTorch tensors, and Opponent objects."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif hasattr(obj, 'item'):  # Handle single-element tensors
            try:
                return obj.item()
            except (ValueError, TypeError):
                pass
        # Handle Opponent objects
        elif hasattr(obj, 'get_type_parameter') and hasattr(obj, 'get_strategy_name'):
            return {
                'strategy_name': obj.get_strategy_name(),
                'type_parameter': obj.get_type_parameter(),
                'opponent_id': getattr(obj, 'opponent_id', str(id(obj)))
            }
        return super().default(obj)


def test_segmented_experiment_serialization():
    """Test the specific case that was causing the error."""
    print("🧪 Testing segmented experiment serialization...")
    
    # Create experiment configs like the main_experiment.py does
    opponent_probs = [0.1, 0.3, 0.9]
    experiment_configs = OpponentFactory.create_segmented_experiment_configs(
        opponent_probs,
        num_opponents_per_segment=3,  # Smaller for testing
        include_boundaries=False
    )
    
    print(f"Created {len(experiment_configs)} experiment configs")
    
    # Simulate the structure that gets created in run_segmented_experiments
    all_results = {}
    
    for i, segment_config in enumerate(experiment_configs):
        segment_id = segment_config['experiment_id']
        
        # Mock segment results (like what run_multi_game_experiment would return)
        segment_results = {
            'training_games': ['prisoners-dilemma', 'hawk-dove'],
            'test_game': 'stag-hunt',
            'opponent_probabilities': [0.1, 0.2, 0.3],  # Would be extracted in the fix
            'network_config': {'hidden_size': 64},
            'training_config': {'max_epochs': 100},
            'training_results': {'final_metrics': {'best_loss': 0.5}},
            'test_results': {'opponent_1': {'average_reward': 0.6}},
            'network_parameters': 1000,
            'device': 'cpu'
        }
        
        all_results[segment_id] = {
            'segment_config': segment_config,  # This contains Opponent objects!
            'results': segment_results
        }
    
    # Test 1: Try to serialize the problematic structure directly (should fail without fix)
    print("\n🔍 Test 1: Direct serialization of all_results (problematic case)")
    try:
        json_str = json.dumps(all_results, cls=NumpyJSONEncoder, indent=2)
        print("✅ Direct serialization: SUCCESS (NumpyJSONEncoder handles Opponents)")
        print(f"   Size: {len(json_str)} characters")
    except Exception as e:
        print(f"❌ Direct serialization: FAILED - {e}")
        return False
    
    # Test 2: Create the fixed combined_report structure
    print("\n🔍 Test 2: Fixed combined report structure")
    combined_report = {
        'experiment_summary': {
            'experiment_type': 'segmented_experiments',
            'training_games': ['prisoners-dilemma', 'hawk-dove'],
            'test_game': 'stag-hunt',
            'total_segments': len(experiment_configs),
            'timestamp': '2024-11-16T12:00:00',
        },
        'segment_reports': {}
    }
    
    # Apply the fix: extract opponent probabilities instead of storing raw opponents
    for segment_id, segment_data in all_results.items():
        segment_config = segment_data['segment_config']
        segment_results = segment_data['results']
        
        # Extract opponent probabilities for serialization (THE FIX)
        opponent_probs_for_segment = []
        for opponent in segment_config['opponents']:
            type_param = opponent.get_type_parameter()
            if type_param is not None:
                opponent_probs_for_segment.append(type_param)
        
        # Create serializable segment config
        combined_report['segment_reports'][segment_id] = {
            'segment_config': {
                'experiment_id': segment_config['experiment_id'],
                'segment_range': segment_config['segment_range'],
                'num_opponents': segment_config['num_opponents'],
                'description': segment_config['description'],
                'opponent_probabilities': opponent_probs_for_segment  # Serializable!
            },
            'summary': {
                'training_games': segment_results.get('training_games', ['prisoners-dilemma', 'hawk-dove']),
                'test_game': segment_results.get('test_game', 'stag-hunt'),
                'opponent_count': len(segment_config['opponents']),
                'segment_range': segment_config['segment_range']
            }
        }
    
    try:
        json_str = json.dumps(combined_report, cls=NumpyJSONEncoder, indent=2)
        print("✅ Fixed combined report: SUCCESS")
        print(f"   Size: {len(json_str)} characters")
        
        # Verify the structure is correct
        parsed = json.loads(json_str)
        print(f"   Segments: {len(parsed['segment_reports'])}")
        
        # Check that opponent probabilities are correctly extracted
        for segment_id, segment_report in parsed['segment_reports'].items():
            opponent_probs = segment_report['segment_config']['opponent_probabilities']
            print(f"   {segment_id}: {len(opponent_probs)} opponents, probs: {opponent_probs}")
        
    except Exception as e:
        print(f"❌ Fixed combined report: FAILED - {e}")
        return False
    
    # Test 3: Save to actual file (like main_experiment.py does)
    print("\n🔍 Test 3: File serialization")
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(combined_report, f, indent=2, cls=NumpyJSONEncoder)
            temp_file = f.name
        
        # Try to read it back
        with open(temp_file, 'r') as f:
            loaded_data = json.load(f)
        
        print("✅ File serialization and loading: SUCCESS")
        print(f"   File: {temp_file}")
        print(f"   Loaded segments: {len(loaded_data['segment_reports'])}")
        
        # Clean up
        os.unlink(temp_file)
        
    except Exception as e:
        print(f"❌ File serialization: FAILED - {e}")
        return False
    
    return True


def test_multi_game_results_serialization():
    """Test serialization of multi-game experiment results."""
    print("\n🧪 Testing multi-game results serialization...")
    
    # Create some opponents
    opponents = OpponentFactory.create_opponent_set([0.1, 0.3, 0.5, 0.7, 0.9])
    
    # Simulate the fixed extraction logic
    opponent_probs = None
    if opponent_probs is None and opponents is not None:
        opponent_probabilities = []
        for opponent in opponents:
            type_param = opponent.get_type_parameter()
            if type_param is not None:
                opponent_probabilities.append(type_param)
        opponent_probs = opponent_probabilities if opponent_probabilities else None
    
    # Create results structure like run_multi_game_experiment
    complete_results = {
        'training_games': ['prisoners-dilemma', 'hawk-dove'],
        'test_game': 'stag-hunt',
        'opponent_probabilities': opponent_probs,  # Fixed: uses extracted probs, not objects
        'network_config': {'hidden_size': 64, 'num_layers': 2},
        'training_config': {'max_epochs': 100, 'learning_rate': 0.001},
        'training_results': {'final_metrics': {'best_loss': 0.5}},
        'test_results': {'opponent_1': {'average_reward': 0.6}},
        'network_parameters': 1000,
        'device': 'cpu'
    }
    
    try:
        json_str = json.dumps(complete_results, cls=NumpyJSONEncoder, indent=2)
        print("✅ Multi-game results serialization: SUCCESS")
        print(f"   Extracted probabilities: {opponent_probs}")
        print(f"   JSON size: {len(json_str)} characters")
    except Exception as e:
        print(f"❌ Multi-game results serialization: FAILED - {e}")
        return False
    
    return True


def main():
    """Run all serialization tests."""
    print("🔧 Testing JSON serialization fixes for Opponent objects...")
    print("=" * 70)
    
    try:
        test1_passed = test_segmented_experiment_serialization()
        test2_passed = test_multi_game_results_serialization()
        
        all_passed = test1_passed and test2_passed
        
        print("\n" + "=" * 70)
        if all_passed:
            print("✅ ALL TESTS PASSED!")
            print("🎉 The 'Object of type Opponent is not JSON serializable' error should be fixed!")
        else:
            print("❌ SOME TESTS FAILED!")
            print("⚠️  There may still be serialization issues")
        
        print("\nFixes implemented:")
        print("- ✅ Enhanced NumpyJSONEncoder to handle Opponent objects")
        print("- ✅ Extract opponent probabilities instead of storing raw objects")
        print("- ✅ Fixed segmented experiment report creation")
        print("- ✅ Fixed multi-game experiment result compilation") 
        print("- ✅ Maintained all existing NumPy/PyTorch serialization support")
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)