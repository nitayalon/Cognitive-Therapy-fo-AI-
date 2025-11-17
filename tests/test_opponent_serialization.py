#!/usr/bin/env python3
"""
Test script to verify JSON serialization works with Opponent objects.
"""

import json
import sys
import os
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


def test_opponent_json_serialization():
    """Test that Opponent objects can be serialized to JSON."""
    print("🧪 Testing Opponent JSON serialization...")
    
    # Create some opponents
    opponents = OpponentFactory.create_opponent_set([0.1, 0.3, 0.5, 0.7, 0.9])
    
    print(f"Created {len(opponents)} opponents")
    
    # Test 1: Direct serialization of opponents list
    try:
        json_str = json.dumps(opponents, cls=NumpyJSONEncoder, indent=2)
        print("✅ Direct opponent list serialization: SUCCESS")
        print(f"   Sample: {json_str[:100]}...")
    except Exception as e:
        print(f"❌ Direct opponent list serialization: FAILED - {e}")
        return False
    
    # Test 2: Serialization in a results dictionary (like in main_experiment.py)
    mock_results = {
        'training_games': ['prisoners-dilemma', 'hawk-dove'],
        'test_game': 'stag-hunt',
        'opponents': opponents,  # This would cause the error
        'opponent_probabilities': [0.1, 0.3, 0.5, 0.7, 0.9],
        'network_parameters': 1000,
        'device': 'cpu'
    }
    
    try:
        json_str = json.dumps(mock_results, cls=NumpyJSONEncoder, indent=2)
        print("✅ Results dictionary with opponents: SUCCESS")
        print(f"   Sample: {json_str[:200]}...")
    except Exception as e:
        print(f"❌ Results dictionary with opponents: FAILED - {e}")
        return False
    
    # Test 3: Extract probabilities from opponents (like the fix)
    extracted_probs = []
    for opponent in opponents:
        type_param = opponent.get_type_parameter()
        if type_param is not None:
            extracted_probs.append(type_param)
    
    fixed_results = {
        'training_games': ['prisoners-dilemma', 'hawk-dove'],
        'test_game': 'stag-hunt',
        'opponent_probabilities': extracted_probs,  # Use extracted probs instead of objects
        'network_parameters': 1000,
        'device': 'cpu'
    }
    
    try:
        json_str = json.dumps(fixed_results, cls=NumpyJSONEncoder, indent=2)
        print("✅ Fixed results dictionary: SUCCESS")
        print(f"   Extracted probabilities: {extracted_probs}")
    except Exception as e:
        print(f"❌ Fixed results dictionary: FAILED - {e}")
        return False
    
    return True


def test_numpy_torch_serialization():
    """Test that NumPy and PyTorch objects still serialize correctly."""
    print("\n🧪 Testing NumPy and PyTorch serialization...")
    
    test_data = {
        'numpy_int': np.int64(42),
        'numpy_float': np.float32(3.14),
        'numpy_array': np.array([1, 2, 3]),
        'torch_tensor': torch.tensor([4.0, 5.0, 6.0]),
        'regular_data': {'key': 'value', 'number': 123}
    }
    
    try:
        json_str = json.dumps(test_data, cls=NumpyJSONEncoder, indent=2)
        print("✅ NumPy and PyTorch serialization: SUCCESS")
        
        # Test deserialization
        parsed_data = json.loads(json_str)
        print("✅ JSON parsing: SUCCESS")
        print(f"   Parsed numpy_int: {parsed_data['numpy_int']} (type: {type(parsed_data['numpy_int'])})")
        print(f"   Parsed numpy_array: {parsed_data['numpy_array']}")
        
    except Exception as e:
        print(f"❌ NumPy and PyTorch serialization: FAILED - {e}")
        return False
    
    return True


def main():
    """Run all serialization tests."""
    print("🔧 Testing JSON serialization fixes...")
    print("=" * 60)
    
    try:
        test1_passed = test_opponent_json_serialization()
        test2_passed = test_numpy_torch_serialization()
        
        all_passed = test1_passed and test2_passed
        
        print("\n" + "=" * 60)
        if all_passed:
            print("✅ ALL TESTS PASSED!")
            print("🎉 JSON serialization should work correctly now")
        else:
            print("❌ SOME TESTS FAILED!")
            print("⚠️  There may still be serialization issues")
        
        print("\nKey fixes implemented:")
        print("- ✅ Updated NumpyJSONEncoder to handle Opponent objects")
        print("- ✅ Added opponent probability extraction in result compilation")
        print("- ✅ Maintained backward compatibility with NumPy/PyTorch types")
        print("- ✅ Graceful handling of edge cases")
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)