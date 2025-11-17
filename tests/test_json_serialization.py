#!/usr/bin/env python3
"""
Test script to verify the JSON serialization fix for TestingMonitor.
"""

import sys
import os
import tempfile
import json
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cognitive_therapy_ai.testing_monitor import TestingMonitor

def test_json_serialization():
    """Test that JSON serialization works with NumPy and PyTorch types."""
    print("Testing JSON serialization fix...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a testing monitor
        monitor = TestingMonitor(temp_dir, save_frequency=1)
        
        # Create test data with various NumPy types that would cause serialization issues
        test_data = {
            'numpy_float32': np.float32(3.14),
            'numpy_float64': np.float64(2.71),
            'numpy_int32': np.int32(42),
            'numpy_int64': np.int64(123),
            'numpy_array': np.array([1.0, 2.0, 3.0]),
            'torch_tensor': torch.tensor([4.0, 5.0]),
            'regular_float': 1.23,
            'regular_int': 456,
            'string': 'test_string'
        }
        
        # Test the _json_serialize_default method
        print("Testing individual type serialization...")
        try:
            for key, value in test_data.items():
                try:
                    serialized = monitor._json_serialize_default(value)
                    print(f"✅ {key} ({type(value).__name__}): {serialized}")
                except TypeError:
                    # Regular Python types should raise TypeError (handled by default JSON encoder)
                    if isinstance(value, (int, float, str)):
                        print(f"✅ {key} ({type(value).__name__}): handled by default encoder")
                    else:
                        print(f"❌ {key} ({type(value).__name__}): unexpected TypeError")
                        raise
        except Exception as e:
            print(f"❌ Individual type serialization failed: {e}")
            return False
        
        # Test full JSON serialization
        print("\nTesting full JSON serialization...")
        try:
            json_str = json.dumps(test_data, default=monitor._json_serialize_default, indent=2)
            print("✅ JSON serialization successful!")
            print("Sample JSON output:")
            print(json_str[:200] + "..." if len(json_str) > 200 else json_str)
        except Exception as e:
            print(f"❌ Full JSON serialization failed: {e}")
            return False
        
        # Test finalize_session (which triggered the original error)
        print("\nTesting finalize_session method...")
        try:
            # Add some mock session data
            monitor.session_stats[1] = [
                {'agent_reward': np.float32(1.5), 'prediction_accuracy': np.float64(0.8), 'agent_action': 0, 'opponent_action': 1},
                {'agent_reward': np.float32(2.0), 'prediction_accuracy': np.float64(0.7), 'agent_action': 1, 'opponent_action': 0}
            ]
            monitor.current_test_session = 1
            
            # This should not raise a JSON serialization error anymore
            monitor.finalize_session()
            print("✅ finalize_session completed without JSON serialization errors!")
            
            # Check that the file was created
            summary_file = os.path.join(temp_dir, 'session_1_summary.json')
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                print(f"✅ Session summary file created and readable: {summary_file}")
                print(f"Sample summary data: {summary_data}")
            else:
                print(f"❌ Session summary file not created: {summary_file}")
                return False
                
        except Exception as e:
            print(f"❌ finalize_session failed: {e}")
            return False
    
    print("\n🎉 All JSON serialization tests passed!")
    return True

if __name__ == "__main__":
    success = test_json_serialization()
    if not success:
        sys.exit(1)