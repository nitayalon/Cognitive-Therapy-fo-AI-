#!/usr/bin/env python3
"""
Simple demonstration of the JSON serialization fix.
"""

import json
import numpy as np
import torch

def test_serialization():
    """Show that the custom serializer handles NumPy types."""
    
    def json_serialize_default(obj):
        """Custom JSON serialization for NumPy and PyTorch types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif torch.is_tensor(obj):
            return obj.detach().cpu().numpy().tolist()
        elif hasattr(obj, 'item'):  # Handle single-element tensors
            try:
                return obj.item()
            except (ValueError, TypeError):
                pass
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    # Test data that would cause "Object of type float32 is not JSON serializable"
    problematic_data = {
        'numpy_float32': np.float32(3.14159),
        'numpy_float64': np.float64(2.71828),
        'numpy_int32': np.int32(42),
        'numpy_array': np.array([1.0, 2.0, 3.0]),
        'regular_value': 1.23
    }
    
    print("Testing JSON serialization fix...")
    print("=" * 50)
    
    # This would fail without the custom serializer
    try:
        json_without_custom = json.dumps(problematic_data)
        print("❌ Standard JSON encoder should have failed!")
    except TypeError as e:
        print(f"✅ Standard JSON encoder failed as expected: {e}")
    
    # This should work with the custom serializer
    try:
        json_with_custom = json.dumps(problematic_data, default=json_serialize_default, indent=2)
        print("✅ Custom JSON encoder succeeded!")
        print("Serialized JSON:")
        print(json_with_custom)
        
        # Verify it can be read back
        parsed = json.loads(json_with_custom)
        print("✅ JSON can be parsed back successfully!")
        print(f"Parsed data: {parsed}")
        
    except Exception as e:
        print(f"❌ Custom JSON encoder failed: {e}")
        return False
    
    print("\n🎉 JSON serialization fix verified!")
    return True

if __name__ == "__main__":
    test_serialization()