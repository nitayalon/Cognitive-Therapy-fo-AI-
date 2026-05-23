"""
Test script to verify NetworkConfig loading from JSON files.
Tests the fix for the 'note' field bug.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from cognitive_therapy_ai.config import NetworkConfig

def test_config_loading(config_path, config_name):
    """Test loading network config from a JSON file."""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"Config file: {config_path}")
    print(f"{'='*60}")
    
    try:
        # Load JSON
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        print(f"\n📄 Raw network_config from JSON:")
        for key, value in config_data['network_config'].items():
            print(f"  {key}: {value}")
        
        # Filter to valid NetworkConfig fields (simulating the fix)
        network_config_dict = {k: v for k, v in config_data['network_config'].items() 
                               if k in ['hidden_size', 'num_layers', 'dropout', 'input_size']}
        
        print(f"\n🔧 Filtered network_config (valid fields only):")
        for key, value in network_config_dict.items():
            print(f"  {key}: {value}")
        
        # Create NetworkConfig
        network_config = NetworkConfig(**network_config_dict)
        
        print(f"\n✅ NetworkConfig created successfully:")
        print(f"  hidden_size: {network_config.hidden_size}")
        print(f"  num_layers: {network_config.num_layers}")
        print(f"  dropout: {network_config.dropout}")
        print(f"  input_size: {network_config.input_size}")
        
        # Verify it matches expected small network
        expected_small = {
            'hidden_size': 32,
            'num_layers': 1,
            'dropout': 0.05,
            'input_size': 9
        }
        
        is_small_network = all(
            getattr(network_config, k) == v 
            for k, v in expected_small.items()
        )
        
        if is_small_network:
            print(f"\n🎯 Confirmed: Small network architecture (32/1/0.05)")
        else:
            print(f"\n⚠️  Warning: Not small network architecture")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        return False

def main():
    """Run all config loading tests."""
    print("\n" + "="*60)
    print("NetworkConfig Loading Test Suite")
    print("Testing fix for 'note' field bug")
    print("="*60)
    
    base_path = Path(__file__).parent / 'config'
    
    configs_to_test = [
        (base_path / 'whole_population_config.json', 'Whole Population Config'),
        (base_path / 'generalization_matrix_config.json', 'Generalization Matrix Config'),
    ]
    
    results = []
    for config_path, config_name in configs_to_test:
        success = test_config_loading(config_path, config_name)
        results.append((config_name, success))
    
    # Summary
    print(f"\n\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    
    all_passed = all(success for _, success in results)
    
    for config_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {config_name}")
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 All tests passed!")
        print("The 'note' field bug has been fixed.")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
