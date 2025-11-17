#!/usr/bin/env python3
"""
Test the equally spaced opponent generation system.
"""

import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from cognitive_therapy_ai import OpponentFactory

def test_equally_spaced_opponents():
    """Test the equally spaced opponent generation."""
    print("🧪 Testing Equally Spaced Opponent Generation")
    print("=" * 60)
    
    # Test case 1: Basic range
    print("\n📊 Test 1: Basic range [0.1, 0.9] with 11 opponents")
    input_probs = [0.1, 0.9]
    opponents = OpponentFactory.create_equally_spaced_opponents(
        input_probs, 
        num_opponents=11, 
        include_boundaries=False
    )
    
    probs = [opp.get_type_parameter() for opp in opponents]
    print(f"Generated {len(opponents)} opponents:")
    for i, prob in enumerate(probs):
        print(f"  {i+1:2d}. {prob:.3f}")
    
    # Test case 2: With boundaries
    print("\n📊 Test 2: Range [0.2, 0.8] with boundaries included")
    input_probs = [0.2, 0.8] 
    opponents = OpponentFactory.create_equally_spaced_opponents(
        input_probs,
        num_opponents=7,
        include_boundaries=True
    )
    
    probs = [opp.get_type_parameter() for opp in opponents]
    print(f"Generated {len(opponents)} opponents:")
    for i, prob in enumerate(probs):
        print(f"  {i+1:2d}. {prob:.3f}")
    
    # Test case 3: Multiple input points (should use min/max)
    print("\n📊 Test 3: Multiple input points [0.1, 0.3, 0.5, 0.7, 0.9]")
    input_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    opponents = OpponentFactory.create_equally_spaced_opponents(
        input_probs,
        num_opponents=11,
        include_boundaries=False
    )
    
    probs = [opp.get_type_parameter() for opp in opponents]
    print(f"Input range: [{min(input_probs):.1f}, {max(input_probs):.1f}]")
    print(f"Generated {len(opponents)} opponents:")
    for i, prob in enumerate(probs):
        print(f"  {i+1:2d}. {prob:.3f}")
    
    # Test case 4: Edge case - single probability
    print("\n📊 Test 4: Edge case - single probability [0.5]")
    input_probs = [0.5]
    opponents = OpponentFactory.create_equally_spaced_opponents(
        input_probs,
        num_opponents=11,
        include_boundaries=False
    )
    
    probs = [opp.get_type_parameter() for opp in opponents]
    print(f"Generated {len(opponents)} opponents:")
    for i, prob in enumerate(probs):
        print(f"  {i+1:2d}. {prob:.3f}")
    
    print("\n✅ All equally spaced tests completed successfully!")


def test_segmented_experiments():
    """Test the segmented experiment generation."""
    print("\n🔬 Testing Segmented Experiment Generation")
    print("=" * 60)
    
    # Test case 1: The user's example - [0.1, 0.3, 0.9]
    print("\n📊 Test 1: User example [0.1, 0.3, 0.9] with 5 opponents per segment")
    input_probs = [0.1, 0.3, 0.9]
    experiment_configs = OpponentFactory.create_segmented_experiment_configs(
        input_probs,
        num_opponents_per_segment=5,
        include_boundaries=False
    )
    
    print(f"Generated {len(experiment_configs)} segment experiments:")
    for i, config in enumerate(experiment_configs):
        print(f"\n  Experiment {i+1}: {config['experiment_id']}")
        print(f"    Description: {config['description']}")
        print(f"    Range: {config['segment_range']}")
        print(f"    Opponents: {config['num_opponents']}")
        
        probs = [opp.get_type_parameter() for opp in config['opponents']]
        print(f"    Defection probabilities: {[f'{p:.3f}' for p in probs]}")
    
    # Test case 2: More segments
    print("\n📊 Test 2: Multiple segments [0.1, 0.3, 0.5, 0.7, 0.9] with 3 opponents per segment")
    input_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    experiment_configs = OpponentFactory.create_segmented_experiment_configs(
        input_probs,
        num_opponents_per_segment=3,
        include_boundaries=False
    )
    
    print(f"Generated {len(experiment_configs)} segment experiments:")
    for i, config in enumerate(experiment_configs):
        print(f"\n  Experiment {i+1}: {config['experiment_id']}")
        print(f"    Range: {config['segment_range']}")
        
        probs = [opp.get_type_parameter() for opp in config['opponents']]
        print(f"    Defection probabilities: {[f'{p:.3f}' for p in probs]}")
    
    print("\n✅ All segmented experiment tests completed successfully!")


def test_all():
    """Run all tests."""
    print("🧪 Testing Opponent Generation Systems")
    print("=" * 80)
    
    test_equally_spaced_opponents()
    test_segmented_experiments()
    
    print("\n🎯 All Tests Completed Successfully!")
    print("=" * 80)

if __name__ == "__main__":
    test_all()