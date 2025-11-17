#!/usr/bin/env python3
"""
Test script to verify that CSV-only export works after removing Excel functionality.
This ensures that the monitoring system works correctly without openpyxl dependency.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_csv_only_export():
    """Test that monitoring works with CSV-only export."""
    print("Testing CSV-only export functionality...")
    print("=" * 50)
    
    try:
        from cognitive_therapy_ai.training_monitor import TrainingMonitor
        from cognitive_therapy_ai.testing_monitor import TestingMonitor
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir}")
            
            # Test TrainingMonitor
            print("\n🔧 Testing TrainingMonitor...")
            training_dir = os.path.join(temp_dir, "training_logs")
            training_monitor = TrainingMonitor(
                output_dir=training_dir,
                network_serial_id="test_network_001",
                save_frequency=5
            )
            
            # Check that only CSV path exists (no Excel path)
            assert hasattr(training_monitor, 'csv_path'), "TrainingMonitor should have csv_path"
            assert not hasattr(training_monitor, 'excel_path'), "TrainingMonitor should not have excel_path"
            
            print(f"  ✅ TrainingMonitor CSV path: {training_monitor.csv_path}")
            
            # Test TestingMonitor
            print("\n🔧 Testing TestingMonitor...")
            testing_dir = os.path.join(temp_dir, "testing_logs")
            testing_monitor = TestingMonitor(
                output_dir=testing_dir,
                network_serial_id="test_network_001",
                save_frequency=5
            )
            
            # Check that only CSV path exists (no Excel path)
            assert hasattr(testing_monitor, 'csv_path'), "TestingMonitor should have csv_path"
            assert not hasattr(testing_monitor, 'excel_path'), "TestingMonitor should not have excel_path"
            
            print(f"  ✅ TestingMonitor CSV path: {testing_monitor.csv_path}")
            
            # Simulate some training data
            print("\n📝 Simulating training data logging...")
            for i in range(3):
                training_monitor.log_training_step(
                    iteration=i,
                    epoch=1,
                    game_step=i,
                    game_name="prisoners_dilemma",
                    opponent_name=f"opp_{0.5}",
                    opponent_type="probabilistic",
                    total_loss=1.5 - (i * 0.1),
                    rl_loss=0.8 - (i * 0.05),
                    rl_loss_normalized=0.8 - (i * 0.05),
                    opponent_policy_loss=0.7 - (i * 0.05),
                    opponent_policy_loss_normalized=0.7 - (i * 0.05),
                    value_loss=0.0,
                    value_loss_normalized=0.0,
                    agent_action=1,
                    opponent_action=0,
                    agent_reward=0.0,
                    opponent_reward=5.0,
                    cumulative_reward=0.0,
                    agent_policy_cooperate=0.6,
                    agent_policy_defect=0.4,
                    opponent_predicted_cooperate=0.3,
                    opponent_predicted_defect=0.7,
                    agent_predicted_value=2.5,
                    learning_rate=0.001,
                    loss_alpha=0.5,
                    payoff_matrix=[[3, 0], [5, 1]]
                )
            
            # Force save
            training_monitor._save_to_disk()
            
            # Check that CSV was created
            csv_file = training_monitor.csv_path
            assert os.path.exists(csv_file), f"CSV file should exist: {csv_file}"
            print(f"  ✅ Training CSV file created: {csv_file}")
            
            # Check CSV content
            with open(csv_file, 'r') as f:
                content = f.read()
                assert 'network_serial_id' in content, "CSV should contain headers"
                assert 'test_network_001' in content, "CSV should contain test data"
            print(f"  ✅ Training CSV file contains expected data")
            
            # Simulate some testing data
            print("\n📝 Simulating testing data logging...")
            for i in range(3):
                testing_monitor.log_test_step(
                    test_session=1,
                    test_iteration=i,
                    game_name="stag_hunt",
                    opponent_name=f"opp_{0.7}",
                    opponent_type="probabilistic",
                    game_step_in_session=i,
                    predicted_opponent_policy_defect=0.7,
                    predicted_opponent_policy_cooperate=0.3,
                    predicted_opponent_value=1.5,
                    predicted_opponent_cooperation_likelihood=0.3,
                    agent_policy_logits_cooperate=0.1,
                    agent_policy_logits_defect=-0.1,
                    agent_policy_probs_cooperate=0.55,
                    agent_policy_probs_defect=0.45,
                    agent_value_estimate=2.0,
                    agent_action=1,
                    opponent_action=1,
                    agent_reward=3.0,
                    opponent_reward=3.0,
                    payoff_matrix=[[3, 0], [5, 1]],
                    game_round=i+1,
                    session_cumulative_reward=3.0 * (i+1)
                )
            
            # Force save
            testing_monitor._save_to_disk()
            
            # Check that CSV was created
            csv_file = testing_monitor.csv_path
            assert os.path.exists(csv_file), f"CSV file should exist: {csv_file}"
            print(f"  ✅ Testing CSV file created: {csv_file}")
            
            # Check CSV content
            with open(csv_file, 'r') as f:
                content = f.read()
                assert 'network_serial_id' in content, "CSV should contain headers"
                assert 'test_network_001' in content, "CSV should contain test data"
            print(f"  ✅ Testing CSV file contains expected data")
            
            print("\n🎉 All tests passed!")
            return True
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_no_excel_references():
    """Test that Excel references have been removed from monitor classes."""
    print("\n🔍 Checking for Excel references...")
    
    try:
        from cognitive_therapy_ai.training_monitor import TrainingMonitor
        from cognitive_therapy_ai.testing_monitor import TestingMonitor
        
        # Create instances to check attributes
        with tempfile.TemporaryDirectory() as temp_dir:
            training_monitor = TrainingMonitor(
                output_dir=temp_dir,
                network_serial_id="test",
                save_frequency=5
            )
            
            testing_monitor = TestingMonitor(
                output_dir=temp_dir,
                network_serial_id="test",
                save_frequency=5
            )
            
            # Check that no Excel attributes exist
            assert not hasattr(training_monitor, 'excel_path'), "TrainingMonitor should not have excel_path attribute"
            assert not hasattr(testing_monitor, 'excel_path'), "TestingMonitor should not have excel_path attribute"
            
            print("  ✅ No excel_path attributes found in monitor classes")
            
            # Check methods don't reference Excel
            import inspect
            
            # Check TrainingMonitor methods
            training_source = inspect.getsource(TrainingMonitor)
            assert 'to_excel' not in training_source, "TrainingMonitor source should not contain 'to_excel'"
            assert '.xlsx' not in training_source, "TrainingMonitor source should not contain '.xlsx'"
            
            # Check TestingMonitor methods  
            testing_source = inspect.getsource(TestingMonitor)
            assert 'to_excel' not in testing_source, "TestingMonitor source should not contain 'to_excel'"
            assert '.xlsx' not in testing_source, "TestingMonitor source should not contain '.xlsx'"
            
            print("  ✅ No Excel references found in monitor class source code")
            return True
            
    except Exception as e:
        print(f"❌ Excel reference check failed: {e}")
        return False

def main():
    """Run all tests."""
    print("CSV-Only Export Verification")
    print("=" * 60)
    
    # Test 1: Basic CSV functionality
    test1_result = test_csv_only_export()
    
    # Test 2: No Excel references
    test2_result = test_no_excel_references()
    
    print("\n" + "=" * 60)
    if test1_result and test2_result:
        print("🎉 All tests passed! Excel export has been successfully removed.")
        print("   Only CSV files will be created for monitoring data.")
        print("   The 'No module named openpyxl' error should no longer occur.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())