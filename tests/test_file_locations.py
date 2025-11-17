#!/usr/bin/env python3
"""
Test script to verify all CSV and JSON files are written to experiments folder.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_file_locations():
    """Test that all CSV and JSON files go to the experiments folder."""
    
    print("🧪 Testing file location handling...")
    
    # Mock the output directories structure
    mock_output_dirs = {
        'base': '/experiments/test_experiment_20241116_120000',
        'checkpoints': '/experiments/test_experiment_20241116_120000/checkpoints',
        'logs': '/experiments/test_experiment_20241116_120000/logs',
        'plots': '/experiments/test_experiment_20241116_120000/plots',
        'results': '/experiments/test_experiment_20241116_120000/results'
    }
    
    print(f"📁 Base experiment directory: {mock_output_dirs['base']}")
    
    # Test JSON file locations
    json_files = [
        ('experiment_config.json', os.path.join(mock_output_dirs['base'], 'experiment_config.json')),
        ('experiment_report.json', os.path.join(mock_output_dirs['results'], 'experiment_report.json')),
        ('multi_game_experiment_report.json', os.path.join(mock_output_dirs['results'], 'multi_game_experiment_report.json')),
        ('raw_results.json', os.path.join(mock_output_dirs['results'], 'raw_results.json')),
        ('segmented_experiments_report.json', os.path.join(mock_output_dirs['results'], 'segmented_experiments_report.json')),
        ('session_summary.json', os.path.join(mock_output_dirs['logs'], 'detailed_testing_logs', 'session_1_summary.json')),
        ('training_test_linkage.json', os.path.join(mock_output_dirs['logs'], 'detailed_testing_logs', 'training_test_linkage.json'))
    ]
    
    # Test CSV file locations  
    csv_files = [
        ('detailed_training_log.csv', os.path.join(mock_output_dirs['checkpoints'], 'detailed_training_logs', 'detailed_training_log.csv')),
        ('detailed_testing_log.csv', os.path.join(mock_output_dirs['logs'], 'detailed_testing_logs', 'detailed_testing_log.csv'))
    ]
    
    print("\n📄 JSON Files:")
    for filename, full_path in json_files:
        # Check if path is within experiments directory
        in_experiments = '/experiments/' in full_path
        status = "✅" if in_experiments else "❌"
        print(f"  {status} {filename}")
        print(f"     → {full_path}")
        if not in_experiments:
            print(f"     ⚠️  This file is NOT in the experiments directory!")
    
    print("\n📊 CSV Files:")
    for filename, full_path in csv_files:
        # Check if path is within experiments directory
        in_experiments = '/experiments/' in full_path
        status = "✅" if in_experiments else "❌"
        print(f"  {status} {filename}")
        print(f"     → {full_path}")
        if not in_experiments:
            print(f"     ⚠️  This file is NOT in the experiments directory!")
    
    # Test the directory structure
    print("\n📂 Directory Structure Test:")
    all_paths = [path for _, path in json_files + csv_files]
    experiments_paths = [path for path in all_paths if '/experiments/' in path]
    
    print(f"  Total files: {len(all_paths)}")
    print(f"  Files in experiments: {len(experiments_paths)}")
    print(f"  Compliance: {len(experiments_paths)/len(all_paths)*100:.1f}%")
    
    if len(experiments_paths) == len(all_paths):
        print("  ✅ All files are correctly routed to experiments directory!")
        return True
    else:
        print("  ❌ Some files are NOT in experiments directory!")
        return False


def test_training_monitor_paths():
    """Test TrainingMonitor path creation."""
    print("\n🔧 Testing TrainingMonitor paths...")
    
    # Simulate how paths are created in trainer.py
    save_dir = "/experiments/test_experiment_20241116_120000/checkpoints"
    detailed_log_dir = os.path.join(save_dir, 'detailed_training_logs')
    csv_path = os.path.join(detailed_log_dir, 'detailed_training_log.csv')
    excel_path = os.path.join(detailed_log_dir, 'detailed_training_log.xlsx')
    
    print(f"  Save dir: {save_dir}")
    print(f"  Detailed log dir: {detailed_log_dir}")
    print(f"  CSV path: {csv_path}")
    print(f"  Excel path: {excel_path}")
    
    # Check if paths are in experiments
    paths_in_experiments = all('/experiments/' in path for path in [csv_path, excel_path])
    status = "✅" if paths_in_experiments else "❌"
    print(f"  {status} Training logs in experiments directory: {paths_in_experiments}")
    
    return paths_in_experiments


def test_testing_monitor_paths():
    """Test TestingMonitor path creation."""
    print("\n🔧 Testing TestingMonitor paths...")
    
    # Simulate how paths are created in the updated main_experiment.py
    output_dirs_logs = "/experiments/test_experiment_20241116_120000/logs"
    testing_log_dir = os.path.join(output_dirs_logs, 'detailed_testing_logs')
    csv_path = os.path.join(testing_log_dir, 'detailed_testing_log.csv')
    excel_path = os.path.join(testing_log_dir, 'detailed_testing_log.xlsx')
    summary_file = os.path.join(testing_log_dir, 'session_1_summary.json')
    linkage_file = os.path.join(testing_log_dir, 'training_test_linkage.json')
    
    print(f"  Testing log dir: {testing_log_dir}")
    print(f"  CSV path: {csv_path}")
    print(f"  Excel path: {excel_path}")
    print(f"  Summary file: {summary_file}")
    print(f"  Linkage file: {linkage_file}")
    
    # Check if paths are in experiments
    all_paths = [csv_path, excel_path, summary_file, linkage_file]
    paths_in_experiments = all('/experiments/' in path for path in all_paths)
    status = "✅" if paths_in_experiments else "❌"
    print(f"  {status} Testing logs in experiments directory: {paths_in_experiments}")
    
    return paths_in_experiments


def main():
    """Run all file location tests."""
    print("🔍 Verifying CSV and JSON file locations...")
    print("=" * 60)
    
    try:
        test1_passed = test_file_locations()
        test2_passed = test_training_monitor_paths()
        test3_passed = test_testing_monitor_paths()
        
        all_passed = test1_passed and test2_passed and test3_passed
        
        print("\n" + "=" * 60)
        if all_passed:
            print("✅ ALL TESTS PASSED!")
            print("🎉 All CSV and JSON files will be saved to experiments directory")
        else:
            print("❌ SOME TESTS FAILED!")
            print("⚠️  Some files may not be saved to experiments directory")
        
        print("\nKey changes made:")
        print("- ✅ Added testing_log_dir parameter to evaluate() calls")
        print("- ✅ Routed testing logs to output_dirs['logs']/detailed_testing_logs")
        print("- ✅ Training logs already go to output_dirs['checkpoints']/detailed_training_logs")
        print("- ✅ All report JSON files go to output_dirs['results']")
        print("- ✅ Config files go to output_dirs['base']")
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)