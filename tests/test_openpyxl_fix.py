#!/usr/bin/env python3
"""
DEPRECATED: This test is no longer needed.
Excel export functionality has been removed from the monitoring system.
Only CSV files are now exported, eliminating the need for openpyxl dependency.
"""

def test_openpyxl_import():
    """Test that openpyxl can be imported."""
    try:
        import openpyxl
        print(f"✅ openpyxl successfully imported (version: {openpyxl.__version__})")
        return True
    except ImportError as e:
        print(f"❌ Failed to import openpyxl: {e}")
        return False

def test_pandas_excel_export():
    """Test that pandas can export to Excel using openpyxl."""
    try:
        import pandas as pd
        import os
        
        # Create test data similar to what the testing monitor generates
        test_data = {
            'session_id': [1, 2, 3],
            'game': ['prisoners_dilemma', 'hawk_dove', 'stag_hunt'],
            'opponent_id': ['opp_0.3', 'opp_0.7', 'opp_0.5'],
            'agent_action': [1, 0, 1],  # 1=cooperate, 0=defect
            'opponent_action': [0, 1, 1],
            'agent_reward': [0.0, 2.5, 3.0],
            'round_number': [1, 1, 1]
        }
        
        df = pd.DataFrame(test_data)
        
        # Test Excel export (this is what was failing)
        test_file = 'test_monitoring_export.xlsx'
        df.to_excel(test_file, index=False, sheet_name='Testing_Log')
        
        if os.path.exists(test_file):
            print(f"✅ Successfully created Excel file: {test_file}")
            file_size = os.path.getsize(test_file)
            print(f"   File size: {file_size} bytes")
            
            # Clean up
            os.remove(test_file)
            print("✅ Test file cleaned up")
            return True
        else:
            print("❌ Excel file was not created")
            return False
            
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during Excel export: {e}")
        return False

def main():
    """Run all tests to verify openpyxl installation."""
    print("Testing openpyxl installation and Excel export functionality...")
    print("=" * 60)
    
    # Test 1: Basic import
    import_success = test_openpyxl_import()
    
    # Test 2: Full Excel export pipeline
    export_success = test_pandas_excel_export()
    
    print("=" * 60)
    if import_success and export_success:
        print("🎉 All tests passed! The 'No module named openpyxl' error should be resolved.")
        print("   The testing monitor should now be able to create Excel files successfully.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        if not import_success:
            print("   - Install openpyxl: pip install openpyxl")
        if not export_success:
            print("   - Check pandas installation: pip install pandas")

if __name__ == "__main__":
    main()