#!/usr/bin/env python3
"""
Test script to verify that the argument parsing bug is fixed.
This tests various combinations of valid and invalid game arguments.
"""

import sys
import subprocess
import os

def test_argument_parsing():
    """Test different argument combinations to verify the fix."""
    print("Testing argument parsing fix...")
    print("=" * 50)
    
    test_cases = [
        {
            'name': 'Valid single test game',
            'args': ['--test-game', 'stag-hunt', '--training-games', 'prisoners-dilemma', '--max-epochs', '1'],
            'should_succeed': True
        },
        {
            'name': 'Valid test game with multiple training games',
            'args': ['--test-game', 'battle-of-sexes', '--training-games', 'prisoners-dilemma,hawk-dove', '--max-epochs', '1'],
            'should_succeed': True
        },
        {
            'name': 'Invalid test game',
            'args': ['--test-game', 'invalid-game', '--training-games', 'prisoners-dilemma', '--max-epochs', '1'],
            'should_succeed': False
        },
        {
            'name': 'Invalid training game',
            'args': ['--test-game', 'stag-hunt', '--training-games', 'invalid-game', '--max-epochs', '1'],
            'should_succeed': False
        },
        {
            'name': 'Test game with extra spaces (should work now)',
            'args': ['--test-game', ' stag-hunt ', '--training-games', 'prisoners-dilemma', '--max-epochs', '1'],
            'should_succeed': True
        },
    ]
    
    python_executable = sys.executable
    script_path = 'main_experiment.py'
    
    results = []
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        print(f"   Command: python {script_path} {' '.join(test_case['args'])}")
        
        try:
            # Run the script with the test arguments, but add --dry-run equivalent
            cmd = [python_executable, script_path] + test_case['args'] + ['--verbose']
            
            # Capture both stdout and stderr
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=10,  # 10 second timeout
                cwd=os.path.dirname(__file__) or '.'
            )
            
            success = result.returncode == 0
            
            if test_case['should_succeed']:
                if success:
                    print(f"   ✅ PASS: Arguments accepted successfully")
                    results.append(True)
                else:
                    print(f"   ❌ FAIL: Expected success but got error:")
                    print(f"      stderr: {result.stderr[:200]}...")
                    results.append(False)
            else:
                if not success:
                    print(f"   ✅ PASS: Arguments correctly rejected")
                    results.append(True)
                else:
                    print(f"   ❌ FAIL: Expected failure but arguments were accepted")
                    results.append(False)
                    
        except subprocess.TimeoutExpired:
            print(f"   ⏰ TIMEOUT: Script took too long (probably started training)")
            if test_case['should_succeed']:
                print(f"   ✅ PASS: Arguments were accepted (script started)")
                results.append(True)
            else:
                print(f"   ❌ FAIL: Expected rejection but script started")
                results.append(False)
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} tests passed!")
        print("   The argument parsing bug has been fixed.")
        return True
    else:
        print(f"❌ {passed}/{total} tests passed.")
        print("   Some issues remain with argument parsing.")
        return False

def test_specific_bug():
    """Test the specific bug mentioned in the user request."""
    print("\n🎯 Testing the specific bug case...")
    print("=" * 50)
    
    # The original error was: --test-game 'stag-hunt, battle-of-sexes'
    # This should now give a clear error message instead of an argparse error
    
    python_executable = sys.executable
    script_path = 'main_experiment.py'
    
    # Test the problematic case
    cmd = [python_executable, script_path, '--test-game', 'stag-hunt, battle-of-sexes', '--max-epochs', '1']
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=5,
            cwd=os.path.dirname(__file__) or '.'
        )
        
        if result.returncode != 0:
            error_output = result.stderr.strip()
            if "Invalid test game" in error_output:
                print("✅ PASS: Clear error message for invalid test game")
                print(f"   Error: {error_output}")
                return True
            elif "argparse.ArgumentError" in error_output:
                print("❌ FAIL: Still getting argparse error")
                print(f"   Error: {error_output}")
                return False
            else:
                print("❓ Different error occurred:")
                print(f"   Error: {error_output}")
                return False
        else:
            print("❌ FAIL: Invalid argument was accepted")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Script started (probably accepted arguments)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("Argument Parsing Bug Fix Verification")
    print("=" * 60)
    
    # Test general argument parsing
    test1_result = test_argument_parsing()
    
    # Test specific bug case
    test2_result = test_specific_bug()
    
    print("\n" + "=" * 60)
    if test1_result and test2_result:
        print("🎉 Bug fix verified! The argument parsing error has been resolved.")
        print("   You can now use valid single game names for --test-game")
        print("   Invalid games will show clear error messages instead of argparse errors")
    else:
        print("❌ Some issues remain. Please check the test results above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())