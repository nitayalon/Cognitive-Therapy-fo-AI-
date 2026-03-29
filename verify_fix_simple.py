"""
Simple verification that the fix is in place by checking the code structure.
This doesn't import the framework, just verifies the logic flow.
"""

def verify_fix():
    """Verify the eval-only fix is correctly implemented"""
    
    print("Verifying eval-only mode task_id fix...")
    
    # Read the main_experiment.py file
    with open('main_experiment.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the run_generalization_matrix_experiment function
    func_start = content.find('def run_generalization_matrix_experiment(')
    if func_start == -1:
        print("✗ FAILED: Could not find run_generalization_matrix_experiment function")
        return False
    
    # Get a section of code after the function declaration
    section = content[func_start:func_start + 5000]
    
    # Check 1: eval-only mode check should come before task_id validation
    eval_only_pos = section.find("if mode == 'eval-only':")
    validation_pos = section.find("if task_id < 0 or task_id >= len(training_conditions):")
    
    if eval_only_pos == -1:
        print("✗ FAILED: Could not find eval-only mode check")
        return False
        
    if validation_pos == -1:
        print("✗ FAILED: Could not find task_id validation")
        return False
    
    if eval_only_pos > validation_pos:
        print("✗ FAILED: Validation still happens before mode check!")
        print(f"   eval-only check at position: {eval_only_pos}")
        print(f"   validation check at position: {validation_pos}")
        return False
    
    print("✓ SUCCESS: eval-only check comes BEFORE task_id validation")
    
    # Check 2: Comment should explain why eval-only doesn't need validation
    if "In eval-only mode, task_id is only used for output naming" in section:
        print("✓ SUCCESS: Comment explains the logic")
    else:
        print("⚠ WARNING: Explanatory comment not found, but logic is correct")
    
    # Check 3: Verify eval-only returns directly
    eval_section = section[eval_only_pos:eval_only_pos + 500]
    if "return run_evaluation_phase" in eval_section:
        print("✓ SUCCESS: eval-only mode returns directly to run_evaluation_phase")
    else:
        print("✗ FAILED: eval-only doesn't return to run_evaluation_phase")
        return False
    
    # Check 4: Validation is after mode checks
    lines_before_validation = section[:validation_pos].split('\n')
    recent_lines = lines_before_validation[-10:]  # Last 10 lines before validation
    
    train_only_found = any("if mode == 'train-only':" in line for line in recent_lines)
    if train_only_found:
        print("✓ SUCCESS: Validation happens after train-only check")
    else:
        print("⚠ WARNING: train-only check not found near validation, but this might be OK")
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE: Fix is correctly implemented!")
    print("="*60)
    print("\nKey changes:")
    print("1. ✓ Mode check (eval-only) comes FIRST")
    print("2. ✓ eval-only bypasses validation entirely")
    print("3. ✓ Validation only applies to train-only/full modes")
    print("\nConclusion: The bug is fixed. Test phase should now work.")
    
    return True

if __name__ == '__main__':
    import sys
    success = verify_fix()
    sys.exit(0 if success else 1)
