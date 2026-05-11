"""
Verify data completeness for current experiments (May 2026, input_size=9 architecture)
"""
from pathlib import Path
import json

def verify_generalization_matrix():
    """Verify generalization matrix (task-opponent setup) data"""
    print("=" * 70)
    print("VERIFYING GENERALIZATION MATRIX (Task-Opponent Setup)")
    print("=" * 70)
    
    train_dir = Path(r'c:\Users\User\OneDrive - huji.ac.il\מסמכים\Max_Planck\Cognitive-Therapy-fo-AI-\experiments\generalization_matrix_train_913243\training')
    test_dir = Path(r'c:\Users\User\OneDrive - huji.ac.il\מסמכים\Max_Planck\Cognitive-Therapy-fo-AI-\experiments\generalization_matrix_test_913245\testing')
    
    # Expected: 75 training models (3 games × 5 opponents × 5 seeds)
    expected_train = 15 * 5  # 15 conditions × 5 seeds
    
    # Count training directories
    train_conditions = list(train_dir.glob("condition_*_seed_*"))
    print(f"\nTraining Models:")
    print(f"  Expected: {expected_train} models")
    print(f"  Found: {len(train_conditions)} condition directories")
    
    # Verify each has experiment data
    missing_data = []
    for cond_dir in train_conditions:
        task_dirs = list(cond_dir.glob("generalization_matrix_task_*"))
        if not task_dirs:
            missing_data.append(str(cond_dir))
        else:
            # Check for checkpoint and logs
            task_dir = task_dirs[0]
            checkpoint_dir = task_dir / "checkpoints"
            if not checkpoint_dir.exists() or not list(checkpoint_dir.glob("*.pth")):
                missing_data.append(f"{cond_dir} - missing checkpoints")
    
    if missing_data:
        print(f"  ⚠️ ISSUES FOUND:")
        for issue in missing_data[:5]:
            print(f"    - {issue}")
    else:
        print(f"  ✅ All training data verified")
    
    # Count test results
    if test_dir.exists():
        test_files = list(test_dir.glob("**/*.csv"))
        print(f"\nTest Results:")
        print(f"  Expected: 1,050 test results (75 models × 14 conditions)")
        print(f"  Found: {len(test_files)} CSV files")
        print(f"  Status: {'✅' if len(test_files) > 1000 else '⚠️'}")
    else:
        print(f"\n⚠️ Test directory not found: {test_dir}")
    
    return len(train_conditions) == expected_train

def verify_whole_population():
    """Verify whole population (task setup) data"""
    print("\n" + "=" * 70)
    print("VERIFYING WHOLE POPULATION (Task Setup)")
    print("=" * 70)
    
    train_dir = Path(r'c:\Users\User\OneDrive - huji.ac.il\מסמכים\Max_Planck\Cognitive-Therapy-fo-AI-\experiments\whole_population_train_913310\training')
    test_dir = Path(r'c:\Users\User\OneDrive - huji.ac.il\מסמכים\Max_Planck\Cognitive-Therapy-fo-AI-\experiments\whole_population_test_912631\testing')
    
    # Expected: 15 training models (3 games × 5 seeds)
    expected_train = 15
    
    # Count training directories
    train_tasks = list(train_dir.glob("whole_population_task_*"))
    print(f"\nTraining Models:")
    print(f"  Expected: {expected_train} models")
    print(f"  Found: {len(train_tasks)} task directories")
    
    # Verify each has experiment data
    missing_data = []
    for task_dir in train_tasks:
        checkpoint_dir = task_dir / "checkpoints"
        if not checkpoint_dir.exists() or not list(checkpoint_dir.glob("*.pth")):
            missing_data.append(f"{task_dir.name} - missing checkpoints")
        
        # Check config
        config_file = task_dir / "experiment_config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                input_size = config['network_config']['input_size']
                if input_size != 9:
                    missing_data.append(f"{task_dir.name} - wrong input_size: {input_size}")
    
    if missing_data:
        print(f"  ⚠️ ISSUES FOUND:")
        for issue in missing_data[:5]:
            print(f"    - {issue}")
    else:
        print(f"  ✅ All training data verified")
    
    # Count test results
    if test_dir.exists():
        test_files = list(test_dir.glob("**/*.csv"))
        print(f"\nTest Results:")
        print(f"  Expected: 225 test results (15 models × 15 conditions)")
        print(f"  Found: {len(test_files)} CSV files")
        print(f"  Status: {'✅' if len(test_files) > 200 else '⚠️'}")
    else:
        print(f"\n⚠️ Test directory not found: {test_dir}")
    
    return len(train_tasks) == expected_train

if __name__ == "__main__":
    gen_ok = verify_generalization_matrix()
    pop_ok = verify_whole_population()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Generalization Matrix: {'✅ PASS' if gen_ok else '⚠️ ISSUES'}")
    print(f"Whole Population: {'✅ PASS' if pop_ok else '⚠️ ISSUES'}")
    print("=" * 70)
