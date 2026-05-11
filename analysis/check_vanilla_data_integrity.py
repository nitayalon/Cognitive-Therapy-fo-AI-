"""
Comprehensive data integrity check for vanilla RL experiments.

Verifies that all 16 tasks have complete data:
- Expected directory structure
- Key result files
- Evaluation data
- Training logs
"""

import os
import pickle
import json
from pathlib import Path
from collections import defaultdict

# Base experiments directory
EXPERIMENTS_DIR = Path(__file__).parent / "experiments"

# Map of vanilla RL array directories
VANILLA_DIRS = [
    "vanilla_rl_array_835338_20260223_065612",  # tasks 0-7
    "vanilla_rl_array_835338_20260223_152600",  # task 8
    "vanilla_rl_array_835338_20260223_152654",  # task 9
    "vanilla_rl_array_835338_20260223_152715",  # task 10
    "vanilla_rl_array_835338_20260223_153050",  # task 11
    "vanilla_rl_array_835338_20260223_153053",  # task 12
    "vanilla_rl_array_835338_20260223_153234",  # task 13
    "vanilla_rl_array_835338_20260223_153306",  # task 14
    "vanilla_rl_array_835338_20260223_153933",  # task 15
]

def find_all_vanilla_tasks():
    """Find all vanilla RL task directories."""
    task_paths = {}
    for dir_name in VANILLA_DIRS:
        dir_path = EXPERIMENTS_DIR / dir_name
        if not dir_path.exists():
            continue
        
        # Find all task directories
        for task_dir in dir_path.glob("vanilla_matrix_task*"):
            task_num = int(task_dir.name.replace("vanilla_matrix_task", ""))
            task_paths[task_num] = task_dir
    
    return task_paths

def check_directory_structure(task_path):
    """Check if task has expected directory structure."""
    required_dirs = ["checkpoints", "logs", "plots", "results"]
    issues = []
    
    for req_dir in required_dirs:
        dir_path = task_path / req_dir
        if not dir_path.exists():
            issues.append(f"Missing directory: {req_dir}")
    
    return issues

def check_result_files(task_path):
    """Check for essential result files."""
    issues = []
    
    # Check for experiment config
    config_file = task_path / "experiment_config.json"
    if not config_file.exists():
        issues.append("Missing experiment_config.json")
    else:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            issues.append(f"Cannot read experiment_config.json: {e}")
    
    # Check for matrix results
    results_file = task_path / "results" / "matrix_results.pkl"
    if not results_file.exists():
        issues.append("Missing results/matrix_results.pkl")
    else:
        try:
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
        except Exception as e:
            issues.append(f"Cannot read matrix_results.pkl: {e}")
    
    # Check for at least one checkpoint
    checkpoint_dir = task_path / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if len(checkpoints) == 0:
            issues.append("No checkpoint files found")
    
    return issues

def check_evaluation_logs(task_path):
    """Check for evaluation log directories."""
    logs_dir = task_path / "logs"
    issues = []
    eval_dirs = []
    
    if not logs_dir.exists():
        return ["Logs directory missing"], []
    
    # Find all eval_* directories
    for eval_dir in logs_dir.glob("eval_*"):
        eval_dirs.append(eval_dir.name)
    
    # Expected minimum evaluation directories:
    # - eval_baseline
    # - eval_same_game_* (3 different opponent ranges)
    # - eval_*_same_opponents (3 different games)
    
    if "eval_baseline" not in eval_dirs:
        issues.append("Missing eval_baseline directory")
    
    if len(eval_dirs) < 4:  # baseline + at least 3 generalization tests
        issues.append(f"Only {len(eval_dirs)} evaluation directories (expected at least 4)")
    
    return issues, eval_dirs

def analyze_training_completion(task_path):
    """Check if training completed successfully."""
    issues = []
    
    # Check experiment log
    log_file = task_path / "logs" / "experiment.log"
    if not log_file.exists():
        issues.append("Missing experiment.log")
        return issues, None
    
    # Read log to check for completion
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        if "Training completed" not in log_content and "Converged" not in log_content:
            issues.append("Training may not have completed successfully")
        
        # Extract epochs completed
        epochs_completed = None
        for line in log_content.split('\n'):
            if "Epoch" in line and "/" in line:
                try:
                    # Look for patterns like "Epoch 123/500"
                    parts = line.split("Epoch")[1].split("/")
                    current_epoch = int(parts[0].strip())
                    epochs_completed = current_epoch
                except:
                    pass
        
        return issues, epochs_completed
    except Exception as e:
        issues.append(f"Cannot read experiment.log: {e}")
        return issues, None

def main():
    print("=" * 80)
    print("VANILLA RL DATA INTEGRITY CHECK")
    print("=" * 80)
    print()
    
    # Find all tasks
    task_paths = find_all_vanilla_tasks()
    print(f"Found {len(task_paths)} task directories")
    print()
    
    # Check for missing tasks
    expected_tasks = set(range(16))
    found_tasks = set(task_paths.keys())
    missing_tasks = expected_tasks - found_tasks
    
    if missing_tasks:
        print("❌ MISSING TASKS:")
        for task_num in sorted(missing_tasks):
            print(f"   Task {task_num}")
        print()
    else:
        print("✅ All 16 tasks found")
        print()
    
    # Detailed check for each task
    print("=" * 80)
    print("DETAILED TASK ANALYSIS")
    print("=" * 80)
    print()
    
    all_issues = defaultdict(list)
    task_summary = []
    
    for task_num in sorted(task_paths.keys()):
        task_path = task_paths[task_num]
        print(f"Task {task_num}: {task_path.name}")
        print("-" * 80)
        
        task_issues = []
        
        # Check directory structure
        dir_issues = check_directory_structure(task_path)
        task_issues.extend(dir_issues)
        
        # Check result files
        file_issues = check_result_files(task_path)
        task_issues.extend(file_issues)
        
        # Check evaluation logs
        eval_issues, eval_dirs = check_evaluation_logs(task_path)
        task_issues.extend(eval_issues)
        
        # Check training completion
        training_issues, epochs = analyze_training_completion(task_path)
        task_issues.extend(training_issues)
        
        # Report
        if task_issues:
            print(f"   ⚠️  Issues found: {len(task_issues)}")
            for issue in task_issues:
                print(f"      - {issue}")
            all_issues[task_num] = task_issues
        else:
            print(f"   ✅ No issues")
        
        print(f"   Evaluation directories: {len(eval_dirs)}")
        if epochs:
            print(f"   Training epochs completed: {epochs}")
        
        task_summary.append({
            'task': task_num,
            'path': str(task_path),
            'issues': len(task_issues),
            'eval_dirs': len(eval_dirs),
            'epochs': epochs
        })
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    total_tasks = len(task_paths)
    tasks_with_issues = len(all_issues)
    tasks_ok = total_tasks - tasks_with_issues
    
    print(f"Total tasks found: {total_tasks}/16")
    print(f"Tasks with no issues: {tasks_ok}")
    print(f"Tasks with issues: {tasks_with_issues}")
    print()
    
    if missing_tasks:
        print(f"❌ CRITICAL: {len(missing_tasks)} tasks are missing")
        print()
    
    if tasks_with_issues > 0:
        print("Tasks requiring attention:")
        for task_num, issues in sorted(all_issues.items()):
            print(f"   Task {task_num}: {len(issues)} issue(s)")
        print()
    
    # Data completeness percentage
    completeness = (tasks_ok / 16) * 100 if total_tasks > 0 else 0
    print(f"Data completeness: {completeness:.1f}%")
    print()
    
    if completeness == 100.0:
        print("✅ ALL DATA VERIFIED - Ready for analysis")
    elif completeness >= 75.0:
        print("⚠️  MOSTLY COMPLETE - Analysis possible with caveats")
    else:
        print("❌ INCOMPLETE DATA - Review issues before analysis")
    
    print()
    print("=" * 80)
    
    return task_summary, all_issues

if __name__ == "__main__":
    task_summary, all_issues = main()
