# Test Phase Re-Run Checklist

## Fix Applied
✓ **Eval-only mode task_id validation removed** (2026-03-29)
  - File: `main_experiment.py` line ~1148
  - Change: Moved task_id validation after mode check
  - Impact: eval-only mode can now accept any task_id value

## Pre-Flight Checks

### 1. Verify Training Phase Completed
```bash
# Check that training job completed successfully
ls experiments/generalization_matrix_train_888509/training/
# Should show: condition_0_seed_0 through condition_14_seed_4 (75 directories)

# Quick checkpoint verification
find experiments/generalization_matrix_train_888509/training -name "*_final_checkpoint.pth" | wc -l
# Should show: 75 checkpoint files
```

### 2. Verify Test Scripts
```bash
# Check test scripts exist
ls -lh run_generalization_matrix_test.sh
ls -lh run_generalization_matrix_test_part2.sh

# Verify they reference correct training job
grep "TRAINING_JOB_ID" run_generalization_matrix_test*.sh
```

### 3. Environment Setup
```bash
# Ensure SLURM logs directory exists
mkdir -p slurm_logs

# Verify config file exists
ls -lh config/generalization_matrix_config.json
```

## Running Test Phase

### Part 1: Tasks 0-999
```bash
# Set training job ID
export TRAINING_JOB_ID=888509

# Submit part 1
sbatch run_generalization_matrix_test.sh

# Expected output:
# "Submitted batch job XXXXXX"
```

### Part 2: Tasks 1000-1049
```bash
# Set training job ID (if not already exported)
export TRAINING_JOB_ID=888509

# Submit part 2
sbatch run_generalization_matrix_test_part2.sh

# Expected output:
# "Submitted batch job YYYYYY"
```

## Monitoring

### Check Job Status
```bash
# View queued/running jobs
squeue -u $USER

# Monitor specific job
squeue -j XXXXXX

# Check recent outputs
tail -f slurm_logs/test_gen_matrix_test_p1_XXXXXX_*.out
tail -f slurm_logs/test_gen_matrix_test_p1_XXXXXX_*.err
```

### Expected Behavior
With the fix, you should see:
```
INFO:    Running generalization-matrix experiment (mode: eval-only, task 998)
INFO:    Running EVAL-ONLY mode (task_id 998 for naming)
INFO:    === EVALUATION PHASE: Testing Task 998 ===
```

**No more**: `ValueError: Invalid task_id 998. Must be 0-14`

### Success Indicators
```bash
# All tasks should complete successfully (exit status 0)
# Check for errors in logs
grep -i "error\|failed\|invalid" slurm_logs/test_*.err | grep -v "module: command not found"

# Count successful completions
grep "Testing task.*completed (exit: 0)" slurm_logs/test_*.out | wc -l
# Should show: 1050 (1000 from part1 + 50 from part2)

# Verify output files created
find experiments/generalization_matrix_test_*/testing -name "*.csv" | wc -l
# Should show: 1050 CSV files (one per test task)
```

## Post-Run Validation

### Check Output Structure
```bash
# View test output directory
ls -lh experiments/generalization_matrix_test_*/testing/

# Should contain directories like:
# model_0_test_cond_1, model_0_test_cond_2, ..., model_74_test_cond_13, model_74_test_cond_14
```

### Verify CSV Files
```bash
# Check a sample CSV
head experiments/generalization_matrix_test_*/testing/model_0_test_cond_1/results/test_*.csv

# Should show columns: opponent_id, cooperation_rate, avg_reward, etc.
```

### Analyze Results
```bash
# Use existing analysis scripts
python experiments/analysis_scripts/analyze_generalization_matrix.py \
    --test-dir experiments/generalization_matrix_test_XXXXXX
```

## Troubleshooting

### If Jobs Still Fail

1. **Check task_id being passed**:
   ```bash
   grep "task_id" slurm_logs/test_*.out | head
   ```

2. **Verify checkpoint paths**:
   ```bash
   grep "Checkpoint:" slurm_logs/test_*.out | head
   ```

3. **Check Python errors**:
   ```bash
   grep -A 5 "Traceback" slurm_logs/test_*.err
   ```

4. **Validate fix is deployed**:
   ```bash
   # Check main_experiment.py has the fix
   grep -A 3 "if mode == 'eval-only':" main_experiment.py | head -n 10
   # Should show mode check BEFORE task_id validation
   ```

## Expected Timeline

- **Part 1** (1000 tasks, 100 concurrent): ~6-10 hours
- **Part 2** (50 tasks, 50 concurrent): ~0.5-1 hour
- **Total**: ~6-11 hours (depending on cluster load)

## Summary

✓ Fix applied: eval-only mode now accepts arbitrary task_id values
✓ Training phase completed: 75 models ready for testing
✓ Test scripts ready: Both parts configured for job 888509
✓ Expected output: 1050 test results (75 models × 14 test conditions each)

**Status**: Ready to re-run test phase
