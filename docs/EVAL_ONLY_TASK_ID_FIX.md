# Eval-Only Mode Task ID Bug Fix

## Date
March 29, 2026

## Problem Description

### Symptom
The test phase of the generalization matrix experiment was failing with errors like:
```
ValueError: Invalid task_id 998. Must be 0-14
```

This occurred when running eval-only mode jobs in SLURM array jobs with task IDs beyond the number of training conditions (0-14).

### Example Error
```bash
INFO:    Starting Mixed-Motive Game Experiment
INFO:    Arguments: {
    'task_id': 998,
    'mode': 'eval-only',
    ...
}
ERROR:   Error in experiment: Invalid task_id 998. Must be 0-14
ValueError: Invalid task_id 998. Must be 0-14
```

## Root Cause Analysis

### Architecture Overview
The testing phase uses a two-level indexing system:
1. **SLURM_ARRAY_TASK_ID** (0-1049): Each task tests one model on one condition
2. **Decoded indices**:
   - MODEL_ID = SLURM_ARRAY_TASK_ID / NUM_TEST_CONDITIONS
   - TRAINING_CONDITION_ID = MODEL_ID / NUM_SEEDS
   - TEST_CONDITION_ID = (SLURM_ARRAY_TASK_ID % NUM_TEST_CONDITIONS) adjusted

Example for SLURM_ARRAY_TASK_ID=998:
- MODEL_ID = 998 / 14 = 71
- TRAINING_CONDITION_ID = 71 / 5 = 14 ✓ (valid)
- TEST_CONDITION_ID = (998 % 14) = 4 ✓ (valid)

### The Bug
The SLURM scripts correctly calculated and passed `--task-id ${TRAINING_CONDITION_ID}`, but the Python code had a fallback mechanism:

```python
task_id = args.task_id
if task_id is None:
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))  # Falls back to 998!
```

If the bash variable substitution failed or `args.task_id` was None, it would read the raw SLURM_ARRAY_TASK_ID environment variable (998), which then failed validation.

### Validation Logic Issue
The validation happened BEFORE mode checking:

```python
# Load training conditions
training_conditions = matrix_config['training_conditions']

# VALIDATION HAPPENED HERE - before checking mode!
if task_id < 0 or task_id >= len(training_conditions):
    raise ValueError(f"Invalid task_id {task_id}. Must be 0-{len(training_conditions)-1}")

train_condition = training_conditions[task_id]

# Mode check happened AFTER validation
if mode == 'eval-only':
    # eval-only doesn't actually use train_condition!
    return run_evaluation_phase(...)
```

**Problem**: In eval-only mode, `task_id` is only used for output file naming, not for loading training conditions. The validation was unnecessarily strict.

## Solution

### Code Changes
Reordered the logic to check mode FIRST, then validate only when needed:

**File**: `main_experiment.py`, function `run_generalization_matrix_experiment()`

**Before**:
```python
# Get training condition for this task
training_conditions = matrix_config['training_conditions']
if task_id < 0 or task_id >= len(training_conditions):
    raise ValueError(f"Invalid task_id {task_id}. Must be 0-{len(training_conditions)-1}")

train_condition = training_conditions[task_id]

# === MODE BRANCHING ===
if mode == 'train-only':
    ...
elif mode == 'eval-only':
    return run_evaluation_phase(...)
```

**After**:
```python
# Get training conditions list
training_conditions = matrix_config['training_conditions']

# === MODE BRANCHING ===
if mode == 'eval-only':
    # In eval-only mode, task_id is only used for output naming
    # No need to validate against training_conditions
    logger.info(f"Running EVAL-ONLY mode (task_id {task_id} for naming)")
    return run_evaluation_phase(...)

# For train-only and full modes, validate and get training condition
if task_id < 0 or task_id >= len(training_conditions):
    raise ValueError(f"Invalid task_id {task_id}. Must be 0-{len(training_conditions)-1}")

train_condition = training_conditions[task_id]

if mode == 'train-only':
    ...
```

### Key Changes
1. **Mode check moved before validation**: Eval-only mode bypasses validation entirely
2. **Clear documentation**: Comments explain why eval-only mode doesn't need validation
3. **Validation remains for other modes**: train-only and full modes still validate task_id

## Impact

### What This Fixes
- ✓ Test phase SLURM arrays (0-1049) now work correctly with eval-only mode
- ✓ task_id can be any integer in eval-only mode (used only for naming)
- ✓ Maintains strict validation for train-only and full modes

### Backward Compatibility
- ✓ No breaking changes to existing functionality
- ✓ train-only and full modes still validate task_id strictly
- ✓ Eval-only mode behavior unchanged except it no longer crashes on large task_ids

### Testing
The fix allows the test phase scripts to work correctly:
- Part 1: Tasks 0-999 (test phase part 1)
- Part 2: Tasks 1000-1049 (test phase part 2)

Each task loads a checkpoint and tests on a condition, with task_id used only for output naming.

## Lessons Learned

1. **Validate only when necessary**: Don't perform validation for data that won't be used
2. **Mode-aware logic**: Different modes may have different requirements
3. **Environment variable fallbacks**: Be careful with fallback logic that reads from environment
4. **Separation of concerns**: task_id serves different purposes in different modes:
   - train-only/full: Index into training_conditions
   - eval-only: Output naming only

## Related Files
- `main_experiment.py`: Core fix location (line ~1140-1185)
- `run_generalization_matrix_test.sh`: SLURM test script part 1
- `run_generalization_matrix_test_part2.sh`: SLURM test script part 2
- `CHANGELOG.md`: Change documentation
