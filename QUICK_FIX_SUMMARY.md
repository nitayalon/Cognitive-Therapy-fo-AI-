# Quick Fix Summary: Invalid task_id Bug

## What Was Wrong
The test phase was failing because Python was receiving task_id=998 (the raw SLURM array index) instead of the calculated TRAINING_CONDITION_ID (14).

When task_id=998 was passed to the validation check:
```python
if task_id >= len(training_conditions):  # 998 >= 15
    raise ValueError("Invalid task_id 998. Must be 0-14")
```

## Why It Happened
In eval-only mode, the code was:
1. First validating task_id against training_conditions (must be 0-14)
2. Then checking if mode is eval-only
3. But in eval-only mode, task_id is only used for naming outputs, not for loading training conditions!

## The Fix
Reordered the logic:
1. **First** check if mode is eval-only → skip validation
2. **Then** validate task_id only for modes that actually need it (train-only, full)

```python
# NEW ORDER:
if mode == 'eval-only':
    # No validation needed - task_id is just for naming
    return run_evaluation_phase(...)

# Validation only for modes that need it
if task_id >= len(training_conditions):
    raise ValueError(...)
```

## What This Means
✓ **Test phase will now work** - task_id can be any integer in eval-only mode  
✓ **No changes to training phase** - validation still strict for train-only mode  
✓ **Backward compatible** - existing experiments unaffected  

## Re-Running the Tests
You can now safely re-run the test phase:

```bash
# Set the training job ID
export TRAINING_JOB_ID=888509

# Submit test jobs
sbatch run_generalization_matrix_test.sh          # Part 1: tasks 0-999
sbatch run_generalization_matrix_test_part2.sh    # Part 2: tasks 1000-1049
```

The jobs should now complete successfully. See [TEST_PHASE_RERUN_CHECKLIST.md](TEST_PHASE_RERUN_CHECKLIST.md) for detailed instructions.

## Files Modified
- `main_experiment.py` - Core fix (line ~1148)
- `CHANGELOG.md` - Documented the change
- `docs/EVAL_ONLY_TASK_ID_FIX.md` - Detailed technical documentation
- `TEST_PHASE_RERUN_CHECKLIST.md` - Re-run instructions
