# FSM Training - Split Job Workflow

## Problem
Training 10,000 episodes timed out on 8-hour limit for larger networks (H=16, 32).

## Solution
Split training into 3 parallel job arrays by network size, each with appropriate time limits:

| Job | Networks | Tasks | Time Limit | Memory |
|-----|----------|-------|------------|--------|
| **Small** | H=2,4 | 180 | 4 hours | 8G |
| **Medium** | H=8 | 120 | 8 hours | 8G |
| **Large** | H=16,32 | 120 | 12 hours | 16G |
| **Total** | | **420** | | |

## Task Breakdown

### Small Networks (180 tasks)
- **no_game H=2**: 60 tasks (3 games × 2 opponents × 10 seeds)
- **no_game H=4**: 60 tasks
- **game_tag H=4**: 60 tasks

### Medium Networks (120 tasks)
- **no_game H=8**: 60 tasks (3 games × 2 opponents × 10 seeds)
- **game_tag H=8**: 60 tasks

### Large Networks (120 tasks)
- **no_game H=16**: 60 tasks (3 games × 2 opponents × 10 seeds)
- **no_game H=32**: 60 tasks

## Workflow

### Step 1: Submit All Training Jobs

```bash
cd ~/Cognitive-Therapy-fo-AI-
git pull origin FSM_representation

# Submit all three jobs at once
bash submit_all_fsm_training.sh

# This will output job IDs like:
# Small (H=2,4):   943500
# Medium (H=8):    943501
# Large (H=16,32): 943502

# Save these for later:
export SMALL_JOB=943500
export MEDIUM_JOB=943501
export LARGE_JOB=943502
```

### Step 2: Monitor Progress

```bash
# Check all jobs
squeue -j ${SMALL_JOB},${MEDIUM_JOB},${LARGE_JOB}

# Count completed checkpoints
find experiments/fsm_train_small_${SMALL_JOB} -name 'checkpoint.pth' | wc -l   # Target: 180
find experiments/fsm_train_medium_${MEDIUM_JOB} -name 'checkpoint.pth' | wc -l # Target: 120
find experiments/fsm_train_large_${LARGE_JOB} -name 'checkpoint.pth' | wc -l   # Target: 120

# Check for errors
tail slurm_logs/fsm_train_small_*_{$SMALL_JOB}_*.err
tail slurm_logs/fsm_train_medium_*_{$MEDIUM_JOB}_*.err
tail slurm_logs/fsm_train_large_*_{$LARGE_JOB}_*.err
```

### Step 3: Aggregate Checkpoints (Optional)

After all jobs complete, optionally create a unified directory:

```bash
bash aggregate_checkpoints.sh
# Creates: experiments/fsm_train_aggregated/
```

### Step 4: Submit Test Jobs

The test script will automatically find checkpoints across all three directories:

```bash
# Update TRAINING_DIRS environment variable
export TRAINING_DIRS="experiments/fsm_train_small_${SMALL_JOB},experiments/fsm_train_medium_${MEDIUM_JOB},experiments/fsm_train_large_${LARGE_JOB}"

# Submit test jobs
TRAINING_DIRS="${TRAINING_DIRS}" bash submit_fsm_test_jobs.sh
```

## Expected Timelines

- **Small networks**: ~2-3 hours
- **Medium networks**: ~5-6 hours
- **Large networks**: ~8-10 hours
- **All jobs run in parallel**, so total wall time = longest job (~10 hours)

## File Structure

```
experiments/
├── fsm_train_small_<JOB_ID>/
│   └── task_0/ ... task_179/
│       └── checkpoint.pth
├── fsm_train_medium_<JOB_ID>/
│   └── task_0/ ... task_119/
│       └── checkpoint.pth
├── fsm_train_large_<JOB_ID>/
│   └── task_0/ ... task_119/
│       └── checkpoint.pth
└── fsm_train_aggregated/  (optional)
    └── task_0/ ... task_419/
        └── checkpoint.pth
```

## Advantages

✅ **Parallel execution**: All network sizes train simultaneously
✅ **Optimized time limits**: No wasted compute time
✅ **Resource optimization**: Large networks get more memory
✅ **Fault tolerance**: One failing size doesn't block others
✅ **Easy monitoring**: Separate logs per size category

## New Scripts

- `run_fsm_train_small.sh` - Small network training (H=2,4)
- `run_fsm_train_medium.sh` - Medium network training (H=8)
- `run_fsm_train_large.sh` - Large network training (H=16,32)
- `submit_all_fsm_training.sh` - Submit all three jobs
- `aggregate_checkpoints.sh` - Merge results (optional)
