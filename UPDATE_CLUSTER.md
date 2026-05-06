# Update Cluster and Run Whole Population Experiment

## Step 1: SSH to Cluster and Update Code

```bash
# SSH to cluster
ssh your_cluster_address

# Navigate to project directory
cd /path/to/Cognitive-Therapy-fo-AI-

# Pull latest code
git pull origin main

# Verify the update
grep "whole-population" main_experiment.py
# Should show: choices=['basic', 'multi-game', 'segmented', 'generalization-matrix', 'whole-population']
```

## Step 2: Test the Code

```bash
# Quick test to verify --experiment-mode accepts whole-population
python main_experiment.py --experiment-mode whole-population --help

# Should show help without error
```

## Step 3: Submit Jobs

### Option A: One-liner with automatic dependency
```bash
TRAIN_JOB=$(sbatch --parsable run_whole_population_train.sh) && echo "Training: $TRAIN_JOB" && sbatch --dependency=afterok:$TRAIN_JOB --export=TRAINING_JOB_ID=$TRAIN_JOB run_whole_population_test.sh
```

### Option B: Manual (if Option A doesn't work)
```bash
# Submit training
sbatch run_whole_population_train.sh

# Wait for training to complete and note the JOB_ID (e.g., 902235)
# Then submit testing:
TRAINING_JOB_ID=902235 sbatch run_whole_population_test.sh
```

## Step 4: Monitor Jobs

```bash
# Check training status
squeue -u $USER | grep wp_train

# Check specific output
tail -f slurm_logs/wp_train_wp_train_*_0.out

# After training completes, check testing
squeue -u $USER | grep wp_test
```

## Troubleshooting

### If git pull shows conflicts:
```bash
# Stash any local changes
git stash

# Pull again
git pull origin main

# Check if code is updated
git log --oneline -5
# Should show commit: "Implement whole population training experiment"
```

### If module command not found (can ignore these warnings):
The warnings about "module: command not found" can be ignored - they don't affect execution.

### Verify PYTHONPATH:
```bash
echo $PYTHONPATH
# Should include: /your/path/Cognitive-Therapy-fo-AI-/src
```
