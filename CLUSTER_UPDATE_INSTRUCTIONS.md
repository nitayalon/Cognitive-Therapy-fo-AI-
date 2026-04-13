# UPDATE CLUSTER CODE - IMPORTANT

## Problem
The cluster is running an **old version** of the code that doesn't include the whole-population experiment mode.

## Solution
You need to pull the latest code on the cluster. Run these commands **on the cluster**:

```bash
# SSH to the cluster (if not already there)
ssh your_cluster_name

# Navigate to your project directory
cd /path/to/Cognitive-Therapy-fo-AI-

# Pull the latest code
git pull origin main

# Verify the update worked
python test_whole_population_setup.py
```

## Quick Verification
After pulling, verify the experiment mode is available:

```bash
python main_experiment.py --experiment-mode whole-population --help
```

This should show help text WITHOUT an error.

## If Git Pull Doesn't Work

If you get merge conflicts or other git issues:

```bash
# Stash any local changes
git stash

# Pull latest
git pull origin main

# Reapply your changes if needed
git stash pop
```

## After Updating

Once verification passes, you can run the experiment:

```bash
TRAIN_JOB=$(sbatch --parsable run_whole_population_train.sh) && echo "Training: $TRAIN_JOB" && sbatch --dependency=afterok:$TRAIN_JOB --export=TRAINING_JOB_ID=$TRAIN_JOB run_whole_population_test.sh
```

## Troubleshooting

**Still getting the error after git pull?**
- Check you're in the correct directory: `pwd`
- Check git status: `git status`
- Check commit: `git log -1` (should show commit bb45fe6 or later)
- Verify file was updated: `grep "whole-population" main_experiment.py`

**Module command not found warning?**
- This is just a warning, not an error (the real error is the missing experiment mode)
- After updating the code, this can be ignored
