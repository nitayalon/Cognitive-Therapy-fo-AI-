# SLURM Batch Script Execution Guide

**Last Updated:** May 8, 2026  
**Architecture:** Separate Embeddings (9-element input with 6 embedding layers)

This guide provides bash commands for running all SLURM batch scripts in the Cognitive Therapy for AI project on the cluster.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Generalization Matrix Experiments](#generalization-matrix-experiments)
- [Whole Population Experiments](#whole-population-experiments)
- [Monitoring Jobs](#monitoring-jobs)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Ensure you're on the cluster login node
```bash
ssh <your_username>@cluster.address
cd /path/to/Cognitive-Therapy-fo-AI-
```

### 2. Verify directory structure
```bash
ls -la slurm_logs/  # Should exist; if not: mkdir -p slurm_logs
ls -la experiments/ # Should exist; if not: mkdir -p experiments
```

### 3. Check Singularity container availability
```bash
ls -lh /ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif
```

### 4. Verify git is on latest commit
```bash
git status
git pull origin main
```

---

## Generalization Matrix Experiments

**Paradigm:** Task-Opponent  
**Training:** 15 conditions (3 games × 5 opponent types) × 5 seeds = **75 models**  
**Testing:** 75 models × 14 test conditions = **1,050 test tasks** (split into 2 parts)

### Step 1: Submit Training Job

```bash
sbatch run_generalization_matrix_train.sh
```

**What it does:**
- Runs 75 parallel training tasks (array job with `--array=0-74`)
- Each task trains one model on one game-opponent pair with one random seed
- Training takes ~3 days (72 hours) per task
- Saves checkpoints to `experiments/generalization_matrix_train_<JOB_ID>/`

**Monitor training:**
```bash
squeue -u $USER
tail -f slurm_logs/train_gen_matrix_train_<JOB_ID>_0.out  # Check first task
```

**Capture the job ID:**
```bash
TRAIN_JOB=$(sbatch --parsable run_generalization_matrix_train.sh)
echo "Training Job ID: $TRAIN_JOB"
```

### Step 2: Submit Testing Jobs (After Training Completes)

#### Option A: Manual submission after training finishes
```bash
# Wait for training to complete, then:
TRAINING_JOB_ID=<your_training_job_id> sbatch run_generalization_matrix_test.sh
TRAINING_JOB_ID=<your_training_job_id> sbatch run_generalization_matrix_test_part2.sh
```

#### Option B: Automatic submission with dependency
```bash
TRAIN_JOB=$(sbatch --parsable run_generalization_matrix_train.sh)
echo "Training: $TRAIN_JOB"

# Test Part 1: Tasks 0-999
sbatch --dependency=afterok:$TRAIN_JOB \
       --export=TRAINING_JOB_ID=$TRAIN_JOB \
       run_generalization_matrix_test.sh

# Test Part 2: Tasks 1000-1049
sbatch --dependency=afterok:$TRAIN_JOB \
       --export=TRAINING_JOB_ID=$TRAIN_JOB \
       run_generalization_matrix_test_part2.sh
```

**What testing does:**
- Part 1: 1,000 tasks (array `0-999`)
- Part 2: 50 tasks (array `0-49`, offset by 1000)
- Each task loads one trained model and tests on one condition
- Testing takes ~6 hours per task
- Saves results to `experiments/generalization_matrix_test_<TEST_JOB_ID>/`

### Step 3: Verify Results

```bash
# Check training completed
TRAIN_DIR="experiments/generalization_matrix_train_${TRAIN_JOB}"
ls -l ${TRAIN_DIR}/training/ | wc -l  # Should show 75 model directories

# Check testing completed
TEST_DIR="experiments/generalization_matrix_test_<TEST_JOB_ID>"
ls -l ${TEST_DIR}/testing/ | wc -l  # Should show 1050 test directories
```

---

## Whole Population Experiments

**Paradigm:** Task-Only (whole opponent population)  
**Training:** 3 games × 5 seeds = **15 models**  
**Testing:** 15 models × 15 test conditions (3 games × 5 opponent types) = **225 test tasks**

### Step 1: Submit Training Job

```bash
sbatch run_whole_population_train.sh
```

**What it does:**
- Runs 15 parallel training tasks (array job with `--array=0-14`)
- Each task trains one vanilla RL agent against full opponent spectrum [0.0-1.0]
- Training takes ~3 days (72 hours) per task
- Saves checkpoints to `experiments/whole_population_train_<JOB_ID>/`

**Task ID mapping:**
- Tasks 0-4: Prisoner's Dilemma (seeds 0-4)
- Tasks 5-9: Hawk-Dove (seeds 0-4)
- Tasks 10-14: Stag Hunt (seeds 0-4)

**Capture the job ID:**
```bash
TRAIN_JOB=$(sbatch --parsable run_whole_population_train.sh)
echo "Training Job ID: $TRAIN_JOB"
```

### Step 2: Submit Testing Job (After Training Completes)

#### Option A: Manual submission
```bash
TRAINING_JOB_ID=<your_training_job_id> sbatch run_whole_population_test.sh
```

#### Option B: Automatic submission with dependency (RECOMMENDED)
```bash
TRAIN_JOB=$(sbatch --parsable run_whole_population_train.sh)
echo "Training: $TRAIN_JOB"

sbatch --dependency=afterok:$TRAIN_JOB \
       --export=TRAINING_JOB_ID=$TRAIN_JOB \
       run_whole_population_test.sh
```

#### Option C: One-liner (training + testing)
```bash
TRAIN_JOB=$(sbatch --parsable run_whole_population_train.sh) && \
  echo "Training: $TRAIN_JOB" && \
  sbatch --dependency=afterok:$TRAIN_JOB \
         --export=TRAINING_JOB_ID=$TRAIN_JOB \
         run_whole_population_test.sh
```

**What testing does:**
- Runs 225 parallel testing tasks (array job with `--array=0-224`)
- Each task tests one trained model on one game+opponent combination
- Testing takes ~2 hours per task
- Saves results to `experiments/whole_population_test_<TEST_JOB_ID>/`

### Step 3: Verify Results

```bash
# Check training completed
TRAIN_DIR="experiments/whole_population_train_${TRAIN_JOB}"
ls -l ${TRAIN_DIR}/training/ | wc -l  # Should show 15 model directories

# Check testing completed
TEST_DIR="experiments/whole_population_test_<TEST_JOB_ID>"
ls -l ${TEST_DIR}/testing/ | wc -l  # Should show 225 test directories
```

---

## Monitoring Jobs

### Check job status
```bash
# All your jobs
squeue -u $USER

# Specific job
squeue -j <JOB_ID>

# Detailed job info
scontrol show job <JOB_ID>
```

### Monitor array job progress
```bash
# Count completed tasks
squeue -j <JOB_ID> -t COMPLETED | wc -l

# Check running tasks
squeue -j <JOB_ID> -t RUNNING

# Check pending tasks
squeue -j <JOB_ID> -t PENDING
```

### View logs
```bash
# Live monitoring of a specific task
tail -f slurm_logs/train_gen_matrix_train_<JOB_ID>_<TASK_ID>.out

# Check for errors
grep -i error slurm_logs/train_gen_matrix_train_<JOB_ID>_*.err

# View all error logs
ls -lh slurm_logs/*.err
```

### Cancel jobs
```bash
# Cancel specific job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Cancel specific array tasks
scancel <JOB_ID>_[0-10]  # Cancel tasks 0-10
```

---

## Advanced Usage

### Customize Training Parameters

#### Generalization Matrix - Extended Training
```bash
# Train for more epochs
MAX_EPOCHS=1000 sbatch run_generalization_matrix_train.sh

# Train with more games per session
NUM_GAMES=200 sbatch run_generalization_matrix_train.sh

# Combined
MAX_EPOCHS=1000 NUM_GAMES=200 sbatch run_generalization_matrix_train.sh
```

#### Whole Population - Custom Configuration
```bash
# Use custom config file
WP_CONFIG=config/my_custom_config.json sbatch run_whole_population_train.sh
```

### Rerun Specific Tasks

#### Rerun single training task
```bash
# For generalization matrix
sbatch --array=5 run_generalization_matrix_train.sh  # Rerun task 5 only

# For whole population
sbatch --array=10 run_whole_population_train.sh  # Rerun task 10 only
```

#### Rerun range of tasks
```bash
# Rerun tasks 0-9
sbatch --array=0-9 run_generalization_matrix_train.sh

# Rerun tasks 5-14
sbatch --array=5-14 run_whole_population_train.sh
```

### Test Specific Models Only

```bash
# Test only model 0 against all conditions (14 tests)
TRAINING_JOB_ID=<job_id> sbatch --array=0-13 run_generalization_matrix_test.sh

# Test model 5 against all conditions
TRAINING_JOB_ID=<job_id> sbatch --array=70-83 run_generalization_matrix_test.sh
# (5 * 14 = 70, 5 * 14 + 13 = 83)
```

### Limit Concurrent Tasks (Throttling)

```bash
# Run max 10 tasks at once
sbatch --array=0-74%10 run_generalization_matrix_train.sh

# Run max 50 tasks at once for testing
TRAINING_JOB_ID=<job_id> sbatch --array=0-999%50 run_generalization_matrix_test.sh
```

---

## Troubleshooting

### Common Issues

#### 1. "TRAINING_JOB_ID environment variable not set"
**Problem:** Testing script requires training job ID  
**Solution:**
```bash
# Check your training job ID first
squeue -u $USER

# Then set it explicitly
TRAINING_JOB_ID=123456 sbatch run_generalization_matrix_test.sh
```

#### 2. "Training directory not found"
**Problem:** Testing can't find training results  
**Solution:**
```bash
# Verify training directory exists
ls -la experiments/generalization_matrix_train_<TRAIN_JOB_ID>

# If training is still running, wait for it to complete
squeue -j <TRAIN_JOB_ID>
```

#### 3. "Container not found"
**Problem:** Singularity container path is incorrect  
**Solution:**
```bash
# Check container exists
ls -lh /ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif

# If not found, update path in .sh scripts or contact cluster admin
```

#### 4. Jobs pending for long time
**Problem:** Cluster is busy or requesting too many resources  
**Solution:**
```bash
# Check job priority
squeue -j <JOB_ID> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Consider reducing resource requirements or time limits
# Edit the .sh script SBATCH parameters before submitting
```

#### 5. Out of memory errors
**Problem:** Tasks need more RAM  
**Solution:**
```bash
# Edit .sh file and increase memory:
# For training: Change from 32G to 64G
# For testing: Change from 16G to 32G

# Or submit with custom memory:
sbatch --mem=64G run_generalization_matrix_train.sh
```

### Debug a Failed Task

```bash
# 1. Check error log
cat slurm_logs/train_gen_matrix_train_<JOB_ID>_<TASK_ID>.err

# 2. Check output log
cat slurm_logs/train_gen_matrix_train_<JOB_ID>_<TASK_ID>.out

# 3. Check Python traceback
grep -A 20 "Traceback" slurm_logs/train_gen_matrix_train_<JOB_ID>_<TASK_ID>.err

# 4. Verify experiment output
ls -la experiments/generalization_matrix_train_<JOB_ID>/training/
```

### Verify Checkpoint Integrity

```bash
# Check if checkpoint files exist for all models
TRAIN_DIR="experiments/generalization_matrix_train_<JOB_ID>"
find ${TRAIN_DIR}/training -name "*.pth" | wc -l  # Should match number of tasks

# List any missing checkpoints
for i in {0..74}; do
  MODEL_DIR=$(find ${TRAIN_DIR}/training -name "generalization_matrix_task_${i}_*" -type d)
  if [ -z "$MODEL_DIR" ]; then
    echo "Missing: Task $i"
  fi
done
```

---

## Quick Reference

### Training Commands
```bash
# Generalization Matrix (75 tasks, ~3 days)
sbatch run_generalization_matrix_train.sh

# Whole Population (15 tasks, ~3 days)
sbatch run_whole_population_train.sh
```

### Testing Commands
```bash
# Generalization Matrix (1050 tasks split in 2, ~6 hours each)
TRAINING_JOB_ID=<id> sbatch run_generalization_matrix_test.sh
TRAINING_JOB_ID=<id> sbatch run_generalization_matrix_test_part2.sh

# Whole Population (225 tasks, ~2 hours each)
TRAINING_JOB_ID=<id> sbatch run_whole_population_test.sh
```

### Full Pipeline (Automatic)
```bash
# Generalization Matrix
TRAIN=$(sbatch --parsable run_generalization_matrix_train.sh) && \
  sbatch --dependency=afterok:$TRAIN --export=TRAINING_JOB_ID=$TRAIN run_generalization_matrix_test.sh && \
  sbatch --dependency=afterok:$TRAIN --export=TRAINING_JOB_ID=$TRAIN run_generalization_matrix_test_part2.sh

# Whole Population
TRAIN=$(sbatch --parsable run_whole_population_train.sh) && \
  sbatch --dependency=afterok:$TRAIN --export=TRAINING_JOB_ID=$TRAIN run_whole_population_test.sh
```

### Monitoring
```bash
squeue -u $USER                                    # All your jobs
watch -n 10 'squeue -u $USER'                      # Auto-refresh every 10s
tail -f slurm_logs/<script>_<JOB_ID>_<TASK>.out   # Live log monitoring
```

---

## Architecture Notes (Separate Embeddings - May 2026)

**Important:** As of May 8, 2026, the network architecture uses **separate embedding layers**:

- **Input:** 9 elements split into 6 components
  - Environmental: payoff_matrix (4), round_number (1)
  - Social: opponent_action (1), agent_action (1), agent_reward (1), opponent_reward (1)

- **Architecture:** Each component → Linear→ReLU→LayerNorm → concatenate → LSTM

- **Incompatibility:** Models trained before this change (archived in `experiments/archive_pre_separate_embeddings_20260508/`) are NOT compatible with current code.

To analyze old results, either:
1. Use archived analysis scripts
2. Checkout commit `c5a299b` (pre-separate-embeddings)

---

## Contact

For issues with:
- **SLURM/Cluster:** Contact cluster support
- **Code/Experiments:** Check GitHub issues or contact research team
- **Results Analysis:** See `experiments/analysis_scripts/` for analysis tools

---

**Last Updated:** May 8, 2026  
**Git Commit:** 3a10f2d (separate embeddings architecture)  
**Cluster:** Max Planck Institute cluster with Singularity containers
