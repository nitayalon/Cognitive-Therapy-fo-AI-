# Running Generalization Matrix with Seeds 10-19

## Overview

This guide explains how to run the generalization matrix experiment with **additional seeds 10-19** (different from the original seeds 0-9).

## Seed Configuration

- **Original run**: Seeds 0-9 (seeds: 6431, 6441, 6451, 6461, 6471, 6481, 6491, 6501, 6511, 6521)
- **Additional run**: Seeds 10-19 (seeds: 6531, 6541, 6551, 6561, 6571, 6581, 6591, 6601, 6611, 6621)

This gives you **20 total seeds** (0-19) across two separate runs.

## Training Phase (150 tasks)

### Submit Training Job

```bash
sbatch run_generalization_matrix_train_seeds_10_19.sh
```

This creates **150 training tasks** (15 conditions × 10 seeds):
- 15 conditions (5 opponent ranges × 3 games)
- 10 seeds (IDs 10-19)

### Monitor Training

```bash
# Check job status
squeue -u $USER

# Check output logs
tail -f slurm_logs/train_gen_matrix_train_s10_19_*_*.out
```

### Note the Training Job ID

After submission, save the job ID for testing:
```bash
TRAIN_JOB=<job_id_from_sbatch_output>
```

## Testing Phase (2250 tasks split into 3 parts)

Testing requires 2250 tasks (150 models × 15 test conditions). Due to SLURM array limits, this is split into 3 jobs.

### One-Liner Submission (Recommended)

```bash
# Capture training job ID and submit all test jobs with dependencies
TRAIN_JOB=$(sbatch --parsable run_generalization_matrix_train_seeds_10_19.sh) && \
TEST_P1=$(TRAINING_JOB_ID=$TRAIN_JOB sbatch --parsable --dependency=afterok:$TRAIN_JOB run_generalization_matrix_test_seeds_10_19.sh) && \
TRAINING_JOB_ID=$TRAIN_JOB TEST_JOB_ID_PART1=$TEST_P1 sbatch --dependency=afterok:$TRAIN_JOB run_generalization_matrix_test_seeds_10_19_part2.sh && \
TRAINING_JOB_ID=$TRAIN_JOB TEST_JOB_ID_PART1=$TEST_P1 sbatch --dependency=afterok:$TRAIN_JOB run_generalization_matrix_test_seeds_10_19_part3.sh
```

### Step-by-Step Submission

If you prefer to run testing after training completes:

```bash
# Wait for training to complete, then get the job ID
TRAIN_JOB=920XXX  # Replace with actual training job ID

# Submit all three test parts
TEST_P1=$(TRAINING_JOB_ID=$TRAIN_JOB sbatch --parsable run_generalization_matrix_test_seeds_10_19.sh)
TRAINING_JOB_ID=$TRAIN_JOB TEST_JOB_ID_PART1=$TEST_P1 sbatch run_generalization_matrix_test_seeds_10_19_part2.sh
TRAINING_JOB_ID=$TRAIN_JOB TEST_JOB_ID_PART1=$TEST_P1 sbatch run_generalization_matrix_test_seeds_10_19_part3.sh
```

## Task Breakdown

### Training (150 tasks)
- Array: 0-149
- Task ID → Condition ID: `task_id / 10`
- Task ID → Seed ID: `(task_id % 10) + 10` (maps to 10-19)

**Examples:**
- Task 0: Condition 0, Seed 10 (seed value: 6531)
- Task 5: Condition 0, Seed 15 (seed value: 6581)
- Task 10: Condition 1, Seed 10 (seed value: 6531)
- Task 149: Condition 14, Seed 19 (seed value: 6621)

### Testing (2250 tasks split into 3 jobs)

**Part 1** (array 0-999): Tasks 0-999
**Part 2** (array 0-999): Tasks 1000-1999
**Part 3** (array 0-249): Tasks 2000-2249

Each task tests one model on one test condition:
- Task ID → Model ID: `task_id / 15`
- Task ID → Test Condition: `task_id % 15`

**Examples:**
- Task 0: Model 0 (Cond 0, Seed 10) tested on Condition 0
- Task 15: Model 1 (Cond 0, Seed 11) tested on Condition 0
- Task 2249: Model 149 (Cond 14, Seed 19) tested on Condition 14

## Output Directory Structure

```
experiments/
├── generalization_matrix_train_<TRAIN_JOB_ID>/
│   ├── training/
│   │   ├── condition_0_seed_10/
│   │   ├── condition_0_seed_11/
│   │   ├── ...
│   │   └── condition_14_seed_19/
│   └── seed_manifests/
│       └── MASTER_TRAINING_REGISTRY.csv
└── generalization_matrix_test_<TEST_JOB_ID>/
    └── testing/
        ├── model_0_test_cond_0/
        ├── model_0_test_cond_1/
        └── ...
```

## Verification

After experiments complete:

```bash
# Verify training data
python verify_network_config.py \
    experiments/generalization_matrix_train_<JOB_ID>/training/ \
    --config config/generalization_matrix_config.json

# Check training registry
cat experiments/generalization_matrix_train_<JOB_ID>/seed_manifests/MASTER_TRAINING_REGISTRY.csv
```

Expected output:
- 150 training directories (15 conditions × 10 seeds)
- Seeds range from 10-19
- All use small network architecture (32/1/0.05)

## Combining with Original Seeds (0-9)

You now have two separate experiments:
1. **Original**: Seeds 0-9 (job ID: <original_job_id>)
2. **Additional**: Seeds 10-19 (job ID: <new_job_id>)

For analysis, you can combine results from both experiments to get 20 seeds total.

## Troubleshooting

### Training fails immediately
- Check code is updated on cluster: `git pull origin main`
- Verify config file exists: `config/generalization_matrix_config.json`
- Check logs: `cat slurm_logs/train_gen_matrix_train_s10_19_*_*.err`

### Testing can't find checkpoints
- Ensure TRAINING_JOB_ID is set correctly
- Check training directory exists: `ls experiments/generalization_matrix_train_<JOB_ID>/training/`
- Verify checkpoints exist: `find experiments/generalization_matrix_train_<JOB_ID> -name "*_final_checkpoint.pth"`

### DependencyNeverSatisfied
- Training job failed - check training logs
- Resubmit only the failed tasks or restart training
