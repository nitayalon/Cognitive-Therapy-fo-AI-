#!/bin/bash -l
#SBATCH -o ./slurm_logs/test_%x_%A_%a.out
#SBATCH -e ./slurm_logs/test_%x_%A_%a.err
#SBATCH -D ./
#SBATCH --partition=highmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nitay.alon@tuebingen.mpg.de
#SBATCH --time=0-06:00:00
#SBATCH --job-name=gen_matrix_test_s10_19_p2
#SBATCH --array=0-999%100

# TESTING PHASE - SEEDS 10-19 (Additional Run) - PART 2 of 3
# 1000 tasks: Tasks 1000-1999 (out of 2250 total)
# Each task loads one trained model (from seeds 10-19) and tests on one condition

module purge
module load singularity

export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"
export CONTAINER_PATH=/ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

mkdir -p slurm_logs

# User must provide training job ID
TRAINING_JOB_ID="${TRAINING_JOB_ID}"
if [ -z "$TRAINING_JOB_ID" ]; then
    echo "ERROR: TRAINING_JOB_ID environment variable not set"
    echo "Usage: TRAINING_JOB_ID=12345 sbatch run_generalization_matrix_test_seeds_10_19_part2.sh"
    exit 1
fi

TRAINING_DIR="experiments/generalization_matrix_train_${TRAINING_JOB_ID}"
if [ ! -d "$TRAINING_DIR" ]; then
    echo "ERROR: Training directory not found: $TRAINING_DIR"
    exit 1
fi

# Testing output directory (same as part 1)
TEST_OUTPUT_DIR="experiments/generalization_matrix_test_${TEST_JOB_ID_PART1}"
if [ -z "$TEST_JOB_ID_PART1" ]; then
    echo "WARNING: TEST_JOB_ID_PART1 not set, using current job ID"
    TEST_OUTPUT_DIR="experiments/generalization_matrix_test_${SLURM_ARRAY_JOB_ID}"
fi
mkdir -p "${TEST_OUTPUT_DIR}/testing"

# Decode array task ID (offset by 1000)
TASK_ID=$((SLURM_ARRAY_TASK_ID + 1000))
NUM_TEST_CONDITIONS=15
MODEL_ID=$((TASK_ID / NUM_TEST_CONDITIONS))
TEST_CONDITION_ID=$((TASK_ID % NUM_TEST_CONDITIONS))

# Get model details (seeds 10-19)
NUM_SEEDS=10
SEED_OFFSET=10
TRAINING_CONDITION_ID=$((MODEL_ID / NUM_SEEDS))
SEED_ID_RAW=$((MODEL_ID % NUM_SEEDS))
SEED_ID=$((SEED_ID_RAW + SEED_OFFSET))

# Find checkpoint
CONDITION_DIR="${TRAINING_DIR}/training/condition_${TRAINING_CONDITION_ID}_seed_${SEED_ID}"
if [ ! -d "$CONDITION_DIR" ]; then
    echo "ERROR: Condition directory not found: $CONDITION_DIR"
    exit 1
fi

EXP_DIR=$(find "$CONDITION_DIR" -maxdepth 1 -type d -name "generalization_matrix_*" 2>/dev/null | head -n 1)
if [ -z "$EXP_DIR" ] || [ ! -d "$EXP_DIR" ]; then
    echo "ERROR: Experiment directory not found in $CONDITION_DIR"
    exit 1
fi

CHECKPOINT_PATH=$(find "$EXP_DIR/checkpoints" -name "*_final_checkpoint.pth" 2>/dev/null | head -n 1)
if [ -z "$CHECKPOINT_PATH" ] || [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found in $EXP_DIR/checkpoints"
    exit 1
fi

echo "=========================================="
echo "TESTING PHASE (SEEDS 10-19) PART 2 - Task ${TASK_ID}"
echo "Model: ${MODEL_ID} (Condition ${TRAINING_CONDITION_ID}, Seed ${SEED_ID})"
echo "Test Condition: ${TEST_CONDITION_ID}"
echo "=========================================="

# Run testing
time singularity exec ${CONTAINER_PATH} python main_experiment.py \
    --experiment-mode generalization-matrix \
    --mode eval-only \
    --task-id ${TRAINING_CONDITION_ID} \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --test-condition-ids "${TEST_CONDITION_ID}" \
    --matrix-config "config/generalization_matrix_config.json" \
    --output-dir "$TEST_OUTPUT_DIR/testing/model_${MODEL_ID}_test_cond_${TEST_CONDITION_ID}" \
    --num-games 100 \
    --device auto

EXIT_STATUS=$?
echo "Testing task ${TASK_ID} completed (exit: ${EXIT_STATUS})"
exit $EXIT_STATUS
