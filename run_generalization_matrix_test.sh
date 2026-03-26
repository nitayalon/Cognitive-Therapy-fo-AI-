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
#SBATCH --job-name=gen_matrix_test_p1
#SBATCH --array=0-999%100

# TESTING PHASE ONLY - PART 1 of 2
# 1000 tasks: Tasks 0-999 (out of 1050 total)
# MaxArraySize=1001, so split into two jobs
# Each task loads one trained model and tests on one condition

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
    echo "Usage: TRAINING_JOB_ID=12345 sbatch run_generalization_matrix_test.sh"
    exit 1
fi

TRAINING_DIR="experiments/generalization_matrix_train_${TRAINING_JOB_ID}"
if [ ! -d "$TRAINING_DIR" ]; then
    echo "ERROR: Training directory not found: $TRAINING_DIR"
    exit 1
fi

# Testing output directory
TEST_OUTPUT_DIR="experiments/generalization_matrix_test_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${TEST_OUTPUT_DIR}/testing"

# Decode array task ID
NUM_TEST_CONDITIONS=14
MODEL_ID=$((SLURM_ARRAY_TASK_ID / NUM_TEST_CONDITIONS))
TEST_OFFSET=$((SLURM_ARRAY_TASK_ID % NUM_TEST_CONDITIONS))

# Get model details
NUM_SEEDS=5
TRAINING_CONDITION_ID=$((MODEL_ID / NUM_SEEDS))
SEED_ID=$((MODEL_ID % NUM_SEEDS))

# Calculate test condition ID (skip training condition)
# Map test_offset (0-13) to actual condition IDs, skipping training condition
if [ "$TEST_OFFSET" -ge "$TRAINING_CONDITION_ID" ]; then
    TEST_CONDITION_ID=$((TEST_OFFSET + 1))
else
    TEST_CONDITION_ID=$TEST_OFFSET
fi

# Find checkpoint (navigate through experiment directory)
CONDITION_DIR="${TRAINING_DIR}/training/condition_${TRAINING_CONDITION_ID}_seed_${SEED_ID}"
if [ ! -d "$CONDITION_DIR" ]; then
    echo "ERROR: Condition directory not found: $CONDITION_DIR"
    exit 1
fi

# Find experiment directory (generalization_matrix_*)
EXP_DIR=$(find "$CONDITION_DIR" -maxdepth 1 -type d -name "generalization_matrix_*" 2>/dev/null | head -n 1)
if [ -z "$EXP_DIR" ] || [ ! -d "$EXP_DIR" ]; then
    echo "ERROR: Experiment directory not found in $CONDITION_DIR"
    exit 1
fi

# Find checkpoint file inside experiment/checkpoints/
CHECKPOINT_PATH=$(find "$EXP_DIR/checkpoints" -name "*_final_checkpoint.pth" 2>/dev/null | head -n 1)
if [ -z "$CHECKPOINT_PATH" ] || [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found in $EXP_DIR/checkpoints"
    exit 1
fi

echo "=========================================="
echo "TESTING PHASE - Task ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL_ID} (Condition ${TRAINING_CONDITION_ID}, Seed ${SEED_ID})"
echo "Test Condition: ${TEST_CONDITION_ID}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "=========================================="

# Run testing
time singularity exec ${CONTAINER_PATH} python main_experiment.py \
    --experiment-mode generalization-matrix \
    --mode eval-only \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --test-condition-ids "${TEST_CONDITION_ID}" \
    --matrix-config "config/generalization_matrix_config.json" \
    --output-dir "$TEST_OUTPUT_DIR/testing/model_${MODEL_ID}_test_cond_${TEST_CONDITION_ID}" \
    --agent-type vanilla \
    --device auto \
    --verbose

EXIT_STATUS=$?

echo "Testing task ${SLURM_ARRAY_TASK_ID} completed (exit: ${EXIT_STATUS})"

exit $EXIT_STATUS
