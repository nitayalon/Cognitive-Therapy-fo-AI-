#!/bin/bash -l
#SBATCH -o ./slurm_logs/wp_test_%x_%A_%a.out
#SBATCH -e ./slurm_logs/wp_test_%x_%A_%a.err
#SBATCH -D ./
#SBATCH --partition=highmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nitay.alon@tuebingen.mpg.de
#SBATCH --time=0-02:00:00
#SBATCH --job-name=wp_test
#SBATCH --array=0-224

# ============================================================================
# WHOLE POPULATION TESTING PHASE
# ============================================================================
# 225 tasks: 15 trained models × 15 test conditions (3 games × 5 opponents)
# Each task loads one trained model and tests on one game+opponent combination
#
# Test conditions per model:
#   - 3 games (PD, HD, SH) × 5 opponents [0.1, 0.3, 0.5, 0.7, 0.9] = 15 tests
#
# Task ID mapping:
#   MODEL_ID = TASK_ID / 15  (which trained model: 0-14)
#   TEST_ID  = TASK_ID % 15  (which test condition: 0-14)
# ============================================================================

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
    echo "Usage: TRAINING_JOB_ID=12345 sbatch run_whole_population_test.sh"
    exit 1
fi

TRAINING_DIR="experiments/whole_population_train_${TRAINING_JOB_ID}"
if [ ! -d "$TRAINING_DIR" ]; then
    echo "ERROR: Training directory not found: $TRAINING_DIR"
    exit 1
fi

# Testing output directory
TEST_OUTPUT_DIR="experiments/whole_population_test_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${TEST_OUTPUT_DIR}/testing"

# Configuration
WP_CONFIG="${WP_CONFIG:-config/whole_population_config.json}"

# Decode array task ID
NUM_TEST_CONDITIONS=15  # 3 games × 5 opponents
MODEL_ID=$((SLURM_ARRAY_TASK_ID / NUM_TEST_CONDITIONS))
TEST_CONDITION_ID=$((SLURM_ARRAY_TASK_ID % NUM_TEST_CONDITIONS))

# Decode test condition to game and opponent
GAME_ID=$((TEST_CONDITION_ID / 5))
OPP_ID=$((TEST_CONDITION_ID % 5))

# Map to actual game names
GAMES=("prisoners-dilemma" "hawk-dove" "stag-hunt")
TEST_GAME=${GAMES[$GAME_ID]}

# Map to test opponent values [0.1, 0.3, 0.5, 0.7, 0.9]
OPPONENTS=(0.1 0.3 0.5 0.7 0.9)
TEST_OPPONENT=${OPPONENTS[$OPP_ID]}

# Find checkpoint from training
# Checkpoint naming: {TRAINING_DIR}/training/whole_population_task_{MODEL_ID}_*/checkpoints/{GAME}_final_checkpoint.pth
CHECKPOINT_PATTERN="${TRAINING_DIR}/training/whole_population_task_${MODEL_ID}_*/checkpoints/*_final_checkpoint.pth"
CHECKPOINT_PATH=$(ls ${CHECKPOINT_PATTERN} 2>/dev/null | head -n 1)

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found for model ${MODEL_ID}"
    echo "Searched: ${CHECKPOINT_PATTERN}"
    exit 1
fi

echo "============================================================================"
echo "WHOLE POPULATION TESTING - Task ${SLURM_ARRAY_TASK_ID}"
echo "============================================================================" 
echo "Job ID:            ${SLURM_ARRAY_JOB_ID}"
echo "Array Task:        ${SLURM_ARRAY_TASK_ID}"
echo "Model ID:          ${MODEL_ID}"
echo "Test Condition:    ${TEST_CONDITION_ID}"
echo "Test Game:         ${TEST_GAME}"
echo "Test Opponent:     ${TEST_OPPONENT}"
echo "Checkpoint:        ${CHECKPOINT_PATH}"
echo "Training Job:      ${TRAINING_JOB_ID}"
echo "Node:              ${SLURMD_NODENAME}"
echo "============================================================================"

# Run evaluation
singularity exec ${CONTAINER_PATH} python main_experiment.py \
    --experiment-mode whole-population \
    --mode eval-only \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --wp-config ${WP_CONFIG} \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --test-game ${TEST_GAME} \
    --test-opponents ${TEST_OPPONENT} \
    --device auto \
    --agent-type vanilla \
    --output-dir ${TEST_OUTPUT_DIR}/testing

EXIT_CODE=$?

echo "Testing task ${SLURM_ARRAY_TASK_ID} completed with exit code: ${EXIT_CODE}"

# Verify results were created
RESULTS_PATTERN="${TEST_OUTPUT_DIR}/testing/whole_population_task_${SLURM_ARRAY_TASK_ID}_*/results/*.csv"
if ls ${RESULTS_PATTERN} 1> /dev/null 2>&1; then
    echo "✓ Results verified"
else
    echo "✗ WARNING: Results not found!"
fi

exit ${EXIT_CODE}
