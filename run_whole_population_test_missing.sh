#!/bin/bash -l
#SBATCH -o ./slurm_logs/wp_test_missing_%x_%A_%a.out
#SBATCH -e ./slurm_logs/wp_test_missing_%x_%A_%a.err
#SBATCH -D ./
#SBATCH --partition=highmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nitay.alon@tuebingen.mpg.de
#SBATCH --time=0-02:00:00
#SBATCH --job-name=wp_test_missing
#SBATCH --array=0-224

# ============================================================================
# WHOLE POPULATION TESTING RERUN
# ============================================================================
# Reason for rerun:
#   The original test job 902267 completed evaluations but failed while saving
#   results due to KeyError('opponent_range') in main_experiment.py.
#
# Missing scope verified locally:
#   experiments/whole_population_test_902267/testing contains 0 result CSVs,
#   so the full 225-task test array must be rerun.
#
# Task mapping:
#   MODEL_ID = TASK_ID / 15  (trained model: 0-14)
#   TEST_ID  = TASK_ID % 15  (test condition: 0-14)
# ============================================================================

module purge
module load singularity

export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"
export CONTAINER_PATH=/ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

mkdir -p slurm_logs

# Training job that produced the checkpoints for this rerun.
TRAINING_JOB_ID="${TRAINING_JOB_ID:-902266}"
FAILED_TEST_JOB_ID="${FAILED_TEST_JOB_ID:-902267}"

TRAINING_DIR="experiments/whole_population_train_${TRAINING_JOB_ID}"
if [ ! -d "$TRAINING_DIR" ]; then
    echo "ERROR: Training directory not found: $TRAINING_DIR"
    exit 1
fi

TEST_OUTPUT_DIR="experiments/whole_population_test_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${TEST_OUTPUT_DIR}/testing"

WP_CONFIG="${WP_CONFIG:-config/whole_population_config.json}"

NUM_TEST_CONDITIONS=15
MODEL_ID=$((SLURM_ARRAY_TASK_ID / NUM_TEST_CONDITIONS))
TEST_CONDITION_ID=$((SLURM_ARRAY_TASK_ID % NUM_TEST_CONDITIONS))

GAME_ID=$((TEST_CONDITION_ID / 5))
OPP_ID=$((TEST_CONDITION_ID % 5))

GAMES=("prisoners-dilemma" "hawk-dove" "stag-hunt")
TEST_GAME=${GAMES[$GAME_ID]}

OPPONENTS=(0.1 0.3 0.5 0.7 0.9)
TEST_OPPONENT=${OPPONENTS[$OPP_ID]}

CHECKPOINT_PATTERN="${TRAINING_DIR}/training/whole_population_task_${MODEL_ID}_*/checkpoints/*_final_checkpoint.pth"
CHECKPOINT_PATH=$(ls ${CHECKPOINT_PATTERN} 2>/dev/null | head -n 1)

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found for model ${MODEL_ID}"
    echo "Searched: ${CHECKPOINT_PATTERN}"
    exit 1
fi

echo "============================================================================"
echo "WHOLE POPULATION TESTING RERUN - Task ${SLURM_ARRAY_TASK_ID}"
echo "============================================================================"
echo "Rerunning failed job: ${FAILED_TEST_JOB_ID}"
echo "Training job:        ${TRAINING_JOB_ID}"
echo "Current job:         ${SLURM_ARRAY_JOB_ID}"
echo "Array task:          ${SLURM_ARRAY_TASK_ID}"
echo "Model ID:            ${MODEL_ID}"
echo "Test condition:      ${TEST_CONDITION_ID}"
echo "Test game:           ${TEST_GAME}"
echo "Test opponent:       ${TEST_OPPONENT}"
echo "Checkpoint:          ${CHECKPOINT_PATH}"
echo "Node:                ${SLURMD_NODENAME}"
echo "============================================================================"

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

RESULTS_PATTERN="${TEST_OUTPUT_DIR}/testing/whole_population_task_${SLURM_ARRAY_TASK_ID}_*/results/*.csv"
if ls ${RESULTS_PATTERN} 1> /dev/null 2>&1; then
    echo "✓ Results verified"
else
    echo "✗ WARNING: Results not found!"
    exit 1
fi

exit ${EXIT_CODE}