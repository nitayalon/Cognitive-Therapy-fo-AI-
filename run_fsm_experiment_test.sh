#!/bin/bash -l
#SBATCH -o ./slurm_logs/fsm_test_%x_%A_%a.out
#SBATCH -e ./slurm_logs/fsm_test_%x_%A_%a.err
#SBATCH -D ./
#SBATCH --partition=highmem
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nitay.alon@tuebingen.mpg.de
#SBATCH --time=00:30:00
#SBATCH --job-name=fsm_repr_test
#SBATCH --array=0-999

# FSM REPRESENTATION EXPERIMENT - COMPREHENSIVE TESTING PHASE
# 11,340 tasks TOTAL: 420 trained models × 27 test conditions
# Test conditions: 3 games × 9 opponents = CROSS-GAME + CROSS-OPPONENT generalization
# 
# Due to cluster array limit (1001), this script handles chunks of 1000 tasks
# Use submit_fsm_test_jobs.sh to submit all chunks automatically
# Or manually: TASK_OFFSET=0 sbatch run_fsm_experiment_test.sh (tasks 0-999)
#              TASK_OFFSET=1000 sbatch run_fsm_experiment_test.sh (tasks 1000-1999)
#              etc.
#
# Each task:
#   1. Load trained checkpoint
#   2. Evaluate on test game+opponent (possibly unseen)
#   3. Extract FSM on test data
#   4. Measure generalization (reward, fidelity, policy shift)

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
    echo "Usage: TRAINING_JOB_ID=12345 sbatch run_fsm_experiment_test.sh"
    exit 1
fi

TRAINING_DIR="experiments/fsm_representation_train_${TRAINING_JOB_ID}"
if [ ! -d "$TRAINING_DIR" ]; then
    echo "ERROR: Training directory not found: $TRAINING_DIR"
    exit 1
fi

# Testing output directory
TEST_OUTPUT_DIR="experiments/fsm_representation_test_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${TEST_OUTPUT_DIR}"

# Handle task offset for multi-batch submission (cluster array limit: 1001)
TASK_OFFSET=${TASK_OFFSET:-0}
ACTUAL_TASK_ID=$((TASK_OFFSET + SLURM_ARRAY_TASK_ID))

# Validate task ID is within valid range
if [ $ACTUAL_TASK_ID -ge 11340 ]; then
    echo "ERROR: Task ID $ACTUAL_TASK_ID exceeds total tasks (11340)"
    exit 1
fi

# Decode array task ID
# 420 models × 27 test conditions (3 games × 9 opponents) = 11,340 tasks
GAMES=("prisoners-dilemma" "stag-hunt" "hawk-dove")
ALL_OPPONENTS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)  # ALL opponents for comprehensive testing

NUM_TEST_CONDITIONS=27  # 3 games × 9 opponents
MODEL_ID=$((ACTUAL_TASK_ID / NUM_TEST_CONDITIONS))
TEST_COND_ID=$((ACTUAL_TASK_ID % NUM_TEST_CONDITIONS))

# Map test condition to game and opponent
TEST_GAME_IDX=$((TEST_COND_ID / 9))
TEST_OPP_IDX=$((TEST_COND_ID % 9))
TEST_GAME=${GAMES[$TEST_GAME_IDX]}
TEST_OPPONENT=${ALL_OPPONENTS[$TEST_OPP_IDX]}

# Get model details
NUM_SEEDS=10
CONDITION_ID=$((MODEL_ID / NUM_SEEDS))
SEED_ID=$((MODEL_ID % NUM_SEEDS))

# Find checkpoint
CONDITION_DIR="${TRAINING_DIR}/condition_${CONDITION_ID}_seed_${SEED_ID}"
if [ ! -d "$CONDITION_DIR" ]; then
    echo "ERROR: Condition directory not found: $CONDITION_DIR"
    exit 1
fi

# Find checkpoint file
CHECKPOINT_PATH=$(find "$CONDITION_DIR" -name "checkpoint.pth" 2>/dev/null | head -n 1)
if [ -z "$CHECKPOINT_PATH" ] || [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found in $CONDITION_DIR"
    exit 1
fi

# Game abbreviation for output directory
case ${TEST_GAME} in
    "prisoners-dilemma") GAME_ABBR="PD" ;;
    "stag-hunt") GAME_ABBR="SH" ;;
    "hawk-dove") GAME_ABBR="HD" ;;
esac

echo "=========================================="
echo "FSM TESTING - Task ${ACTUAL_TASK_ID} (Array ${SLURM_ARRAY_TASK_ID} + Offset ${TASK_OFFSET})"
echo "Model: ${MODEL_ID} (Condition ${CONDITION_ID}, Seed ${SEED_ID})"
echo "Test Game: ${TEST_GAME}"
echo "Test Opponent: ${TEST_OPPONENT}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "=========================================="

# Output directory for this test
TEST_TASK_DIR="${TEST_OUTPUT_DIR}/model_${MODEL_ID}_test_${GAME_ABBR}_${TEST_OPPONENT}"
mkdir -p "$TEST_TASK_DIR"

# Run testing + FSM extraction on test condition
time singularity exec ${CONTAINER_PATH} python run_fsm_experiment.py \
    --mode test \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --test-games ${TEST_GAME} \
    --test-opponents ${TEST_OPPONENT} \
    --n-test-episodes 100 \
    --save-trajectories \
    --save-every-nth-episode 10 \
    --output-dir "$TEST_TASK_DIR"

echo "Task ${ACTUAL_TASK_ID} (Array ${SLURM_ARRAY_TASK_ID}) complete"
