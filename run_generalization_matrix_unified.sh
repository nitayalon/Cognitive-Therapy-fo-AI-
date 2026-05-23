#!/bin/bash -l
#SBATCH -o ./slurm_logs/gen_matrix_unified_%x_%A_%a.out
#SBATCH -e ./slurm_logs/gen_matrix_unified_%x_%A_%a.err
#SBATCH -D ./
#SBATCH --partition=highmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nitay.alon@tuebingen.mpg.de
#SBATCH --time=3-00:00:00
#SBATCH --job-name=gen_matrix_unified
#SBATCH --array=0-2399

# ============================================================================
# GENERALIZATION MATRIX UNIFIED TRAINING + TESTING
# ============================================================================
# Tasks 0-149:    Training (15 conditions × 10 seeds)
# Tasks 150-2399: Testing (150 trained models × 15 test conditions)
#
# Training conditions: 15 (3 games × 5 opponent ranges)
# Seeds per condition: 10
# Test conditions: 15 (3 games × 5 opponent ranges)
#
# Testing task mapping:
#   Test task = 150 + (model_id × 15) + test_condition_id
# ============================================================================

module purge
module load singularity

export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"
export CONTAINER_PATH=/ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

mkdir -p slurm_logs
mkdir -p experiments

# Configuration
NUM_SEEDS=10
NUM_CONDITIONS=15  # 3 games × 5 opponent ranges
NUM_TRAINING_TASKS=$((NUM_CONDITIONS * NUM_SEEDS))  # 150
NUM_TEST_CONDITIONS=15  # 3 games × 5 opponent ranges

SEED_BASE=6431
SEED_GAP=10

MAX_EPOCHS="${MAX_EPOCHS:-1000}"
NUM_GAMES="${NUM_GAMES:-100}"
MATRIX_CONFIG="${MATRIX_CONFIG:-config/generalization_matrix_config.json}"
ARRAY_OUTPUT_DIR="experiments/generalization_matrix_unified_${SLURM_ARRAY_JOB_ID}"

mkdir -p "${ARRAY_OUTPUT_DIR}"
mkdir -p "${ARRAY_OUTPUT_DIR}/training"
mkdir -p "${ARRAY_OUTPUT_DIR}/testing"
mkdir -p "${ARRAY_OUTPUT_DIR}/seed_manifests"

# Determine phase
if [ ${SLURM_ARRAY_TASK_ID} -lt ${NUM_TRAINING_TASKS} ]; then
    PHASE="training"
    TRAINING_TASK_ID=${SLURM_ARRAY_TASK_ID}
    
    CONDITION_ID=$((TRAINING_TASK_ID / NUM_SEEDS))
    SEED_ID=$((TRAINING_TASK_ID % NUM_SEEDS))
    SEED=$((SEED_BASE + SEED_ID * SEED_GAP))
    
    # Training manifest
    MASTER_REGISTRY="${ARRAY_OUTPUT_DIR}/seed_manifests/MASTER_TRAINING_REGISTRY.csv"
    if [ ! -f "$MASTER_REGISTRY" ]; then
        echo "array_job_id,array_task_id,condition_id,seed_id,seed,phase,start_time,node" > "$MASTER_REGISTRY"
    fi
    
    START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
    echo "${SLURM_ARRAY_JOB_ID},${SLURM_ARRAY_TASK_ID},${CONDITION_ID},${SEED_ID},${SEED},training,${START_TIME},${SLURMD_NODENAME}" >> "$MASTER_REGISTRY"
    
    echo "============================================================================"
    echo "TRAINING PHASE - Task ${SLURM_ARRAY_TASK_ID}"
    echo "============================================================================"
    echo "Job ID:       ${SLURM_ARRAY_JOB_ID}"
    echo "Task ID:      ${SLURM_ARRAY_TASK_ID}"
    echo "Condition ID: ${CONDITION_ID}"
    echo "Seed ID:      ${SEED_ID}"
    echo "Seed:         ${SEED}"
    echo "Node:         ${SLURMD_NODENAME}"
    echo "Start Time:   ${START_TIME}"
    echo "Config:       ${MATRIX_CONFIG}"
    echo "============================================================================"
    
    # Run training
    singularity exec ${CONTAINER_PATH} python main_experiment.py \
        --experiment-mode generalization-matrix \
        --mode train-only \
        --task-id ${CONDITION_ID} \
        --matrix-config ${MATRIX_CONFIG} \
        --seed ${SEED} \
        --max-epochs ${MAX_EPOCHS} \
        --num-games ${NUM_GAMES} \
        --device auto \
        --agent-type vanilla \
        --output-dir "${ARRAY_OUTPUT_DIR}/training/condition_${CONDITION_ID}_seed_${SEED_ID}"
    
    EXIT_CODE=$?
    
    echo "Training completed with exit code: ${EXIT_CODE}"
    
    # Verify checkpoint
    CHECKPOINT_DIR="${ARRAY_OUTPUT_DIR}/training/condition_${CONDITION_ID}_seed_${SEED_ID}/checkpoints"
    if ls ${CHECKPOINT_DIR}/*_final_checkpoint.pth 1> /dev/null 2>&1; then
        echo "✓ Checkpoint verified"
    else
        echo "✗ WARNING: No checkpoint found"
        exit 1
    fi
    
else
    PHASE="testing"
    TEST_TASK_ID=$((SLURM_ARRAY_TASK_ID - NUM_TRAINING_TASKS))
    
    # Decode test task to model and test condition
    MODEL_ID=$((TEST_TASK_ID / NUM_TEST_CONDITIONS))
    TEST_CONDITION_ID=$((TEST_TASK_ID % NUM_TEST_CONDITIONS))
    
    # Model info (which training condition and seed)
    TRAIN_CONDITION_ID=$((MODEL_ID / NUM_SEEDS))
    TRAIN_SEED_ID=$((MODEL_ID % NUM_SEEDS))
    TRAIN_SEED=$((SEED_BASE + TRAIN_SEED_ID * SEED_GAP))
    
    # Testing manifest
    MASTER_REGISTRY="${ARRAY_OUTPUT_DIR}/seed_manifests/MASTER_TESTING_REGISTRY.csv"
    if [ ! -f "$MASTER_REGISTRY" ]; then
        echo "array_job_id,array_task_id,model_id,train_condition_id,train_seed_id,train_seed,test_condition_id,phase,start_time,node" > "$MASTER_REGISTRY"
    fi
    
    START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
    echo "${SLURM_ARRAY_JOB_ID},${SLURM_ARRAY_TASK_ID},${MODEL_ID},${TRAIN_CONDITION_ID},${TRAIN_SEED_ID},${TRAIN_SEED},${TEST_CONDITION_ID},testing,${START_TIME},${SLURMD_NODENAME}" >> "$MASTER_REGISTRY"
    
    echo "============================================================================"
    echo "TESTING PHASE - Task ${SLURM_ARRAY_TASK_ID}"
    echo "============================================================================"
    echo "Job ID:             ${SLURM_ARRAY_JOB_ID}"
    echo "Task ID:            ${SLURM_ARRAY_TASK_ID}"
    echo "Model ID:           ${MODEL_ID}"
    echo "Train Condition ID: ${TRAIN_CONDITION_ID}"
    echo "Train Seed ID:      ${TRAIN_SEED_ID}"
    echo "Train Seed:         ${TRAIN_SEED}"
    echo "Test Condition ID:  ${TEST_CONDITION_ID}"
    echo "Node:               ${SLURMD_NODENAME}"
    echo "Start Time:         ${START_TIME}"
    echo "============================================================================"
    
    # Find checkpoint
    CHECKPOINT_PATTERN="${ARRAY_OUTPUT_DIR}/training/condition_${TRAIN_CONDITION_ID}_seed_${TRAIN_SEED_ID}/checkpoints/*_final_checkpoint.pth"
    CHECKPOINT_PATH=$(ls ${CHECKPOINT_PATTERN} 2>/dev/null | head -n 1)
    
    if [ -z "$CHECKPOINT_PATH" ]; then
        echo "❌ ERROR: Checkpoint not found"
        echo "   Train condition: ${TRAIN_CONDITION_ID}, Seed: ${TRAIN_SEED_ID}"
        echo "   Searched: ${CHECKPOINT_PATTERN}"
        exit 1
    fi
    
    echo "Checkpoint: ${CHECKPOINT_PATH}"
    
    # Run testing
    singularity exec ${CONTAINER_PATH} python main_experiment.py \
        --experiment-mode generalization-matrix \
        --mode eval-only \
        --task-id ${TEST_CONDITION_ID} \
        --matrix-config ${MATRIX_CONFIG} \
        --checkpoint-path ${CHECKPOINT_PATH} \
        --seed ${TRAIN_SEED} \
        --num-games ${NUM_GAMES} \
        --device auto \
        --agent-type vanilla \
        --output-dir "${ARRAY_OUTPUT_DIR}/testing/model_${MODEL_ID}_testcond_${TEST_CONDITION_ID}"
    
    EXIT_CODE=$?
    
    echo "Testing completed with exit code: ${EXIT_CODE}"
fi

exit ${EXIT_CODE}
