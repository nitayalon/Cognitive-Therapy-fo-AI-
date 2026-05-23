#!/bin/bash -l
#SBATCH -o ./slurm_logs/wp_unified_%x_%A_%a.out
#SBATCH -e ./slurm_logs/wp_unified_%x_%A_%a.err
#SBATCH -D ./
#SBATCH --partition=highmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nitay.alon@tuebingen.mpg.de
#SBATCH --time=3-00:00:00
#SBATCH --job-name=wp_unified
#SBATCH --array=0-479

# ============================================================================
# WHOLE POPULATION UNIFIED TRAINING + TESTING
# ============================================================================
# Tasks 0-29:    Training (3 games × 10 seeds)
# Tasks 30-479:  Testing (30 trained models × 15 test conditions)
#
# Training task mapping:
#   0-9:   Prisoner's Dilemma (seeds 0-9)
#   10-19: Hawk-Dove (seeds 0-9)
#   20-29: Stag Hunt (seeds 0-9)
#
# Testing task mapping:
#   Each trained model tested on 15 conditions (3 games × 5 opponents)
#   Test task = 30 + (model_id × 15) + test_condition_id
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
NUM_CONDITIONS=3  # 3 games
NUM_TRAINING_TASKS=$((NUM_CONDITIONS * NUM_SEEDS))  # 30
NUM_TEST_CONDITIONS=15  # 3 games × 5 opponents

SEED_BASE=42
SEED_GAP=10

MAX_EPOCHS="${MAX_EPOCHS:-1000}"
NUM_GAMES="${NUM_GAMES:-100}"
WP_CONFIG="${WP_CONFIG:-config/whole_population_config.json}"
ARRAY_OUTPUT_DIR="experiments/whole_population_unified_${SLURM_ARRAY_JOB_ID}"

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
    echo "Config:       ${WP_CONFIG}"
    echo "============================================================================"
    
    # Run training
    singularity exec ${CONTAINER_PATH} python main_experiment.py \
        --experiment-mode whole-population \
        --mode train-only \
        --task-id ${TRAINING_TASK_ID} \
        --wp-config ${WP_CONFIG} \
        --seed ${SEED} \
        --max-epochs ${MAX_EPOCHS} \
        --num-games ${NUM_GAMES} \
        --device auto \
        --agent-type vanilla \
        --output-dir ${ARRAY_OUTPUT_DIR}/training
    
    EXIT_CODE=$?
    
    echo "Training completed with exit code: ${EXIT_CODE}"
    
    # Verify checkpoint
    CHECKPOINT_DIR="${ARRAY_OUTPUT_DIR}/training/whole_population_task_${TRAINING_TASK_ID}_"*"/checkpoints"
    if ls ${CHECKPOINT_DIR}/*_final_checkpoint.pth 1> /dev/null 2>&1; then
        echo "✓ Checkpoint verified"
    else
        echo "✗ WARNING: No checkpoint found"
        exit 1
    fi
    
else
    PHASE="testing"
    TEST_TASK_ID=$((SLURM_ARRAY_TASK_ID - NUM_TRAINING_TASKS))
    
    # Decode test task to model and condition
    MODEL_ID=$((TEST_TASK_ID / NUM_TEST_CONDITIONS))
    TEST_CONDITION_ID=$((TEST_TASK_ID % NUM_TEST_CONDITIONS))
    
    # Model info
    MODEL_CONDITION_ID=$((MODEL_ID / NUM_SEEDS))
    MODEL_SEED_ID=$((MODEL_ID % NUM_SEEDS))
    MODEL_SEED=$((SEED_BASE + MODEL_SEED_ID * SEED_GAP))
    
    # Test condition (game + opponent)
    TEST_GAME_ID=$((TEST_CONDITION_ID / 5))
    TEST_OPPONENT_ID=$((TEST_CONDITION_ID % 5))
    
    # Map to actual games and opponents
    GAMES=("prisoners-dilemma" "hawk-dove" "stag-hunt")
    OPPONENTS=(0.1 0.3 0.5 0.7 0.9)
    
    TEST_GAME=${GAMES[$TEST_GAME_ID]}
    TEST_OPPONENT=${OPPONENTS[$TEST_OPPONENT_ID]}
    
    # Testing manifest
    MASTER_REGISTRY="${ARRAY_OUTPUT_DIR}/seed_manifests/MASTER_TESTING_REGISTRY.csv"
    if [ ! -f "$MASTER_REGISTRY" ]; then
        echo "array_job_id,array_task_id,model_id,model_condition,model_seed_id,model_seed,test_game,test_opponent,phase,start_time,node" > "$MASTER_REGISTRY"
    fi
    
    START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
    echo "${SLURM_ARRAY_JOB_ID},${SLURM_ARRAY_TASK_ID},${MODEL_ID},${MODEL_CONDITION_ID},${MODEL_SEED_ID},${MODEL_SEED},${TEST_GAME},${TEST_OPPONENT},testing,${START_TIME},${SLURMD_NODENAME}" >> "$MASTER_REGISTRY"
    
    echo "============================================================================"
    echo "TESTING PHASE - Task ${SLURM_ARRAY_TASK_ID}"
    echo "============================================================================"
    echo "Job ID:           ${SLURM_ARRAY_JOB_ID}"
    echo "Task ID:          ${SLURM_ARRAY_TASK_ID}"
    echo "Model ID:         ${MODEL_ID}"
    echo "Model Condition:  ${MODEL_CONDITION_ID}"
    echo "Model Seed:       ${MODEL_SEED}"
    echo "Test Game:        ${TEST_GAME}"
    echo "Test Opponent:    ${TEST_OPPONENT}"
    echo "Node:             ${SLURMD_NODENAME}"
    echo "Start Time:       ${START_TIME}"
    echo "============================================================================"
    
    # Find checkpoint
    CHECKPOINT_PATTERN="${ARRAY_OUTPUT_DIR}/training/whole_population_task_${MODEL_ID}_*/checkpoints/*_final_checkpoint.pth"
    CHECKPOINT_PATH=$(ls ${CHECKPOINT_PATTERN} 2>/dev/null | head -n 1)
    
    if [ -z "$CHECKPOINT_PATH" ]; then
        echo "❌ ERROR: Checkpoint not found for model ${MODEL_ID}"
        echo "   Searched: ${CHECKPOINT_PATTERN}"
        exit 1
    fi
    
    echo "Checkpoint: ${CHECKPOINT_PATH}"
    
    # Run testing
    singularity exec ${CONTAINER_PATH} python main_experiment.py \
        --experiment-mode whole-population \
        --mode eval-only \
        --task-id ${TEST_TASK_ID} \
        --wp-config ${WP_CONFIG} \
        --checkpoint-path ${CHECKPOINT_PATH} \
        --test-game ${TEST_GAME} \
        --test-opponents ${TEST_OPPONENT} \
        --seed ${MODEL_SEED} \
        --num-games ${NUM_GAMES} \
        --device auto \
        --agent-type vanilla \
        --output-dir ${ARRAY_OUTPUT_DIR}/testing
    
    EXIT_CODE=$?
    
    echo "Testing completed with exit code: ${EXIT_CODE}"
fi

exit ${EXIT_CODE}
