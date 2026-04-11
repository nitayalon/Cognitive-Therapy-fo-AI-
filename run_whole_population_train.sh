#!/bin/bash -l
#SBATCH -o ./slurm_logs/wp_train_%x_%A_%a.out
#SBATCH -e ./slurm_logs/wp_train_%x_%A_%a.err
#SBATCH -D ./
#SBATCH --partition=highmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nitay.alon@tuebingen.mpg.de
#SBATCH --time=3-00:00:00
#SBATCH --job-name=wp_train
#SBATCH --array=0-14

# ============================================================================
# WHOLE POPULATION TRAINING PHASE
# ============================================================================
# 15 tasks: 3 games × 5 seeds
# Each task trains one vanilla RL agent against full opponent spectrum [0.0-1.0]
# and saves checkpoint for testing phase
#
# Task ID mapping:
#   0-4:   Prisoner's Dilemma (seeds 0-4)
#   5-9:   Hawk-Dove (seeds 0-4)
#   10-14: Stag Hunt (seeds 0-4)
# ============================================================================

module purge
module load singularity

export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"
export CONTAINER_PATH=/ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

mkdir -p slurm_logs
mkdir -p experiments

# Nested array calculation for seeds
NUM_SEEDS=5
SEED_BASE=42
SEED_GAP=10

CONDITION_ID=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_ID=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
SEED=$((SEED_BASE + SEED_ID * SEED_GAP))

# Training-specific parameters
MAX_EPOCHS="${MAX_EPOCHS:-1000}"
NUM_GAMES="${NUM_GAMES:-100}"
WP_CONFIG="${WP_CONFIG:-config/whole_population_config.json}"
ARRAY_OUTPUT_DIR="experiments/whole_population_train_${SLURM_ARRAY_JOB_ID}"

mkdir -p "${ARRAY_OUTPUT_DIR}"
mkdir -p "${ARRAY_OUTPUT_DIR}/training"
mkdir -p "${ARRAY_OUTPUT_DIR}/seed_manifests"

# Seed manifest
MASTER_REGISTRY="${ARRAY_OUTPUT_DIR}/seed_manifests/MASTER_TRAINING_REGISTRY.csv"
if [ ! -f "$MASTER_REGISTRY" ]; then
    echo "array_job_id,array_task_id,condition_id,seed_id,seed,phase,start_time,node" > "$MASTER_REGISTRY"
fi

START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "${SLURM_ARRAY_JOB_ID},${SLURM_ARRAY_TASK_ID},${CONDITION_ID},${SEED_ID},${SEED},training,${START_TIME},${SLURMD_NODENAME}" >> "$MASTER_REGISTRY"

echo "============================================================================"
echo "WHOLE POPULATION TRAINING - Task ${SLURM_ARRAY_TASK_ID}"
echo "============================================================================"
echo "Job ID:       ${SLURM_ARRAY_JOB_ID}"
echo "Array Task:   ${SLURM_ARRAY_TASK_ID}"
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
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --wp-config ${WP_CONFIG} \
    --seed ${SEED} \
    --max-epochs ${MAX_EPOCHS} \
    --num-games ${NUM_GAMES} \
    --device cuda \
    --agent-type vanilla \
    --output-dir ${ARRAY_OUTPUT_DIR}/training

EXIT_CODE=$?

echo "Training task ${SLURM_ARRAY_TASK_ID} completed with exit code: ${EXIT_CODE}"
echo "Checkpoint should be in: ${ARRAY_OUTPUT_DIR}/training/whole_population_task_${SLURM_ARRAY_TASK_ID}_*/checkpoints/"

# Verify checkpoint was created
CHECKPOINT_DIR="${ARRAY_OUTPUT_DIR}/training/whole_population_task_${SLURM_ARRAY_TASK_ID}_"*"/checkpoints"
if ls ${CHECKPOINT_DIR}/*_final_checkpoint.pth 1> /dev/null 2>&1; then
    echo "✓ Checkpoint verified"
else
    echo "✗ WARNING: Checkpoint not found!"
    exit 1
fi

exit ${EXIT_CODE}
