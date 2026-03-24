#!/bin/bash -l
#SBATCH -o ./slurm_logs/train_%x_%A_%a.out
#SBATCH -e ./slurm_logs/train_%x_%A_%a.err
#SBATCH -D ./
#SBATCH --partition=highmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nitay.alon@tuebingen.mpg.de
#SBATCH --time=3-00:00:00
#SBATCH --job-name=gen_matrix_train
#SBATCH --array=0-74

# TRAINING PHASE ONLY
# 75 tasks: 15 conditions × 5 seeds
# Each task trains one model and saves checkpoint

module purge
module load singularity

export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"
export CONTAINER_PATH=/ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

mkdir -p slurm_logs
mkdir -p experiments

# Nested array calculation
NUM_SEEDS=5
SEED_BASE=6431
SEED_GAP=10

CONDITION_ID=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_ID=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
SEED=$((SEED_BASE + SEED_ID * SEED_GAP))

# Training-specific parameters
MAX_EPOCHS="${MAX_EPOCHS:-500}"
NUM_GAMES="${NUM_GAMES:-100}"
MATRIX_CONFIG="${MATRIX_CONFIG:-config/generalization_matrix_config.json}"
ARRAY_OUTPUT_DIR="experiments/generalization_matrix_train_${SLURM_ARRAY_JOB_ID}"

mkdir -p "${ARRAY_OUTPUT_DIR}"
mkdir -p "${ARRAY_OUTPUT_DIR}/training"
mkdir -p "${ARRAY_OUTPUT_DIR}/seed_manifests"

# Seed manifest
MASTER_REGISTRY="${ARRAY_OUTPUT_DIR}/seed_manifests/MASTER_TRAINING_REGISTRY.csv"
if [ ! -f "$MASTER_REGISTRY" ]; then
    echo "array_job_id,array_task_id,condition_id,seed_id,seed,phase,start_time,node" > "$MASTER_REGISTRY"
fi
echo "${SLURM_ARRAY_JOB_ID},${SLURM_ARRAY_TASK_ID},${CONDITION_ID},${SEED_ID},${SEED},training,$(date -Iseconds),${SLURM_NODELIST}" >> "$MASTER_REGISTRY"

# Manifest file
MANIFEST_FILE="${ARRAY_OUTPUT_DIR}/seed_manifests/train_task_${SLURM_ARRAY_TASK_ID}_manifest.txt"
cat > "$MANIFEST_FILE" << EOF
========================================
TRAINING PHASE - SEED MANIFEST
========================================
Array Job ID:        ${SLURM_ARRAY_JOB_ID}
Array Task ID:       ${SLURM_ARRAY_TASK_ID}
Condition ID:        ${CONDITION_ID}
Seed ID:             ${SEED_ID}
Random Seed:         ${SEED}
Phase:               TRAINING
----------------------------------------
Node:                ${SLURM_NODELIST}
Start Time:          $(date)
Output Directory:    ${ARRAY_OUTPUT_DIR}
========================================
EOF

echo "=========================================="
echo "TRAINING PHASE - Task ${SLURM_ARRAY_TASK_ID}"
echo "Condition: ${CONDITION_ID}, Seed: ${SEED}"
echo "=========================================="

# Run training
time singularity exec ${CONTAINER_PATH} python main_experiment.py \
    --experiment-mode generalization-matrix \
    --mode train-only \
    --task-id ${CONDITION_ID} \
    --matrix-config "$MATRIX_CONFIG" \
    --max-epochs "$MAX_EPOCHS" \
    --num-games "$NUM_GAMES" \
    --output-dir "$ARRAY_OUTPUT_DIR/training/condition_${CONDITION_ID}_seed_${SEED_ID}" \
    --seed ${SEED} \
    --agent-type vanilla \
    --device auto \
    --verbose

EXIT_STATUS=$?

# Update manifest
cat >> "$MANIFEST_FILE" << EOF

========================================
COMPLETION STATUS
========================================
End Time:            $(date)
Exit Status:         ${EXIT_STATUS}
Duration:            ${SECONDS} seconds
========================================
EOF

echo "Training task ${SLURM_ARRAY_TASK_ID} completed (exit: ${EXIT_STATUS})"

exit $EXIT_STATUS
