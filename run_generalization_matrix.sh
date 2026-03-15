#!/bin/bash -l
#SBATCH -o ./slurm_logs/%x_%A_%a.out
#SBATCH -e ./slurm_logs/%x_%A_%a.err
# Initial working directory:
#SBATCH -D ./
#
# Queue (Partition):
#SBATCH --partition=gpu # nyx partitions: compute, highmem, gpu
#SBATCH --gres=gpu:1
#
# Number of nodes and MPI tasks per node:
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nitay.alon@tuebingen.mpg.de
# Wall clock limit:
#SBATCH --time=2-00:00:00
#SBATCH --job-name=gen_matrix
#SBATCH --array=0-74

# Generalization Matrix Experiment - Multi-Seed Nested Array Job
# 
# NESTED ARRAY STRUCTURE:
# - 15 training conditions (3 games × 5 opponent ranges)
# - 5 random seeds per condition
# - Total: 75 array tasks (0-74)
#
# Task ID Mapping:
#   CONDITION_ID = TASK_ID / 5  (integer division)
#   SEED_ID = TASK_ID % 5       (modulo)
#   SEED = 42 + SEED_ID * 10    (seeds: 42, 52, 62, 72, 82)
#
# Training Conditions (15 total):
#   Conditions 0-4:   Prisoner's Dilemma × [very_low, low, mid, high, very_high]
#   Conditions 5-9:   Hawk-Dove × [very_low, low, mid, high, very_high]
#   Conditions 10-14: Stag-Hunt × [very_low, low, mid, high, very_high]
#
# Example Task Mapping:
#   Task 0-4:   PD+very_low with seeds 42,52,62,72,82
#   Task 5-9:   PD+low with seeds 42,52,62,72,82
#   Task 10-14: PD+mid with seeds 42,52,62,72,82
#   ...
#   Task 70-74: SH+very_high with seeds 42,52,62,72,82
#
# Each task trains on one (game, opponent_range, seed) combination and tests on:
#   - Same game, same opponents (in-distribution baseline)
#   - Same game, new opponents (opponent generalization)
#   - New game, same opponents (game structure generalization)
#   - New game, new opponents (full out-of-distribution)

module purge
module load singularity

export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"
export CONTAINER_PATH=/ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif

# Set Python path for module imports
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Create logs directory if it doesn't exist
mkdir -p slurm_logs
mkdir -p experiments

# ========================================
# NESTED ARRAY CALCULATION
# ========================================
# Number of seeds per condition
NUM_SEEDS=5
SEED_BASE=6431
SEED_GAP=10

# Calculate condition ID and seed ID from array task ID
CONDITION_ID=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_ID=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
SEED=$((SEED_BASE + SEED_ID * SEED_GAP))

# Default experiment parameters (can be overridden via environment variables)
MAX_EPOCHS="${MAX_EPOCHS:-500}"
NUM_GAMES="${NUM_GAMES:-100}"
MATRIX_CONFIG="${MATRIX_CONFIG:-config/generalization_matrix_config.json}"
ARRAY_OUTPUT_DIR="experiments/generalization_matrix_${SLURM_ARRAY_JOB_ID}"

# Create output directories
mkdir -p "${ARRAY_OUTPUT_DIR}"
mkdir -p "${ARRAY_OUTPUT_DIR}/seed_manifests"

# ========================================
# MASTER SEED REGISTRY
# ========================================
# Append this run to the master registry (thread-safe with locking)
MASTER_REGISTRY="${ARRAY_OUTPUT_DIR}/seed_manifests/MASTER_SEED_REGISTRY.csv"

# Create header if file doesn't exist
if [ ! -f "$MASTER_REGISTRY" ]; then
    echo "array_job_id,array_task_id,condition_id,seed_id,seed,start_time,node" > "$MASTER_REGISTRY"
fi

# Append this task's info to master registry
echo "${SLURM_ARRAY_JOB_ID},${SLURM_ARRAY_TASK_ID},${CONDITION_ID},${SEED_ID},${SEED},$(date -Iseconds),${SLURM_NODELIST}" >> "$MASTER_REGISTRY"

# ========================================
# SEED MANIFEST CREATION
# ========================================
# Create a manifest file documenting this specific run
MANIFEST_FILE="${ARRAY_OUTPUT_DIR}/seed_manifests/task_${SLURM_ARRAY_TASK_ID}_manifest.txt"
cat > "$MANIFEST_FILE" << EOF
========================================
GENERALIZATION MATRIX - SEED MANIFEST
========================================
Array Job ID:        ${SLURM_ARRAY_JOB_ID}
Array Task ID:       ${SLURM_ARRAY_TASK_ID}
Condition ID:        ${CONDITION_ID}
Seed ID:             ${SEED_ID}
Random Seed:         ${SEED}
----------------------------------------
Job Name:            ${SLURM_JOB_NAME}
Node:                ${SLURM_NODELIST}
CPUs:                ${SLURM_CPUS_PER_TASK}
Memory:              ${SLURM_MEM_PER_NODE}
Start Time:          $(date)
----------------------------------------
Matrix Config:       ${MATRIX_CONFIG}
Max Epochs:          ${MAX_EPOCHS}
Games per Session:   ${NUM_GAMES}
Output Directory:    ${ARRAY_OUTPUT_DIR}
========================================
EOF

echo "=========================================="
echo "GENERALIZATION MATRIX EXPERIMENT"
echo "=========================================="
echo "ARRAY TASK MAPPING:"
echo "  Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "  Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "  -> Condition ID: $CONDITION_ID (of 15)"
echo "  -> Seed ID: $SEED_ID (of 5)"
echo "  -> Random Seed: $SEED"
echo "=========================================="
echo "JOB DETAILS:"
echo "  Job Name: $SLURM_JOB_NAME"
echo "  Node: $SLURM_NODELIST"
echo "  Start Time: $(date)"
echo "=========================================="
echo "EXPERIMENT CONFIGURATION:"
echo "  Matrix config: $MATRIX_CONFIG"
echo "  Max epochs: $MAX_EPOCHS"
echo "  Games per session: $NUM_GAMES"
echo "  Output directory: $ARRAY_OUTPUT_DIR"
echo "  Manifest file: $MANIFEST_FILE"
echo "=========================================="

# Run the experiment inside the singularity container
time singularity exec ${CONTAINER_PATH} python main_experiment.py \
    --experiment-mode generalization-matrix \
    --task-id ${CONDITION_ID} \
    --matrix-config "$MATRIX_CONFIG" \
    --max-epochs "$MAX_EPOCHS" \
    --num-games "$NUM_GAMES" \
    --output-dir "$ARRAY_OUTPUT_DIR" \
    --seed ${SEED} \
    --adaptive-loss \
    --device auto \
    --verbose

EXIT_STATUS=$?

# ========================================
# UPDATE SEED MANIFEST WITH COMPLETION
# ========================================
cat >> "$MANIFEST_FILE" << EOF

========================================
COMPLETION STATUS
========================================
End Time:            $(date)
Exit Status:         ${EXIT_STATUS}
Duration:            ${SECONDS} seconds
========================================
EOF

echo "=========================================="
echo "TASK COMPLETED"
echo "=========================================="
echo "  Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "  Condition ID: $CONDITION_ID"
echo "  Seed: $SEED"
echo "  End Time: $(date)"
echo "  Exit Status: $EXIT_STATUS"
echo "  Duration: ${SECONDS} seconds"
echo "  Manifest: $MANIFEST_FILE"
echo "=========================================="

exit $EXIT_STATUS
