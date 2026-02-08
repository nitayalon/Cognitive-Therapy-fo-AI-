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
#SBATCH --array=0-15

# Generalization Matrix Experiment - SLURM Array Job
# 
# This script runs 16 parallel experiments (tasks 0-15):
#   Tasks 0-3:   Prisoner's Dilemma with opponent ranges [low, mid_low, mid_high, high]
#   Tasks 4-7:   Hawk-Dove with opponent ranges [low, mid_low, mid_high, high]
#   Tasks 8-11:  Stag-Hunt with opponent ranges [low, mid_low, mid_high, high]
#   Tasks 12-15: Battle-of-Sexes with opponent ranges [low, mid_low, mid_high, high]
#
# Each task trains on one game+opponent combination and tests on:
#   - Same game, same opponents (baseline)
#   - Same game, different opponents (3 other ranges)
#   - Different game, same opponents (3 other games)
#   - Different game, different opponents (cross-generalization samples)

module purge
module load singularity

export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"
export CONTAINER_PATH=/ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif

# Set Python path for module imports
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Create logs directory if it doesn't exist
mkdir -p slurm_logs
mkdir -p experiments

# Default experiment parameters (can be overridden via environment variables)
MAX_EPOCHS="${MAX_EPOCHS:-500}"
NUM_GAMES="${NUM_GAMES:-100}"
MATRIX_CONFIG="${MATRIX_CONFIG:-config/generalization_matrix_config.json}"
ARRAY_OUTPUT_DIR="experiments/generalization_matrix_${SLURM_ARRAY_JOB_ID}"

mkdir -p "${ARRAY_OUTPUT_DIR}"

echo "=========================================="
echo "GENERALIZATION MATRIX EXPERIMENT"
echo "=========================================="
echo "SLURM Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo "Experiment Configuration:"
echo "  Matrix config: $MATRIX_CONFIG"
echo "  Max epochs: $MAX_EPOCHS"
echo "  Games per session: $NUM_GAMES"
echo "  Output directory: $ARRAY_OUTPUT_DIR"
echo "  Task seed: $((42 + SLURM_ARRAY_TASK_ID))"
echo "=========================================="

# Run the experiment inside the singularity container
time singularity exec ${CONTAINER_PATH} python main_experiment.py \
    --experiment-mode generalization-matrix \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --matrix-config "$MATRIX_CONFIG" \
    --max-epochs "$MAX_EPOCHS" \
    --num-games "$NUM_GAMES" \
    --output-dir "$ARRAY_OUTPUT_DIR" \
    --seed $((42 + SLURM_ARRAY_TASK_ID)) \
    --adaptive-loss \
    --device auto \
    --verbose

EXIT_STATUS=$?

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID completed"
echo "End Time: $(date)"
echo "Exit Status: $EXIT_STATUS"
echo "=========================================="

exit $EXIT_STATUS
