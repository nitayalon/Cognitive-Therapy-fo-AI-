#!/bin/bash -l
#SBATCH -o ./slurm_logs/fsm_test_generalist_large_%x_%A_%a.out
#SBATCH -e ./slurm_logs/fsm_test_generalist_large_%x_%A_%a.err
#SBATCH -D ./
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nitay.alon@tuebingen.mpg.de
#SBATCH --time=24:00:00
#SBATCH --job-name=fsm_test_generalist_large
#SBATCH --array=0-29

# FSM REPRESENTATION EXPERIMENT - GENERALIST TESTING PHASE (LARGE NETWORKS, H=16)
# 30 tasks: 3 games x 10 seeds (mirrors run_fsm_train_generalist_large.sh)
# Each task evaluates the single generalist checkpoint at
# TRAIN_DIR/task_N/checkpoint.pth on all 3 games x 5 opponents (15 combos).
# Generalist checkpoints have config['opponent_coop'] = None, so
# --exclude-train-combo would be a no-op and is intentionally omitted.

module purge
module load singularity

export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"
export CONTAINER_PATH=/ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p slurm_logs

# Generalist training run to evaluate (update with the actual job ID once
# run_fsm_train_generalist_large.sh has completed)
TRAIN_DIR="experiments/fsm_train_generalist_large_REPLACE_WITH_JOB_ID"
TEST_OUTPUT_DIR="experiments/fsm_test_generalist_large_${SLURM_ARRAY_JOB_ID}"

TASK_DIR="${TRAIN_DIR}/task_${SLURM_ARRAY_TASK_ID}"
CHECKPOINT="${TASK_DIR}/checkpoint.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

OUT_DIR="${TEST_OUTPUT_DIR}/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "FSM GENERALIST TESTING (LARGE H=16) - Task ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT}"
echo ""

time singularity exec ${CONTAINER_PATH} python -u run_fsm_experiment.py \
    --mode test \
    --checkpoint-path "${CHECKPOINT}" \
    --test-games prisoners-dilemma stag-hunt hawk-dove \
    --test-opponents 0.1 0.3 0.5 0.7 0.9 \
    --n-test-episodes 100 \
    --save-trajectories \
    --save-every-nth-episode 10 \
    --output-dir "${OUT_DIR}"

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
