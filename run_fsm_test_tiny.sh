#!/bin/bash -l
#SBATCH -o ./slurm_logs/fsm_test_tiny_%x_%A_%a.out
#SBATCH -e ./slurm_logs/fsm_test_tiny_%x_%A_%a.err
#SBATCH -D ./
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nitay.alon@tuebingen.mpg.de
#SBATCH --time=24:00:00
#SBATCH --job-name=fsm_test_tiny
#SBATCH --array=0-29

# FSM REPRESENTATION EXPERIMENT - TESTING PHASE (TINY NETWORKS, H=2)
# 30 tasks: 3 games x 10 seeds (mirrors run_fsm_train_tiny.sh)
# Each task evaluates the 5 trained checkpoints (one per opponent: 0.1, 0.3, 0.5, 0.7, 0.9)
# from TRAIN_DIR/task_N/opp_X.X/checkpoint.pth on all other 14 of the 15
# (game, opponent) combinations (--exclude-train-combo skips the trained-on combo).

module purge
module load singularity

export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"
export CONTAINER_PATH=/ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p slurm_logs

# Training run to evaluate (update if re-trained)
TRAIN_DIR="experiments/fsm_train_tiny_946438"
TEST_OUTPUT_DIR="experiments/fsm_test_tiny_${SLURM_ARRAY_JOB_ID}"

OPPONENTS=(0.1 0.3 0.5 0.7 0.9)

TASK_DIR="${TRAIN_DIR}/task_${SLURM_ARRAY_TASK_ID}"
if [ ! -d "$TASK_DIR" ]; then
    echo "ERROR: Training task directory not found: $TASK_DIR"
    exit 1
fi

echo "=========================================="
echo "FSM TESTING (TINY H=2) - Task ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="
echo "Training task dir: ${TASK_DIR}"
echo ""

for OPP in "${OPPONENTS[@]}"; do
    CHECKPOINT="${TASK_DIR}/opp_${OPP}/checkpoint.pth"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "WARNING: checkpoint not found: ${CHECKPOINT} - skipping"
        continue
    fi

    OUT_DIR="${TEST_OUTPUT_DIR}/task_${SLURM_ARRAY_TASK_ID}/opp_${OPP}"
    mkdir -p "${OUT_DIR}"

    echo "------------------------------------------"
    echo "Testing checkpoint: ${CHECKPOINT}"
    echo "------------------------------------------"

    time singularity exec ${CONTAINER_PATH} python -u run_fsm_experiment.py \
        --mode test \
        --checkpoint-path "${CHECKPOINT}" \
        --test-games prisoners-dilemma stag-hunt hawk-dove \
        --test-opponents 0.1 0.3 0.5 0.7 0.9 \
        --exclude-train-combo \
        --n-test-episodes 100 \
        --save-trajectories \
        --save-every-nth-episode 10 \
        --output-dir "${OUT_DIR}"
done

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
