#!/bin/bash -l
#SBATCH -o ./slurm_logs/fsm_train_small_%x_%A_%a.out
#SBATCH -e ./slurm_logs/fsm_train_small_%x_%A_%a.err
#SBATCH -D ./
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nitay.alon@tuebingen.mpg.de
#SBATCH --time=24:00:00
#SBATCH --job-name=fsm_train_small
#SBATCH --array=0-29

# FSM REPRESENTATION EXPERIMENT - TRAINING PHASE (SMALL NETWORKS, H=4)
# 30 tasks: 3 games x 10 seeds
# Each task trains 5 agents (one per opponent: 0.1, 0.3, 0.5, 0.7, 0.9)
# Results saved to opp_X.X/ subdirectories within each task output dir

module purge
module load singularity

export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"
export CONTAINER_PATH=/ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p slurm_logs
mkdir -p experiments

GAMES=("prisoners-dilemma" "stag-hunt" "hawk-dove")
SEED_VALUES=(42 123 456 789 1011 1213 1415 1617 1819 2021)

# Task ID -> (game, seed)
GAME_IDX=$((SLURM_ARRAY_TASK_ID / 10))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 10))

GAME=${GAMES[$GAME_IDX]}
SEED=${SEED_VALUES[$SEED_IDX]}
HIDDEN_SIZE=4
INPUT_CONDITION="no_game"

ARRAY_OUTPUT_DIR="experiments/fsm_train_small_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${ARRAY_OUTPUT_DIR}/task_${SLURM_ARRAY_TASK_ID}"

echo "=========================================="
echo "FSM TRAINING (SMALL H=4) - Task ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="
echo "Game:            ${GAME}"
echo "Hidden size:     ${HIDDEN_SIZE}"
echo "Input condition: ${INPUT_CONDITION}"
echo "Seed:            ${SEED}"
echo "Opponents:       0.1 0.3 0.5 0.7 0.9"
echo ""

time singularity exec ${CONTAINER_PATH} python -u run_fsm_experiment.py \
    --mode train \
    --game ${GAME} \
    --opponents 0.1 0.3 0.5 0.7 0.9 \
    --hidden-size ${HIDDEN_SIZE} \
    --input-condition ${INPUT_CONDITION} \
    --seed ${SEED} \
    --n-episodes 10000 \
    --save-checkpoint \
    --save-trajectories \
    --save-every-nth-episode 100 \
    --output-dir "${ARRAY_OUTPUT_DIR}/task_${SLURM_ARRAY_TASK_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
