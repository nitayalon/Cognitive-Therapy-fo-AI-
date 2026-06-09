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
#SBATCH --array=0-179

# FSM REPRESENTATION EXPERIMENT - TRAINING PHASE (SMALL NETWORKS)
# 180 tasks: H={2,4} networks only
# Breakdown:
#   - no_game H=2: 60 tasks (3 games × 2 opponents × 10 seeds)
#   - no_game H=4: 60 tasks
#   - game_tag H=4: 60 tasks

module purge
module load singularity

export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"
export CONTAINER_PATH=/ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

mkdir -p slurm_logs
mkdir -p experiments

# Nested array calculation
NUM_SEEDS=10
SEED_VALUES=(42 123 456 789 1011 1213 1415 1617 1819 2021)

# Task mapping for small networks
# Tasks 0-59: no_game H=2
# Tasks 60-119: no_game H=4
# Tasks 120-179: game_tag H=4

GAMES=("prisoners-dilemma" "stag-hunt" "hawk-dove")
OPPONENTS=(0.1 0.7)

if [ $SLURM_ARRAY_TASK_ID -lt 60 ]; then
    # no_game H=2 (tasks 0-59)
    INPUT_CONDITION="no_game"
    HIDDEN_SIZE=2
    LOCAL_TASK=$SLURM_ARRAY_TASK_ID
    GAME_IDX=$((LOCAL_TASK / 20))
    OPP_IDX=$(((LOCAL_TASK / 10) % 2))
    SEED_ID=$((LOCAL_TASK % 10))
elif [ $SLURM_ARRAY_TASK_ID -lt 120 ]; then
    # no_game H=4 (tasks 60-119)
    INPUT_CONDITION="no_game"
    HIDDEN_SIZE=4
    LOCAL_TASK=$((SLURM_ARRAY_TASK_ID - 60))
    GAME_IDX=$((LOCAL_TASK / 20))
    OPP_IDX=$(((LOCAL_TASK / 10) % 2))
    SEED_ID=$((LOCAL_TASK % 10))
else
    # game_tag H=4 (tasks 120-179)
    INPUT_CONDITION="game_tag"
    HIDDEN_SIZE=4
    LOCAL_TASK=$((SLURM_ARRAY_TASK_ID - 120))
    GAME_IDX=$((LOCAL_TASK / 20))
    OPP_IDX=$(((LOCAL_TASK / 10) % 2))
    SEED_ID=$((LOCAL_TASK % 10))
fi

GAME=${GAMES[$GAME_IDX]}
OPPONENT=${OPPONENTS[$OPP_IDX]}
SEED=${SEED_VALUES[$SEED_ID]}

# Output directory
ARRAY_OUTPUT_DIR="experiments/fsm_train_small_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${ARRAY_OUTPUT_DIR}"
mkdir -p "${ARRAY_OUTPUT_DIR}/task_${SLURM_ARRAY_TASK_ID}"

echo "=========================================="
echo "FSM TRAINING (SMALL) - Task ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="
echo "Game: ${GAME}"
echo "Opponent: ${OPPONENT}"
echo "Hidden size: ${HIDDEN_SIZE}"
echo "Input condition: ${INPUT_CONDITION}"
echo "Seed: ${SEED}"
echo ""

# Run training with episode-level metrics (no full trajectories - will extract FSM separately)
time singularity exec ${CONTAINER_PATH} python -u run_fsm_experiment.py \
    --mode train \
    --game ${GAME} \
    --opponent ${OPPONENT} \
    --hidden-size ${HIDDEN_SIZE} \
    --input-condition ${INPUT_CONDITION} \
    --seed ${SEED} \
    --n-episodes 10000 \
    --save-checkpoint \
    --save-trajectories \
    --save-every-nth-episode 100 \
    --output-dir "${ARRAY_OUTPUT_DIR}/task_${SLURM_ARRAY_TASK_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
