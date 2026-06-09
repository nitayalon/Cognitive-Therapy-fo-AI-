#!/bin/bash -l
#SBATCH -o ./slurm_logs/fsm_train_%x_%A_%a.out
#SBATCH -e ./slurm_logs/fsm_train_%x_%A_%a.err
#SBATCH -D ./
#SBATCH --partition=highmem
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nitay.alon@tuebingen.mpg.de
#SBATCH --time=08:00:00
#SBATCH --job-name=fsm_repr_train
#SBATCH --array=0-419

# FSM REPRESENTATION EXPERIMENT - TRAINING PHASE
# 420 tasks: 42 conditions × 10 seeds
# Each task:
#   1. Train agent (10,000 episodes)
#   2. Extract FSM (L* algorithm)
#   3. Compute attribution
#   4. Validate best-response alignment
#   5. Save checkpoint for testing phase

module purge
module load singularity

export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"
export CONTAINER_PATH=/ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

mkdir -p slurm_logs
mkdir -p experiments

# Nested array calculation
NUM_CONDITIONS=42
NUM_SEEDS=10
SEED_VALUES=(42 123 456 789 1011 1213 1415 1617 1819 2021)

CONDITION_ID=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_ID=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
SEED=${SEED_VALUES[$SEED_ID]}

# Output directory
ARRAY_OUTPUT_DIR="experiments/fsm_representation_train_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${ARRAY_OUTPUT_DIR}"
mkdir -p "${ARRAY_OUTPUT_DIR}/condition_${CONDITION_ID}_seed_${SEED_ID}"
mkdir -p "${ARRAY_OUTPUT_DIR}/seed_manifests"

# Seed manifest (tracking)
MASTER_REGISTRY="${ARRAY_OUTPUT_DIR}/seed_manifests/MASTER_TRAINING_REGISTRY.csv"
if [ ! -f "$MASTER_REGISTRY" ]; then
    echo "array_job_id,array_task_id,condition_id,seed_id,seed,phase,start_time,node" > "$MASTER_REGISTRY"
fi
echo "${SLURM_ARRAY_JOB_ID},${SLURM_ARRAY_TASK_ID},${CONDITION_ID},${SEED_ID},${SEED},training,$(date -Iseconds),${SLURM_NODELIST}" >> "$MASTER_REGISTRY"

# Task manifest
MANIFEST_FILE="${ARRAY_OUTPUT_DIR}/seed_manifests/train_task_${SLURM_ARRAY_TASK_ID}_manifest.txt"
cat > "$MANIFEST_FILE" << EOF
========================================
FSM REPRESENTATION - TRAINING PHASE
========================================
Array Job ID:        ${SLURM_ARRAY_JOB_ID}
Array Task ID:       ${SLURM_ARRAY_TASK_ID}
Condition ID:        ${CONDITION_ID}
Seed ID:             ${SEED_ID}
Random Seed:         ${SEED}
Phase:               TRAINING + FSM EXTRACTION
----------------------------------------
Node:                ${SLURM_NODELIST}
Start Time:          $(date)
Output Directory:    ${ARRAY_OUTPUT_DIR}/condition_${CONDITION_ID}_seed_${SEED_ID}
========================================
EOF

echo "=========================================="
echo "FSM TRAINING - Task ${SLURM_ARRAY_TASK_ID}"
echo "Condition: ${CONDITION_ID}, Seed: ${SEED}"
echo "=========================================="

# Map condition ID to experiment parameters
# 42 conditions: 30 no_game + 12 game_tag
GAMES=("prisoners-dilemma" "stag-hunt" "hawk-dove")
OPPONENTS=(0.1 0.7)  # Training opponents
H_VALUES_NO_GAME=(2 4 8 16 32)
H_VALUES_GAME_TAG=(4 8)

if [ $CONDITION_ID -lt 30 ]; then
    # no_game conditions (0-29): 3 games × 2 opponents × 5 H values
    INPUT_CONDITION="no_game"
    GAME_IDX=$(( (CONDITION_ID / 10) % 3 ))
    OPP_IDX=$(( (CONDITION_ID / 5) % 2 ))
    H_IDX=$((CONDITION_ID % 5))
    GAME=${GAMES[$GAME_IDX]}
    OPPONENT=${OPPONENTS[$OPP_IDX]}
    HIDDEN_SIZE=${H_VALUES_NO_GAME[$H_IDX]}
else
    # game_tag conditions (30-41): 3 games × 2 opponents × 2 H values
    INPUT_CONDITION="game_tag"
    COND_OFFSET=$((CONDITION_ID - 30))
    GAME_IDX=$(( (COND_OFFSET / 4) % 3 ))
    OPP_IDX=$(( (COND_OFFSET / 2) % 2 ))
    H_IDX=$((COND_OFFSET % 2))
    GAME=${GAMES[$GAME_IDX]}
    OPPONENT=${OPPONENTS[$OPP_IDX]}
    HIDDEN_SIZE=${H_VALUES_GAME_TAG[$H_IDX]}
fi

echo "Game: ${GAME}"
echo "Opponent: ${OPPONENT}"
echo "Hidden size: ${HIDDEN_SIZE}"
echo "Input condition: ${INPUT_CONDITION}"

# Run training + FSM extraction
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
    --save-every-nth-episode 10 \
    --output-dir "${ARRAY_OUTPUT_DIR}/condition_${CONDITION_ID}_seed_${SEED_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
