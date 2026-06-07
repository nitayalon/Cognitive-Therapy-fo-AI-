#!/bin/bash
# Aggregate training checkpoints from split jobs for testing

if [ -z "$SMALL_JOB" ] || [ -z "$MEDIUM_JOB" ] || [ -z "$LARGE_JOB" ]; then
    echo "ERROR: Job IDs not set"
    echo "Usage:"
    echo "  export SMALL_JOB=<job_id>"
    echo "  export MEDIUM_JOB=<job_id>"
    echo "  export LARGE_JOB=<job_id>"
    echo "  bash aggregate_checkpoints.sh"
    exit 1
fi

AGGREGATE_DIR="experiments/fsm_train_aggregated"
mkdir -p "${AGGREGATE_DIR}"

echo "========================================"
echo "Aggregating Training Checkpoints"
echo "========================================"
echo "Small job:  ${SMALL_JOB}"
echo "Medium job: ${MEDIUM_JOB}"
echo "Large job:  ${LARGE_JOB}"
echo ""
echo "Target directory: ${AGGREGATE_DIR}"
echo ""

# Create symbolic links to preserve original structure
task_counter=0

# Copy small networks (tasks 0-179)
echo "Copying small network checkpoints..."
for task_id in {0..179}; do
    src="experiments/fsm_train_small_${SMALL_JOB}/task_${task_id}"
    dest="${AGGREGATE_DIR}/task_${task_counter}"
    if [ -d "$src" ]; then
        ln -s "$(realpath $src)" "$dest" 2>/dev/null || cp -r "$src" "$dest"
    fi
    ((task_counter++))
done

# Copy medium networks (tasks 180-299)
echo "Copying medium network checkpoints..."
for task_id in {0..119}; do
    src="experiments/fsm_train_medium_${MEDIUM_JOB}/task_${task_id}"
    dest="${AGGREGATE_DIR}/task_${task_counter}"
    if [ -d "$src" ]; then
        ln -s "$(realpath $src)" "$dest" 2>/dev/null || cp -r "$src" "$dest"
    fi
    ((task_counter++))
done

# Copy large networks (tasks 300-419)
echo "Copying large network checkpoints..."
for task_id in {0..119}; do
    src="experiments/fsm_train_large_${LARGE_JOB}/task_${task_id}"
    dest="${AGGREGATE_DIR}/task_${task_counter}"
    if [ -d "$src" ]; then
        ln -s "$(realpath $src)" "$dest" 2>/dev/null || cp -r "$src" "$dest"
    fi
    ((task_counter++))
done

echo ""
echo "Aggregation complete!"
echo ""
echo "Checkpoint count:"
find "${AGGREGATE_DIR}" -name "checkpoint.pth" | wc -l
echo "(Should be 420)"
echo ""
echo "Use this directory for testing:"
echo "  export TRAINING_DIR=${AGGREGATE_DIR}"
