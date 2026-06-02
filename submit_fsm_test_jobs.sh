#!/bin/bash
# Submit FSM test jobs in chunks to comply with cluster array limit (1001)
# 
# Usage: TRAINING_JOB_ID=12345 bash submit_fsm_test_jobs.sh
#
# This script submits 12 job arrays, each handling up to 1000 tasks
# Total: 11,340 test tasks (420 models × 27 test conditions)

if [ -z "$TRAINING_JOB_ID" ]; then
    echo "ERROR: TRAINING_JOB_ID environment variable not set"
    echo "Usage: TRAINING_JOB_ID=12345 bash submit_fsm_test_jobs.sh"
    exit 1
fi

echo "=========================================="
echo "FSM Test Job Submission"
echo "=========================================="
echo "Training Job ID: ${TRAINING_JOB_ID}"
echo "Total Tasks: 11,340 (420 models × 27 test conditions)"
echo "Submitting in 12 batches (max 1000 tasks/batch)"
echo ""

# Array to store job IDs
declare -a JOB_IDS

# Submit 12 batches
for BATCH in {0..11}; do
    OFFSET=$((BATCH * 1000))
    
    # Calculate number of tasks for this batch
    if [ $BATCH -eq 11 ]; then
        # Last batch: only 340 tasks (11000-11339)
        ARRAY_SIZE="0-339"
        TASKS=340
    else
        # All other batches: 1000 tasks
        ARRAY_SIZE="0-999"
        TASKS=1000
    fi
    
    echo "Batch $((BATCH + 1))/12:"
    echo "  Task range: ${OFFSET}-$((OFFSET + TASKS - 1))"
    echo "  Array size: ${ARRAY_SIZE}"
    
    # Submit job
    JOB_ID=$(sbatch --parsable \
        --array=${ARRAY_SIZE} \
        --export=ALL,TRAINING_JOB_ID=${TRAINING_JOB_ID},TASK_OFFSET=${OFFSET} \
        run_fsm_experiment_test.sh)
    
    if [ $? -eq 0 ]; then
        JOB_IDS+=($JOB_ID)
        echo "  ✓ Submitted: Job ${JOB_ID}"
        # Add delay to avoid overwhelming SLURM controller (increased for stability)
        sleep 3
    else
        echo "  ✗ Failed to submit batch ${BATCH}"
        exit 1
    fi
    echo ""
done

echo "=========================================="
echo "All batches submitted successfully!"
echo "=========================================="
echo "Job IDs:"
for i in "${!JOB_IDS[@]}"; do
    echo "  Batch $((i + 1)): ${JOB_IDS[$i]}"
done
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all with: scancel ${JOB_IDS[@]}"
echo ""
echo "Results will be in:"
echo "  experiments/fsm_representation_test_<JOB_ID>/"
