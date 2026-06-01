#!/bin/bash
# Complete FSM experiment submission: training + testing with dependencies
#
# Usage: bash submit_complete_fsm_experiment.sh
#
# This script:
# 1. Submits training job (420 tasks)
# 2. Submits 12 testing job arrays (11,340 tasks total) with dependency on training
# 3. All testing jobs wait for training to complete successfully

echo "=========================================="
echo "FSM Representation Experiment - Complete Submission"
echo "=========================================="
echo ""

# Step 1: Submit training job
echo "STEP 1: Submitting Training Job"
echo "  Tasks: 420 (42 conditions × 10 seeds)"
echo "  Time: 2 hours per task"
echo ""

TRAINING_JOB_ID=$(sbatch --parsable run_fsm_experiment_train.sh)

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to submit training job"
    exit 1
fi

echo "  ✓ Training job submitted: ${TRAINING_JOB_ID}"
echo ""

# Allow SLURM controller to process training job submission
sleep 2

# Step 2: Submit testing jobs with dependency
echo "STEP 2: Submitting Testing Jobs (with dependency on training)"
echo "  Total tasks: 11,340 (420 models × 27 test conditions)"
echo "  Batches: 12 (max 1000 tasks each)"
echo "  Dependency: afterok:${TRAINING_JOB_ID}"
echo ""

declare -a TEST_JOB_IDS

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
    
    echo "  Batch $((BATCH + 1))/12: Tasks ${OFFSET}-$((OFFSET + TASKS - 1))"
    
    # Submit with dependency on training job
    JOB_ID=$(sbatch --parsable \
        --dependency=afterok:${TRAINING_JOB_ID} \
        --array=${ARRAY_SIZE} \
        --export=ALL,TRAINING_JOB_ID=${TRAINING_JOB_ID},TASK_OFFSET=${OFFSET} \
        run_fsm_experiment_test.sh)
    
    if [ $? -eq 0 ]; then
        TEST_JOB_IDS+=($JOB_ID)
        echo "    ✓ Job ${JOB_ID}"
        # Add delay to avoid overwhelming SLURM controller
        sleep 1
    else
        echo "    ✗ Failed to submit batch ${BATCH}"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "ALL JOBS SUBMITTED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Training Job:"
echo "  Job ID: ${TRAINING_JOB_ID}"
echo "  Tasks: 420"
echo "  Output: experiments/fsm_representation_train_${TRAINING_JOB_ID}/"
echo ""
echo "Testing Jobs (will start after training completes):"
for i in "${!TEST_JOB_IDS[@]}"; do
    echo "  Batch $((i + 1)): ${TEST_JOB_IDS[$i]}"
done
echo "  Total testing tasks: 11,340"
echo ""
echo "Monitoring:"
echo "  squeue -u \$USER"
echo "  squeue -j ${TRAINING_JOB_ID}"
echo ""
echo "Cancel all:"
echo "  scancel ${TRAINING_JOB_ID} ${TEST_JOB_IDS[@]}"
echo ""
echo "Expected completion:"
echo "  Training: ~2-4 hours (depends on cluster load)"
echo "  Testing: ~30 min after training completes"
echo "=========================================="
