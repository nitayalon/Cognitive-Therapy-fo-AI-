#!/bin/bash
# Submit all FSM training jobs in parallel (split by network size)

echo "========================================"
echo "FSM Training - Parallel Submission"
echo "========================================"
echo "Total: 420 tasks split by network size"
echo ""

# Submit small networks (H=2,4) - 180 tasks - 4 hours
echo "1. Submitting SMALL networks (H=2,4):"
echo "   Tasks: 180"
echo "   Time: 4 hours"
SMALL_JOB=$(sbatch --parsable run_fsm_train_small.sh)
echo "   Job ID: ${SMALL_JOB}"
echo ""

sleep 2

# Submit medium networks (H=8) - 120 tasks - 8 hours
echo "2. Submitting MEDIUM networks (H=8):"
echo "   Tasks: 120"
echo "   Time: 8 hours"
MEDIUM_JOB=$(sbatch --parsable run_fsm_train_medium.sh)
echo "   Job ID: ${MEDIUM_JOB}"
echo ""

sleep 2

# Submit large networks (H=16,32) - 120 tasks - 12 hours
echo "3. Submitting LARGE networks (H=16,32):"
echo "   Tasks: 120"
echo "   Time: 12 hours"
LARGE_JOB=$(sbatch --parsable run_fsm_train_large.sh)
echo "   Job ID: ${LARGE_JOB}"
echo ""

echo "========================================"
echo "All training jobs submitted!"
echo "========================================"
echo "Job IDs:"
echo "  Small (H=2,4):   ${SMALL_JOB}"
echo "  Medium (H=8):    ${MEDIUM_JOB}"
echo "  Large (H=16,32): ${LARGE_JOB}"
echo ""
echo "Monitor with:"
echo "  squeue -j ${SMALL_JOB},${MEDIUM_JOB},${LARGE_JOB}"
echo ""
echo "Check checkpoints:"
echo "  find experiments/fsm_train_small_${SMALL_JOB} -name 'checkpoint.pth' | wc -l  # Should be 180"
echo "  find experiments/fsm_train_medium_${MEDIUM_JOB} -name 'checkpoint.pth' | wc -l  # Should be 120"
echo "  find experiments/fsm_train_large_${LARGE_JOB} -name 'checkpoint.pth' | wc -l  # Should be 120"
echo ""
echo "After all complete, save job IDs for testing:"
echo "export SMALL_JOB=${SMALL_JOB}"
echo "export MEDIUM_JOB=${MEDIUM_JOB}"
echo "export LARGE_JOB=${LARGE_JOB}"
