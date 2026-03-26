#!/bin/bash
# Verify checkpoint discovery logic before running full test array

TRAINING_JOB_ID="${1:-888509}"
TRAINING_DIR="experiments/generalization_matrix_train_${TRAINING_JOB_ID}"

echo "=========================================="
echo "CHECKPOINT PATH VERIFICATION"
echo "Training Job ID: ${TRAINING_JOB_ID}"
echo "Training Directory: ${TRAINING_DIR}"
echo "=========================================="

if [ ! -d "$TRAINING_DIR" ]; then
    echo "❌ ERROR: Training directory not found: $TRAINING_DIR"
    exit 1
fi

echo "✓ Training directory found"
echo ""

# Test a few sample conditions
TEST_CONDITIONS=(
    "0 0"  # condition 0, seed 0
    "0 1"  # condition 0, seed 1
    "7 2"  # condition 7, seed 2
    "14 4" # condition 14, seed 4 (last condition, last seed)
)

SUCCESS_COUNT=0
TOTAL_TESTS=${#TEST_CONDITIONS[@]}

for test in "${TEST_CONDITIONS[@]}"; do
    read -r TRAINING_CONDITION_ID SEED_ID <<< "$test"
    
    echo "Testing: Condition ${TRAINING_CONDITION_ID}, Seed ${SEED_ID}"
    echo "----------------------------------------"
    
    # Find checkpoint (same logic as test scripts)
    CONDITION_DIR="${TRAINING_DIR}/training/condition_${TRAINING_CONDITION_ID}_seed_${SEED_ID}"
    if [ ! -d "$CONDITION_DIR" ]; then
        echo "  ❌ Condition directory not found: $CONDITION_DIR"
        echo ""
        continue
    fi
    echo "  ✓ Condition directory: $CONDITION_DIR"
    
    # Find experiment directory
    EXP_DIR=$(find "$CONDITION_DIR" -maxdepth 1 -type d -name "generalization_matrix_*" 2>/dev/null | head -n 1)
    if [ -z "$EXP_DIR" ] || [ ! -d "$EXP_DIR" ]; then
        echo "  ❌ Experiment directory not found in $CONDITION_DIR"
        echo ""
        continue
    fi
    echo "  ✓ Experiment directory: $EXP_DIR"
    
    # Find checkpoint file
    CHECKPOINT_PATH=$(find "$EXP_DIR/checkpoints" -name "*_final_checkpoint.pth" 2>/dev/null | head -n 1)
    if [ -z "$CHECKPOINT_PATH" ] || [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "  ❌ Checkpoint not found in $EXP_DIR/checkpoints"
        echo ""
        continue
    fi
    
    # Get checkpoint size
    CHECKPOINT_SIZE=$(du -h "$CHECKPOINT_PATH" | cut -f1)
    echo "  ✓ Checkpoint found: $(basename $CHECKPOINT_PATH)"
    echo "    Path: $CHECKPOINT_PATH"
    echo "    Size: $CHECKPOINT_SIZE"
    
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    echo ""
done

echo "=========================================="
echo "VERIFICATION SUMMARY"
echo "=========================================="
echo "Successful: ${SUCCESS_COUNT}/${TOTAL_TESTS}"
echo ""

if [ "$SUCCESS_COUNT" -eq "$TOTAL_TESTS" ]; then
    echo "✅ All test cases passed!"
    echo "✅ Checkpoint discovery logic is working correctly"
    echo ""
    echo "You can now submit the full test jobs:"
    echo "  TRAINING_JOB_ID=${TRAINING_JOB_ID} sbatch run_generalization_matrix_test.sh"
    echo "  TRAINING_JOB_ID=${TRAINING_JOB_ID} sbatch run_generalization_matrix_test_part2.sh"
    exit 0
else
    echo "❌ Some test cases failed"
    echo "❌ Please fix issues before submitting full test array"
    exit 1
fi
