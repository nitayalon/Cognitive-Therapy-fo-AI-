# Deployment Checklist: Network Input Modification (5→6 Elements)

## Overview
**Modification**: Added opponent's previous action to network input
- **Old input size**: 5 elements (payoff_matrix + round_number)
- **New input size**: 6 elements (payoff_matrix + round_number + opponent_prev_action)
- **First trial**: opponent_prev_action = -1.0 (no previous action)
- **Subsequent trials**: opponent_prev_action = 0.0 (COOPERATE) or 1.0 (DEFECT)

## ✅ Code Changes Completed

### 1. Core Architecture
- [x] `src/cognitive_therapy_ai/games.py` - Updated `get_state_vector()` and `get_state_size()`
- [x] `src/cognitive_therapy_ai/network.py` - Updated documentation for 6-element input
- [x] `src/cognitive_therapy_ai/config.py` - Updated `NetworkConfig.input_size` default to 6

### 2. Configuration Files
- [x] `config/default_config.json` - Added `"input_size": 6`
- [x] `config/quick_test_config.json` - Added `"input_size": 6`
- [x] `config/generalization_matrix_config.json` - Added `"input_size": 6`
- [x] `config/whole_population_config.json` - Added `"input_size": 6`

### 3. Testing & Validation
- [x] `test_input_modification.py` - Comprehensive test suite created
- [x] All unit tests pass
- [x] Integration test successful (training loop verified)
- [x] All game types tested (PD, HD, SH)

## ✅ SLURM Scripts Verification

### Generalization Matrix Experiment
- [x] `run_generalization_matrix_train.sh` - Uses `generalization_matrix_config.json` ✓
- [x] `run_generalization_matrix_test.sh` - Loads checkpoints, no hardcoded sizes ✓
- [x] `run_generalization_matrix_test_part2.sh` - Loads checkpoints, no hardcoded sizes ✓

### Whole Population Experiment
- [x] `run_whole_population_train.sh` - Uses `whole_population_config.json` ✓
- [x] `run_whole_population_test.sh` - Loads checkpoints, no hardcoded sizes ✓

### Other Scripts
- [x] No hardcoded `input_size=5` found in any `.sh` files ✓
- [x] All scripts rely on configuration files ✓

## ✅ Main Experiment Script
- [x] `main_experiment.py` - No hardcoded input sizes ✓
- [x] Uses config files for all network parameters ✓
- [x] Compatible with both training and testing modes ✓

## 📝 Deployment Instructions

### For Generalization Matrix Experiment:
```bash
# Training phase (75 tasks: 15 conditions × 5 seeds)
sbatch run_generalization_matrix_train.sh

# After training completes, get the job ID and run testing:
TRAINING_JOB_ID=<job_id> sbatch run_generalization_matrix_test.sh
TRAINING_JOB_ID=<job_id> sbatch run_generalization_matrix_test_part2.sh
```

### For Whole Population Experiment:
```bash
# Training phase (15 tasks: 3 games × 5 seeds)
sbatch run_whole_population_train.sh

# After training completes, get the job ID and run testing:
TRAINING_JOB_ID=<job_id> sbatch run_whole_population_test.sh
```

## 🔍 What to Monitor

### During Training:
1. **State vector shape**: Should be (6,) in logs
2. **Network input**: Verify 6-element tensors in forward pass
3. **Loss convergence**: Compare to baseline (5-element) experiments
4. **Checkpoint files**: Ensure they save correctly

### During Testing:
1. **Checkpoint loading**: Verify correct network architecture loaded
2. **Test performance metrics**: Compare to baseline experiments
3. **Opponent prediction accuracy**: May improve with opponent action input

## 🚨 Potential Issues & Solutions

### Issue 1: Old checkpoints incompatible
**Problem**: Trying to load old (5-element) checkpoints with new (6-element) code
**Solution**: 
- Old experiments are in separate directories
- New experiments will have different job IDs
- Keep old code version for analyzing old checkpoints if needed

### Issue 2: Config file not found
**Problem**: Script doesn't find updated config
**Solution**: Verify `$PYTHONPATH` includes project root in SLURM scripts

### Issue 3: State vector size mismatch
**Problem**: Error about input size mismatch
**Solution**: Check that config file has `"input_size": 6` and is being loaded

## 📊 Post-Deployment Analysis Plan

Once new experiments complete, use the comparison analysis script:
```bash
python experiments/analysis_scripts/compare_input_modifications.py \
  --baseline-train <old_train_job_id> \
  --baseline-test <old_test_job_id> \
  --modified-train <new_train_job_id> \
  --modified-test <new_test_job_id> \
  --experiment-type {generalization-matrix|whole-population}
```

## ✅ Ready for Deployment
- [x] All code changes committed and pushed (commit: `f164b5e`)
- [x] All configuration files updated
- [x] SLURM scripts verified
- [x] Tests passing
- [x] Analysis comparison script prepared

**Status**: ✅ **READY FOR CLUSTER DEPLOYMENT**
