# Quick Start Guide: Network Input Modification Experiments

## Overview
This guide covers running experiments with the **modified network architecture** that includes opponent's previous action as input (6 elements instead of 5).

**Modification Details:**
- Input size: 5 → 6 elements
- New element: opponent's previous action (-1.0 for first trial, 0.0/1.0 for COOPERATE/DEFECT)
- All configuration files updated
- All SLURM scripts verified

---

## 1. Generalization Matrix Experiment

### Training Phase (75 tasks: 15 conditions × 5 seeds)
```bash
sbatch run_generalization_matrix_train.sh
```

**What it does:**
- Trains 75 models (15 training conditions × 5 random seeds)
- Each model trained on one game + opponent range combination
- Saves checkpoints for testing phase
- Uses `config/generalization_matrix_config.json` with `input_size: 6`

**Monitor progress:**
```bash
# Check running jobs
squeue -u $USER

# Check logs
tail -f slurm_logs/train_gen_matrix_train_<job_id>_*.out

# Check training progress
ls experiments/generalization_matrix_train_<job_id>/training/
```

### Testing Phase (1050 tasks: 75 models × 14 test conditions each)
After training completes, get the job ID and run:

```bash
# Get the training job ID from your email or squeue
TRAINING_JOB_ID=<your_train_job_id>

# Run testing (split into two jobs due to array size limit)
TRAINING_JOB_ID=$TRAINING_JOB_ID sbatch run_generalization_matrix_test.sh        # Tasks 0-999
TRAINING_JOB_ID=$TRAINING_JOB_ID sbatch run_generalization_matrix_test_part2.sh  # Tasks 1000-1049
```

**Monitor testing:**
```bash
tail -f slurm_logs/test_gen_matrix_test*_*.out
```

---

## 2. Whole Population Experiment

### Training Phase (15 tasks: 3 games × 5 seeds)
```bash
sbatch run_whole_population_train.sh
```

**What it does:**
- Trains 15 vanilla RL agents (3 games × 5 seeds)
- Each agent trained against full opponent spectrum [0.0, 0.1, ..., 1.0]
- Saves checkpoints for cross-game generalization testing
- Uses `config/whole_population_config.json` with `input_size: 6`

**Monitor progress:**
```bash
tail -f slurm_logs/wp_train_wp_train_<job_id>_*.out
ls experiments/whole_population_train_<job_id>/training/
```

### Testing Phase (225 tasks: 15 models × 15 test conditions)
```bash
TRAINING_JOB_ID=<your_train_job_id> sbatch run_whole_population_test.sh
```

**What it tests:**
- Each trained model tested on all 3 games × 5 opponents = 15 conditions
- Measures within-game and cross-game generalization

**Monitor testing:**
```bash
tail -f slurm_logs/wp_test_wp_test_<job_id>_*.out
```

---

## 3. Local Testing (Before Cluster Submission)

### Quick smoke test
```bash
# Activate environment
source .venv/Scripts/activate  # Windows
# or
source .venv/bin/activate      # Linux

# Set Python path
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Run quick test (10 epochs, small network)
python main_experiment.py \
  --config config/quick_test_config.json \
  --train-game prisoners-dilemma \
  --opponents 0.5 \
  --num-games 10 \
  --max-epochs 10
```

### Verify network input
```bash
python test_input_modification.py
```

Expected output:
```
✓ State size is correct (6 elements)
✓ Initial state has opponent_prev_action = -1.0
✓ Opponent's actions correctly recorded
✓ Network successfully processes 6-element input
ALL TESTS PASSED! ✓
```

---

## 4. Analysis After Experiments Complete

### Compare to Baseline (5-element input)
Once both old and new experiments are complete:

```bash
python experiments/analysis_scripts/compare_input_modifications.py \
  --baseline-train <old_train_job_id> \
  --baseline-test <old_test_job_id> \
  --modified-train <new_train_job_id> \
  --modified-test <new_test_job_id> \
  --experiment-type {generalization-matrix|whole-population} \
  --output-dir comparison_results
```

**Outputs:**
- `training_convergence_comparison.png` - Loss curves comparison
- `test_performance_comparison.png` - Test metrics comparison
- `generalization_comparison.csv` - Detailed generalization scores
- `comparison_summary.txt` - Summary report

### Standard analysis (same as before)
```bash
# For generalization matrix
python experiments/analysis_scripts/analyze_generalization_matrix.py \
  --train-job-id <train_job_id> \
  --test-job-id <test_job_id>

# For whole population
python experiments/analysis_scripts/analyze_whole_population_generalization.py \
  --train-job-id <train_job_id> \
  --test-job-id <test_job_id>
```

---

## 5. Key Differences from Previous Experiments

### What Changed:
1. **State vector**: Now includes opponent's previous action (6 elements instead of 5)
2. **First trial**: opponent_prev_action = -1.0 (indicator for no history)
3. **Subsequent trials**: opponent_prev_action = 0.0 (COOPERATE) or 1.0 (DEFECT)

### What Stayed the Same:
- SLURM scripts (no changes needed)
- Experiment structure and workflow
- Analysis pipeline (with added comparison capability)
- Checkpoint format and loading

### Expected Benefits:
- **Better opponent modeling**: Network can condition on opponent's last action
- **Improved generalization**: More informative state representation
- **Higher prediction accuracy**: Direct access to opponent behavior history

---

## 6. Troubleshooting

### Issue: Jobs fail with input size mismatch
**Solution:** Verify config files have `"input_size": 6`
```bash
grep -r "input_size" config/
```

### Issue: Old checkpoints won't load
**Cause:** Old checkpoints have 5-element input, new code expects 6-element
**Solution:** Keep old experiments separate; use old code version for old checkpoints

### Issue: No training logs generated
**Solution:** Check SLURM output logs for errors
```bash
cat slurm_logs/*_<job_id>_*.err
```

---

## 7. File Locations

### Configuration Files (all updated with input_size: 6)
- `config/default_config.json`
- `config/quick_test_config.json`
- `config/generalization_matrix_config.json`
- `config/whole_population_config.json`

### SLURM Scripts (verified compatible)
- `run_generalization_matrix_train.sh`
- `run_generalization_matrix_test.sh`
- `run_generalization_matrix_test_part2.sh`
- `run_whole_population_train.sh`
- `run_whole_population_test.sh`

### Analysis Scripts
- `experiments/analysis_scripts/compare_input_modifications.py` (NEW)
- `experiments/analysis_scripts/analyze_generalization_matrix.py`
- `experiments/analysis_scripts/analyze_whole_population_generalization.py`

### Tests
- `test_input_modification.py` - Comprehensive test suite

---

## 8. Example Workflow

```bash
# 1. Local testing
python test_input_modification.py

# 2. Submit training
sbatch run_whole_population_train.sh
# Note the job ID (e.g., 123456)

# 3. Wait for training to complete (check email or squeue)

# 4. Submit testing
TRAINING_JOB_ID=123456 sbatch run_whole_population_test.sh
# Note the test job ID (e.g., 123457)

# 5. Wait for testing to complete

# 6. Run comparison (if you have baseline results)
python experiments/analysis_scripts/compare_input_modifications.py \
  --baseline-train 888509 \
  --baseline-test 902267 \
  --modified-train 123456 \
  --modified-test 123457 \
  --experiment-type whole-population

# 7. Review results
ls comparison_results/
```

---

## Questions or Issues?

See `DEPLOYMENT_CHECKLIST.md` for detailed verification steps.
