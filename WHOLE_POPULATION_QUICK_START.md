# Whole Population Training Experiment - Quick Start Guide

This guide explains how to run the whole population training experiment (Section 5.5 of EXPERIMENT_DESIGN.md).

## Overview

**Hypothesis H8**: Agents trained against the entire opponent spectrum will display averaged behavior, with game-specific effects on convergence and generalization.

**Training**: 3 games × 5 seeds = 15 agents  
**Testing**: 15 agents × 3 games × 5 opponents = 225 test conditions

## Experiment Structure

### Training Phase
- **Opponent Spectrum**: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
- **Sampling**: Random uniform sampling from full spectrum each epoch
- **Agent Type**: Vanilla RL only (no ToM auxiliary tasks)
- **Games**: Prisoner's Dilemma, Hawk-Dove, Stag Hunt

### Testing Phase
- **Test Games**: All 3 games (including training game for completeness)
- **Test Opponents**: [0.1, 0.3, 0.5, 0.7, 0.9] (midpoints from generalization matrix)
- **Evaluation**: 20 sessions per opponent type

## Running the Experiment

### Step 1: Training Phase

Submit the training job array:

```bash
sbatch run_whole_population_train.sh
```

This will:
- Start 15 parallel training jobs (3 games × 5 seeds)
- Train each agent against full opponent spectrum [0.0-1.0]
- Save checkpoints to `experiments/whole_population_train_{JOB_ID}/`
- Each job runs for up to 3 days

**Monitor Training:**
```bash
# Check job status
squeue -u $USER | grep wp_train

# Check specific task output
tail -f slurm_logs/wp_train_wp_train_{JOB_ID}_{TASK_ID}.out

# Verify all checkpoints created
ls experiments/whole_population_train_{JOB_ID}/training/*/checkpoints/*.pth
```

### Step 2: Testing Phase

After training completes, note the training JOB_ID and submit testing:

```bash
# Set training job ID
export TRAINING_JOB_ID=12345  # Replace with actual training job ID

# Submit testing array
TRAINING_JOB_ID=${TRAINING_JOB_ID} sbatch run_whole_population_test.sh
```

This will:
- Start 225 parallel testing jobs (15 models × 15 test conditions)
- Each job loads a trained checkpoint and tests on one game+opponent combination
- Save results to `experiments/whole_population_test_{JOB_ID}/`
- Each job runs for up to 2 hours

**Monitor Testing:**
```bash
# Check job status
squeue -u $USER | grep wp_test

# Check progress
ls experiments/whole_population_test_{JOB_ID}/testing/*/results/*.csv | wc -l
# Should eventually show 225 files

# View specific test result
tail slurm_logs/wp_test_wp_test_{JOB_ID}_{TASK_ID}.out
```

## Task ID Mapping

### Training (0-14)
```
Task 0-4:   Prisoner's Dilemma (seeds 42, 52, 62, 72, 82)
Task 5-9:   Hawk-Dove (seeds 42, 52, 62, 72, 82)
Task 10-14: Stag Hunt (seeds 42, 52, 62, 72, 82)
```

### Testing (0-224)
```
MODEL_ID = TASK_ID / 15
TEST_ID  = TASK_ID % 15

Test condition breakdown (per model):
  Test 0-4:   Prisoner's Dilemma × opponents [0.1, 0.3, 0.5, 0.7, 0.9]
  Test 5-9:   Hawk-Dove × opponents [0.1, 0.3, 0.5, 0.7, 0.9]
  Test 10-14: Stag Hunt × opponents [0.1, 0.3, 0.5, 0.7, 0.9]
```

## Configuration

### Default Settings
- **Config**: `config/whole_population_config.json`
- **Max Epochs**: 1000
- **Games per Partner**: 100
- **Learning Rate**: 0.001
- **Convergence Threshold**: 1e-6
- **Patience**: 50 epochs

### Custom Configuration

Override in SLURM scripts:
```bash
# Training
MAX_EPOCHS=1500 NUM_GAMES=150 WP_CONFIG=my_config.json sbatch run_whole_population_train.sh

# Testing
WP_CONFIG=my_config.json TRAINING_JOB_ID=12345 sbatch run_whole_population_test.sh
```

## Output Structure

```
experiments/
├── whole_population_train_{TRAIN_JOB_ID}/
│   ├── training/
│   │   ├── whole_population_task_0_{timestamp}/
│   │   │   ├── checkpoints/
│   │   │   │   └── prisoners-dilemma_final_checkpoint.pth
│   │   │   ├── results/
│   │   │   │   ├── training_task_0_metrics.csv
│   │   │   │   ├── training_task_0_report.json
│   │   │   │   └── training_task_0_results.pkl
│   │   │   └── logs/
│   │   ├── whole_population_task_1_{timestamp}/
│   │   └── ...
│   └── seed_manifests/
│       └── MASTER_TRAINING_REGISTRY.csv
│
└── whole_population_test_{TEST_JOB_ID}/
    └── testing/
        ├── whole_population_task_0_{timestamp}/
        │   └── results/
        │       ├── eval_model_0_task_0_report.json
        │       ├── eval_model_0_prisoners-dilemma_on_prisoners-dilemma_opp_0.1.csv
        │       └── eval_model_0_task_0_results.pkl
        ├── whole_population_task_1_{timestamp}/
        └── ...
```

## Expected Results (H8 Predictions)

### Prisoner's Dilemma
- **Fast convergence** to defection policy
- **Low policy entropy** (deterministic)
- **Poor cross-game generalization**

### Stag Hunt 
- **Slower convergence** (multiple equilibria)
- **Higher policy entropy** (mixed strategy)
- **Better cross-game generalization**

### Hawk-Dove
- **Intermediate** convergence and generalization

## Analysis Scripts

After both phases complete, analyze results:

```bash
cd experiments/analysis_scripts

# Create analysis for whole population experiment
python analyze_whole_population_results.py \
    --train-dir ../whole_population_train_{TRAIN_JOB_ID} \
    --test-dir ../whole_population_test_{TEST_JOB_ID}
```

Expected outputs (see Section 9.7 of EXPERIMENT_DESIGN.md):
- Convergence speed comparison (PD vs HD vs SH)
- Policy entropy evolution during training
- 3×3 cross-game transfer matrix
- Comparison to segmented training results
- Opponent-type performance profiles

## Troubleshooting

### Training Issues

**Problem**: Checkpoint not found after training
```bash
# Check if training completed
sacct -j {TRAIN_JOB_ID} --format=JobID,State,ExitCode

# Check output logs
grep "Checkpoint" slurm_logs/wp_train_*_{TASK_ID}.out

# Verify checkpoint manually
ls experiments/whole_population_train_{TRAIN_JOB_ID}/training/*/checkpoints/
```

**Problem**: Training doesn't converge (1000 epochs reached)
- Check convergence threshold in config (may need relaxation)
- Verify learning rate is appropriate
- Check loss trajectories in training logs

### Testing Issues

**Problem**: Cannot find training directory
```bash
# Verify training job ID
ls experiments/ | grep whole_population_train

# Update TRAINING_JOB_ID
export TRAINING_JOB_ID=<correct_id>
```

**Problem**: Some test tasks fail
```bash
# Identify failed tasks
sacct -j {TEST_JOB_ID} --format=JobID,State,ExitCode | grep FAILED

# Rerun specific failed tasks manually
sbatch --array=42,57,103 run_whole_population_test.sh
```

## Comparison to Generalization Matrix

This experiment complements the generalization matrix (Section 5.3):

| Aspect | Generalization Matrix | Whole Population |
|--------|----------------------|------------------|
| Training opponents | Narrow range (e.g., [0.2, 0.4]) | Full spectrum [0.0-1.0] |
| Number of conditions | 15 (3 games × 5 ranges) | 3 (games only) |
| Expected within-task performance | High | Moderate (averaged) |
| Expected generalization | Variable by range | Uniform across opponents |
| Agent types tested | Vanilla + Proto-ToM | Vanilla only (for now) |

## Next Steps

1. **Run full experiment**:
   ```bash
   sbatch run_whole_population_train.sh
   # Wait for completion, note JOB_ID
   TRAINING_JOB_ID={JOB_ID} sbatch run_whole_population_test.sh
   ```

2. **Monitor progress** through SLURM logs and output directories

3. **Analyze results** using Section 9.7 analysis plan

4. **Compare** to generalization matrix results to test H8 predictions

5. **Document findings** in results report

## Questions or Issues?

See:
- `EXPERIMENT_DESIGN.md` Section 5.5 (Experiment design)
- `EXPERIMENT_DESIGN.md` Section 9.7 (Analysis plan)
- `config/whole_population_config.json` (Configuration details)
- `.github/copilot-instructions.md` (Code architecture)
