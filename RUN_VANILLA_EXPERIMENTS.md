# Running Vanilla RL Experiments

This script runs the full vanilla RL baseline experiments across all 16 training conditions.

## Overview

The vanilla RL agent only learns from rewards (no opponent modeling auxiliary tasks). This provides a baseline for comparison with proto-ToM agents.

## Training Conditions

**16 conditions = 4 games × 4 opponent ranges:**

Training conditions are defined in `config/generalization_matrix_config.json`:
- Task ID 0-15 maps to specific game + opponent range combinations
- Each task trains on ONE game + ONE opponent range
- Each task evaluates on ALL games × ALL opponent ranges (comprehensive testing)

### Games:
1. Prisoner's Dilemma (prisoners-dilemma)
2. Hawk-Dove (hawk-dove)  
3. Stag Hunt (stag-hunt)
4. Battle of Sexes (battle-of-sexes)

### Opponent Ranges:
1. **Low defection:** [0.1, 0.2, 0.3, 0.4]
2. **Mid-low defection:** [0.3, 0.4, 0.5, 0.6]
3. **Mid-high defection:** [0.5, 0.6, 0.7, 0.8]
4. **High defection:** [0.7, 0.8, 0.9, 1.0]

### Evaluation Coverage

For each trained agent (16 total), we test on:
1. **Baseline:** Same game, same opponents (1 condition)
2. **Same game, different opponents:** Same game, 3 other opponent ranges (3 conditions)
3. **Different game, same opponents:** 3 other games, same opponent range (3 conditions)
4. **Cross-generalization:** Other games + other opponents (selected combinations)

**Total evaluations per agent:** ~10-13 conditions  
**Total measurements:** 16 agents × ~12 conditions = **~192 evaluation conditions**

## Usage

### Submit Full Experiment Array

```bash
sbatch run_vanilla_rl_experiment.sh
```

This submits 16 jobs (tasks 0-15), running up to 8 simultaneously.

### Check Job Status

```bash
squeue -u $USER
```

### View Logs

```bash
# See experiment progress
tail -f slurm_logs/vanilla_rl_exp_<JOB_ID>_<TASK_ID>.out

# Check for errors
tail -f slurm_logs/vanilla_rl_exp_<JOB_ID>_<TASK_ID>.err
```

## Output Structure

```
experiments/vanilla_rl_array_<JOB_ID>_<TIMESTAMP>/
├── vanilla_matrix_task0/
│   ├── checkpoints/
│   │   └── network_epoch_*.pth
│   ├── logs/
│   │   ├── experiment.log
│   │   ├── eval_baseline/
│   │   ├── eval_same_game_mid_low/
│   │   ├── eval_same_game_mid_high/
│   │   ├── eval_same_game_high/
│   │   ├── eval_hawk-dove_same_opponents/
│   │   ├── eval_stag-hunt_same_opponents/
│   │   ├── eval_battle-of-sexes_same_opponents/
│   │   └── eval_<game>_<opponent_range>/  # Cross-generalization
│   ├── plots/
│   ├── results/
│   │   └── matrix_results.pkl
│   └── experiment_config.json
├── vanilla_matrix_task1/
│   └── ...
├── ...
└── vanilla_matrix_task15/
    └── ...
```

Each task generates comprehensive evaluation data across all test conditions.

## Configuration

Current settings in `run_vanilla_rl_experiment.sh`:
- **Max epochs:** 10,000
- **Games per opponent:** 100
- **Memory:** 32GB
- **CPUs:** 4 per task
- **Time limit:** 3 days
- **Partition:** highmem (can change to gpu if needed)
- **Concurrent tasks:** 8 (adjust with `-a 0-15%N` where N=concurrent)

## Agent Configuration

- **Agent type:** `vanilla`
- **Loss function:** VanillaRLLoss (RL + value loss only)
- **No opponent modeling:** Alpha = 0.0 (no auxiliary tasks)
- **Architecture:** Same GameLSTM as proto-ToM for fair comparison

## Modification Options

### Change Partition
```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
```

### Adjust Concurrency
```bash
#SBATCH -a 0-15%4  # Run 4 at a time instead of 8
```

### Change Training Parameters
Edit the python command in the script:
```bash
--max-epochs 5000 \     # Fewer epochs
--num-games 50 \        # Shorter sessions
```

### Run Single Task for Testing
```bash
sbatch --array=0 run_vanilla_rl_experiment.sh  # Just task 0 (pd_low_defect)
```

## Next Steps After Completion

1. **Compare with Proto-ToM:** Run corresponding proto-tom experiments
2. **Analyze Results:** Use analysis scripts in `experiments/analysis_scripts/`
3. **Generate Reports:** See EXPERIMENT_DESIGN.md Section 9 for analysis plan

## Troubleshooting

### Job Failed
Check error log:
```bash
cat slurm_logs/vanilla_rl_exp_<JOB_ID>_<TASK_ID>.err
```

### Out of Memory
Increase memory allocation:
```bash
#SBATCH --mem=64G
```

### Timeout
Increase time limit or reduce epochs:
```bash
#SBATCH --time=5-00:00:00
```

## Expected Runtime

Per task:
- **Fast convergence:** 2-6 hours
- **Slow convergence:** 12-24 hours
- **Worst case:** 48-72 hours (with 10k epoch limit)

Total wall time with 8 concurrent: ~1-3 days

## Validation

After jobs complete, validate outputs:
```bash
# Check all tasks completed
ls experiments/vanilla_rl_array_*/*/experiment_config.json | wc -l
# Should show 16

# Check for training results
ls experiments/vanilla_rl_array_*/*/results/*.pkl | wc -l
# Should show 16+
```
