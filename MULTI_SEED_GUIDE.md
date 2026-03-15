# Multi-Seed Nested Array Experiments - Quick Guide

## Overview

The generalization matrix experiments use a **nested SLURM array structure** to run multiple random seeds per training condition, enabling statistical testing and robust results.

## Structure

### Total Experiment Size
- **15 training conditions** (3 games × 5 opponent ranges)
- **5 random seeds** per condition (42, 52, 62, 72, 82)
- **2 agent types** (Vanilla RL, Proto-ToM)
- **Total: 150 agents** (15 × 5 × 2)

### Array Task Mapping

Each SLURM array task ID (0-74) encodes:
```
CONDITION_ID = TASK_ID / 5  (integer division, 0-14)
SEED_ID = TASK_ID % 5       (modulo, 0-4)
SEED = 42 + SEED_ID * 10    (actual random seed: 42,52,62,72,82)
```

**Examples:**
- Task 0: Condition 0 (PD + very_low opponents), seed 42
- Task 1: Condition 0 (PD + very_low opponents), seed 52
- Task 5: Condition 1 (PD + low opponents), seed 42
- Task 74: Condition 14 (SH + very_high opponents), seed 82

## Running Experiments

### Submit All Tasks
```bash
# Submit vanilla agents (75 tasks = 15 conditions × 5 seeds)
sbatch --array=0-74 run_generalization_matrix.sh

# Submit proto-ToM agents (requires modification to script or separate submission)
sbatch --array=0-74 run_generalization_matrix_proto_tom.sh
```

### Submit Specific Tasks
```bash
# Test with first few tasks
sbatch --array=0-4 run_generalization_matrix.sh

# Resubmit failed tasks
sbatch --array=23,47,51 run_generalization_matrix.sh
```

### Submit Specific Conditions (All Seeds)
```bash
# Run all 5 seeds for condition 0 (PD + very_low)
sbatch --array=0-4 run_generalization_matrix.sh

# Run all 5 seeds for condition 5 (HD + very_low)
sbatch --array=25-29 run_generalization_matrix.sh
```

## Seed Manifest System

Every run automatically creates documentation files:

### Master Registry
**Location:** `experiments/generalization_matrix_{JOB_ID}/seed_manifests/MASTER_SEED_REGISTRY.csv`

**Format:**
```csv
array_job_id,array_task_id,condition_id,seed_id,seed,start_time,node
834222,0,0,0,42,2026-03-15T10:23:45,nyx001
834222,1,0,1,43,2026-03-15T10:23:47,nyx002
...
```

### Individual Manifests
**Location:** `experiments/generalization_matrix_{JOB_ID}/seed_manifests/task_{TASK_ID}_manifest.txt`

**Contains:**
- Array job and task IDs
- Decoded condition and seed information
- Node and resource allocation
- Complete experiment configuration
- Start and end times
- Exit status

## Validation and Debugging

### Decode Task IDs
```bash
# Decode a specific task
python decode_seed_manifest.py --task-id 37

# Output:
# ============================================================
# TASK ID: 37
# ============================================================
#   Condition ID:    7 (of 15)
#   Seed ID:         2 (of 5)
#   Random Seed:     44
# ------------------------------------------------------------
#   Game:            hawk-dove
#   Opponent Range:  mid
# ============================================================
```

### View All Mappings
```bash
python decode_seed_manifest.py --all
```

### Validate Completed Jobs
```bash
# Check which tasks completed successfully
python decode_seed_manifest.py --validate \
    experiments/generalization_matrix_834222/seed_manifests/MASTER_SEED_REGISTRY.csv

# Output:
# ============================================================
# VALIDATING REGISTRY: experiments/generalization_matrix_834222/...
# ============================================================
# Total entries: 75
# Expected tasks: 75
# Unique tasks: 75
# 
# ✅ All tasks present
# ✅ No duplicates
# ✅ All seeds correct
```

### Resubmit Failed Tasks
```bash
# Generate resubmission script
python decode_seed_manifest.py --check-missing \
    experiments/generalization_matrix_834222/seed_manifests/MASTER_SEED_REGISTRY.csv

# Output: resubmit_missing.sh
# Run: bash resubmit_missing.sh
```

## Output Structure

```
experiments/generalization_matrix_834222/
├── seed_manifests/
│   ├── MASTER_SEED_REGISTRY.csv          # All 75 tasks
│   ├── task_0_manifest.txt               # Condition 0, seed 42
│   ├── task_1_manifest.txt               # Condition 0, seed 43
│   ├── ...
│   └── task_74_manifest.txt              # Condition 14, seed 46
│
├── condition_0_seed_42/                   # Individual run results
│   ├── checkpoints/
│   ├── logs/
│   └── results/
│
├── condition_0_seed_43/
│   └── ...
│
└── ... (75 total directories)
```

## Quick Reference

### Task ID Ranges by Condition
```
Condition  0 (PD + very_low):   Tasks 0-4
Condition  1 (PD + low):        Tasks 5-9
Condition  2 (PD + mid):        Tasks 10-14
Condition  3 (PD + high):       Tasks 15-19
Condition  4 (PD + very_high):  Tasks 20-24
Condition  5 (HD + very_low):   Tasks 25-29
Condition  6 (HD + low):        Tasks 30-34
Condition  7 (HD + mid):        Tasks 35-39
Condition  8 (HD + high):       Tasks 40-44
Condition  9 (HD + very_high):  Tasks 45-49
Condition 10 (SH + very_low):   Tasks 50-54
Condition 11 (SH + low):        Tasks 55-59
Condition 12 (SH + mid):        Tasks 60-64
Condition 13 (SH + high):       Tasks 65-69
Condition 14 (SH + very_high):  Tasks 70-74
```

### Games
- **PD** = Prisoner's Dilemma (conditions 0-4)
- **HD** = Hawk-Dove (conditions 5-9)
- **SH** = Stag Hunt (conditions 10-14)

### Opponent Ranges
- **very_low**: [0.0, 0.2] - mostly cooperative
- **low**: [0.2, 0.4] - somewhat cooperative
- **mid**: [0.4, 0.6] - balanced
- **high**: [0.6, 0.8] - somewhat defective
- **very_high**: [0.8, 1.0] - mostly defective

## Statistical Analysis

With 5 seeds per condition, you can:
- Compute means and standard errors
- Calculate 95% confidence intervals
- Run t-tests and ANOVAs
- Test hypotheses with p-values
- Distinguish signal from noise

**Standard Error Calculation:**
```python
import pandas as pd
import numpy as np

# Load results for a condition
results = []
for seed in [42, 52, 62, 72, 82]:
    result = load_result(condition_id=0, seed=seed)
    results.append(result['test_reward'])

mean_reward = np.mean(results)
std_error = np.std(results) / np.sqrt(5)
ci_95 = 1.96 * std_error

print(f"Reward: {mean_reward:.3f} ± {ci_95:.3f}")
```

## Troubleshooting

### Check Job Status
```bash
# View running/pending jobs
squeue -u $USER

# Check specific array job
squeue -j 834222

# View completed jobs
sacct -j 834222
```

### View Logs
```bash
# SLURM output
cat slurm_logs/gen_matrix_834222_37.out

# SLURM errors
cat slurm_logs/gen_matrix_834222_37.err

# Task manifest
cat experiments/generalization_matrix_834222/seed_manifests/task_37_manifest.txt
```

### Common Issues

**Missing tasks in registry:**
- Tasks may have failed or are still running
- Use `decode_seed_manifest.py --validate` to identify
- Use `decode_seed_manifest.py --check-missing` to resubmit

**Duplicate tasks:**
- May occur if job was submitted twice
- Check registry with `--validate` flag
- Keep first completion, ignore duplicates in analysis

**Incorrect seeds:**
- Should be rare - indicates script error
- Validate with `decode_seed_manifest.py --validate`
- Report as bug if systematic

## Best Practices

1. **Always validate** the master registry after all jobs complete
2. **Keep manifests** for reproducibility and debugging
3. **Check logs** if a task fails to identify the issue
4. **Use task ID decoder** to understand which condition/seed failed
5. **Backup results** before rerunning failed tasks (they may overwrite)

## Contact

For issues or questions about the multi-seed system:
- Check `EXPERIMENT_DESIGN.md` for design details
- Check `run_generalization_matrix.sh` for implementation
- Use `decode_seed_manifest.py --help` for usage info
