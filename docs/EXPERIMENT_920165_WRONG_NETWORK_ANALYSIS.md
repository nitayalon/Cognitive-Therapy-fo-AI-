# Experiment 920165: Wrong Network Architecture - Root Cause Analysis

## Summary
Experiment 920165 used the **STANDARD network** (128 hidden, 2 layers, 0.1 dropout) instead of the **SMALL network** (32 hidden, 1 layer, 0.05 dropout).

## Root Cause: Code Bug in main_experiment.py

### The Problem
**Line 2196 in main_experiment.py:**
```python
network_config = NetworkConfig()
```

This creates a NetworkConfig with **hardcoded default values**, ignoring the settings in the whole_population_config.json file.

### Default Values (from config.py line 22-24)
```python
@dataclass 
class NetworkConfig:
    hidden_size: int = 128    # STANDARD network default
    num_layers: int = 2       # STANDARD network default
    dropout: float = 0.1      # STANDARD network default
    input_size: int = 9
```

### What Should Have Happened
The code should load network_config from the wp_config JSON file, just like it does for other experiment modes (e.g., generalization-matrix mode).

## Timeline Evidence

1. **May 14, 2026** (commit 660c021):
   - Updated `config/whole_population_config.json` with small network settings
   - Changed: 128→32 hidden, 2→1 layers, 0.1→0.05 dropout
   
2. **May 19, 2026** (05:54:28):
   - Job 920165 launched using `--wp-config config/whole_population_config.json`
   - **BUG**: main_experiment.py ignored the config file's network settings
   - Used default NetworkConfig() instead → standard network (128/2/0.1)

## Verification from Logs

From experiment logs (task 9):
```
--experiment-config config/whole_population_config.json
```

From experiment_config.json (task 0):
```json
{
  "network_config": {
    "hidden_size": 128,    ← WRONG: Should be 32
    "num_layers": 2,       ← WRONG: Should be 1
    "dropout": 0.1,        ← WRONG: Should be 0.05
    "input_size": 9
  }
}
```

## Impact

- **All 15 training tasks** in experiment 920165 used the standard network
- Parameters: ~80,000 instead of ~3,500 (23x more than intended)
- Training time: Likely longer than planned
- **NO small network data exists** - need to rerun the experiment

## Fix Required

### Option 1: Fix the Code (Recommended)
Modify main_experiment.py to load network_config from wp_config JSON:

```python
# Around line 2196, replace:
network_config = NetworkConfig()

# With:
if args.experiment_mode == 'whole-population':
    with open(args.wp_config, 'r') as f:
        wp_config = json.load(f)
    network_config = NetworkConfig(**wp_config['network_config'])
else:
    network_config = NetworkConfig()
```

### Option 2: Command-Line Override (Temporary)
Add command-line arguments for network parameters:
```bash
--hidden-size 32 --num-layers 1 --dropout 0.05
```

## Recommendations

1. **Fix the code** to properly load network_config from wp_config JSON
2. **Rerun experiment** with correct small network architecture
3. **Add validation** to verify network architecture matches config file
4. **Update verification script** to check for this type of mismatch

## Related Files

- `main_experiment.py` (line 2196): Bug location
- `src/cognitive_therapy_ai/config.py` (lines 20-24): Default values
- `config/whole_population_config.json`: Correct small network settings
- `run_whole_population_train.sh`: SLURM submission script (correct)
- `verify_experiment_920165.py`: Verification script (detected the issue)

---
**Date**: May 23, 2026
**Detected by**: verify_experiment_920165.py
**Status**: Code bug identified, fix pending
