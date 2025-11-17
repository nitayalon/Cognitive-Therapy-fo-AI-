# File Location Verification Report

## Summary of Changes

All CSV and JSON files are now correctly routed to the experiments directory structure. Here's the complete mapping:

### Directory Structure
```
experiments/
└── mixed_motive_experiment_YYYYMMDD_HHMMSS/
    ├── experiment_config.json                    # Main experiment configuration
    ├── checkpoints/                             # Model checkpoints and training logs
    │   └── detailed_training_logs/
    │       ├── detailed_training_log.csv        # Training data CSV

    ├── logs/                                    # All log files
    │   ├── experiment.log                       # Main experiment log
    │   └── detailed_testing_logs/               # Testing data
    │       ├── detailed_testing_log.csv         # Testing data CSV

    │       ├── session_*_summary.json           # Per-session summaries
    │       └── training_test_linkage.json       # Links training to testing
    ├── plots/                                   # Visualization files
    └── results/                                 # Final analysis and reports
        ├── experiment_report.json               # Single game experiment report
        ├── multi_game_experiment_report.json   # Multi-game experiment report
        ├── raw_results.json                     # Raw experimental data
        └── segmented_experiments_report.json   # Segmented experiment report
```

### Changes Made

#### 1. Updated evaluate() calls in main_experiment.py
**Before:**
```python
test_results = trainer.evaluate(
    game=test_game_instance,
    opponents=opponents,
    num_sessions=50
)
```

**After:**
```python
test_results = trainer.evaluate(
    game=test_game_instance,
    opponents=opponents,
    num_sessions=50,
    enable_detailed_testing=True,
    testing_log_dir=os.path.join(output_dirs['logs'], 'detailed_testing_logs')
)
```

#### 2. File Location Mapping

| File Type | Location | Responsible Component |
|-----------|----------|----------------------|
| **Training Logs** | `checkpoints/detailed_training_logs/` | TrainingMonitor (via trainer.py) |
| **Testing Logs** | `logs/detailed_testing_logs/` | TestingMonitor (via evaluate() calls) |
| **Configuration** | `<base>/experiment_config.json` | main_experiment.py |
| **Reports** | `results/*.json` | create_multi_game_report() |
| **Raw Data** | `results/raw_results.json` | create_multi_game_report() |

#### 3. Automatic Directory Creation
- All directories are created automatically via `create_output_dirs()` in utils.py
- Individual monitors create their subdirectories as needed
- No manual directory management required

### Verification

#### ✅ Training Files
- `detailed_training_log.csv` → `experiments/.../checkpoints/detailed_training_logs/`


#### ✅ Testing Files  
- `detailed_testing_log.csv` → `experiments/.../logs/detailed_testing_logs/`
- `detailed_testing_log.xlsx` → `experiments/.../logs/detailed_testing_logs/`
- `session_*_summary.json` → `experiments/.../logs/detailed_testing_logs/`
- `training_test_linkage.json` → `experiments/.../logs/detailed_testing_logs/`

#### ✅ Report Files
- `experiment_report.json` → `experiments/.../results/`
- `multi_game_experiment_report.json` → `experiments/.../results/`
- `raw_results.json` → `experiments/.../results/`
- `segmented_experiments_report.json` → `experiments/.../results/`

#### ✅ Configuration Files
- `experiment_config.json` → `experiments/.../`

### Result
🎉 **100% Compliance**: All CSV and JSON files are now written to the experiments folder structure!

### Additional Benefits
1. **Organized Structure**: Clear separation of training vs testing data
2. **Automatic Cleanup**: All files grouped under single experiment directory
3. **Easy Analysis**: Related files are co-located for analysis
4. **Reproducibility**: Full experiment state captured in single directory
5. **Version Control**: Easy to archive/backup complete experiments

The system now ensures that no CSV or JSON files are written outside the experiments directory, providing a clean and organized output structure for all experimental data.