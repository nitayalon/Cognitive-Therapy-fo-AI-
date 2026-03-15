# Repository Guide for Collaborators

**Version**: 1.0  
**Date**: March 11, 2026  
**Prepared for**: Joe Barnby
 and Sep Razavi
## 📋 Quick Overview

This repository investigates how RL agents generalize (or fail to generalize) in social games, comparing **Vanilla RL** vs. **Theory of Mind RL (ToM-RL)** architectures across different game structures and opponent types.

## 🚀 Getting Started in 5 Minutes

1. **Clone and Install**
   ```bash
   git clone https://github.com/nitayalon/Cognitive-Therapy-fo-AI-.git
   cd Cognitive-Therapy-fo-AI-
   pip install -e .
   ```

2. **Run Quick Test** (5 minutes)
   ```bash
   python main_experiment.py --config config/quick_test_config.json
   ```

3. **Explore Demo** (10 minutes)
   ```bash
   python tom_rl_demo.py
   ```

4. **Read Documentation**
   - [README.md](README.md) - Complete framework guide
   - [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) - Research design & methods
   - [CHANGELOG.md](CHANGELOG.md) - Recent changes

## 📂 Repository Organization

### Core Code (`src/cognitive_therapy_ai/`)
- **`games.py`** - 4 mixed-motive games (PD, HD, BoS, SH)
- **`network.py`** - GameLSTM architecture (ToM-RL)
- **`opponent.py`** - Opponent strategies and factories
- **`trainer.py`** - Training orchestration
- **`tom_rl_loss.py`** - Main loss function (ACTIVELY USED)
- **`loss.py`** - Legacy loss (kept for compatibility)

### Experiments
- **Main Script**: `main_experiment.py`
- **Configurations**: `config/*.json`
- **Results**: `experiments/*/`
- **Analysis**: `experiments/analysis_scripts/`

### Testing
- **Framework Tests**: `tests/test_framework.py`
- **Generalization Tests**: `tests/test_generalization_*.py`
- **Vanilla Baseline**: `tests/test_vanilla_rl.py`

### Documentation
- **`docs/MODIFICATIONS.md`** - Technical implementation details
- **`docs/COMPLETE_MONITORING_SYSTEM.md`** - Training monitoring guide
- **`.github/copilot-instructions.md`** - Development guidelines

## 🔬 Available Datasets

### 1. Generalization Matrix (`experiments/generalization_matrix_*/`)
- **Design**: Train on each game, test on all games (4×4 matrix)
- **Opponents**: 5 types per game (p ∈ {0.1, 0.3, 0.5, 0.7, 0.9})
- **Analysis**: Cross-game transfer, generalization errors

### 2. Vanilla RL Baseline (`experiments/vanilla_rl_array_*/`)
- **Design**: Same as ToM-RL but without opponent prediction
- **Purpose**: Isolate effect of Theory of Mind
- **Key Comparison**: `experiments/analysis_scripts/analyze_vanilla_vs_proto_tom_comprehensive.py`

### 3. Array Runs (`experiments/array_run_*/`)
- **Design**: SLURM array jobs for parallel execution
- **Data**: Task-specific results with detailed training logs

## 📊 Key Analysis Scripts

All located in `experiments/analysis_scripts/`:

### Performance Comparison
```bash
# Compare Vanilla vs ToM-RL
python analyze_vanilla_vs_proto_tom_comprehensive.py

# Vanilla detailed analysis
python analyze_vanilla_comprehensive.py

# Baseline metrics
python analyze_vanilla_baseline_performance.py
```

### Generalization Analysis
```bash
# Cross-game transfer
python generalization_2x2_analysis.py

# Per-task generalization
python task_generalization_analysis.py

# Aggregate results
python aggregate_generalization_results.py
```

### Behavioral Analysis
```bash
# Cooperation/defection patterns
python analyze_action_distributions.py

# Opponent-specific strategies
python within_task_opponent_analysis.py
```

## 🎯 Common Tasks

### Run New Experiment
```bash
python main_experiment.py \
  --experiment-mode basic \
  --train-game prisoners-dilemma \
  --train-opponents 0.1,0.5,0.9 \
  --test-game stag-hunt \
  --test-opponents 0.3,0.7
```

### Analyze Existing Results
```bash
cd experiments/analysis_scripts
python analyze_vanilla_vs_proto_tom_comprehensive.py
```

### Create Custom Configuration
```bash
cp config/default_config.json config/my_experiment.json
# Edit my_experiment.json
python main_experiment.py --config config/my_experiment.json
```

### Run on SLURM Cluster
```bash
# Edit run_vanilla_rl_experiment.sh with your parameters
sbatch run_vanilla_rl_experiment.sh
```

## 🔑 Key Concepts

### Session-Based Training
- **Session**: T consecutive games (default T=100) with same opponent
- **LSTM State**: Persists across all T games within session
- **Reset**: State resets between sessions, not within

### Opponent Specification
Opponents defined by **defection probability** (p ∈ [0, 1]):
- **p=0.1**: Cooperative (10% defection)
- **p=0.5**: Neutral (50-50 strategy)
- **p=0.9**: Defective (90% defection)

### Generalization Conditions
1. **Same Game, New Opponents**: Opponent robustness
2. **New Game, Same Opponents**: Game structure transfer
3. **New Game, New Opponents**: Full out-of-distribution test

## 📈 Expected Results Structure

Each experiment creates:
```
experiment_name_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── detailed_training_logs/
│   │   └── detailed_training_log.csv  # Per-epoch metrics
│   └── final_model.pth
├── logs/
│   ├── training.log
│   └── evaluation.log
├── plots/
│   ├── training_curves.png
│   └── generalization_matrix.png
├── results/
│   ├── task_*_results.pkl           # Pickled results
│   └── aggregated_results.json
└── experiment_config.json
```

## ⚠️ Important Notes

1. **Loss Functions**: Use `tom_rl_loss.py` (modern), not `loss.py` (legacy)
2. **State Representation**: Always 5D vector [payoff_matrix(4), round(1)]
3. **Device Auto-Detection**: Framework automatically uses CUDA if available
4. **Checkpoints**: Saved per epoch in `checkpoints/` directory
5. **Random Seeds**: Set via `--seed` for reproducibility

## 🆘 Troubleshooting

### Import Errors
```bash
# Ensure package is installed
pip install -e .

# Check Python path
python -c "from cognitive_therapy_ai import GameLSTM; print('OK')"
```

### CUDA/Memory Issues
```bash
# Force CPU
python main_experiment.py --device cpu ...

# Reduce batch size (edit config JSON)
# "num_games_per_partner": 50  # instead of 100
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

## 📧 Getting Help

1. **Check Documentation**
   - [README.md](README.md) - Comprehensive guide
   - [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) - Research methods
   - [docs/MODIFICATIONS.md](docs/MODIFICATIONS.md) - Implementation details

2. **Run Tests**
   ```bash
   python tests/test_framework.py
   ```

3. **Contact**
   - GitHub Issues: Report bugs or ask questions
   - Email: nitay.alon@mail.huji.ac.il

## 🎓 Recommended Reading Order

1. **[README.md](README.md)** - Framework overview (15 min)
2. **Run `tom_rl_demo.py`** - See framework in action (10 min)
3. **[EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)** - Research design (30 min)
4. **Explore `experiments/analysis_scripts/`** - Analysis examples (20 min)
5. **[docs/MODIFICATIONS.md](docs/MODIFICATIONS.md)** - Deep dive (as needed)

## 🚦 Next Steps

### For Analysis
1. Review existing results in `experiments/`
2. Run analysis scripts in `experiments/analysis_scripts/`
3. Generate custom visualizations

### For New Experiments
1. Copy and modify config from `config/default_config.json`
2. Test with `config/quick_test_config.json` first
3. Run full experiment with `main_experiment.py`
4. Analyze results with provided scripts

### For Development
1. Read [.github/copilot-instructions.md](.github/copilot-instructions.md)
2. Follow existing code patterns
3. Add tests for new features
4. Update documentation

---

**Happy researching!** 🧠🎮
