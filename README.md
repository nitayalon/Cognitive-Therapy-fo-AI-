# Cognitive Therapy for AI: Theory of Mind in Mixed-Motive Games

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

This repository implements a comprehensive framework for investigating **generalization failure in social AI** through multi-agent reinforcement learning. We model maladaptation as the inability to generalize across game structures and opponent types, and explore whether incorporating **Theory of Mind (ToM)** auxiliary tasks improves social intelligence.

## 🎯 Research Overview

### Core Research Question
How does the training environment affect an RL agent's ability to generalize in social interactions? Specifically:
- **Same Game, New Opponents**: Can agents adapt to novel opponent strategies?
- **New Game, Same Opponents**: Can agents transfer strategies across game structures?
- **New Game, New Opponents**: Complete out-of-distribution generalization?

### Key Innovation
We compare two architectures:
1. **Vanilla RL**: Standard LSTM policy network optimized for reward maximization
2. **ToM-RL**: Multi-task network that jointly learns:
   - Policy optimization (reward maximization)
   - Opponent behavior prediction (Theory of Mind)

### Cognitive Therapy Analogy
- **Bias Detection**: Identifying failures in opponent modeling → Recognizing cognitive distortions
- **Cognitive Restructuring**: Adapting strategies based on opponent understanding → Reframing maladaptive beliefs
- **Generalization**: Transfer to novel social contexts → Therapeutic intervention efficacy

## 🎮 Mixed-Motive Games

The framework supports three classic game-theoretic environments:

| Game | Description | Key Parameters |
|------|-------------|----------------|
| **Prisoner's Dilemma** | Cooperation vs. defection dilemma | R=3, T=5, S=0, P=1 |
| **Hawk-Dove** | Resource competition with conflict costs | V=4, C=6 |
| **Stag Hunt** | Coordination with security dilemma | Stag=5, Hare=3 |

Each game presents different strategic challenges: from pure competition (Prisoner's Dilemma) to coordination dilemmas (Stag Hunt).

## 🧠 ToM-RL Architecture

### Network Structure
```
Input (5D): [payoff_matrix_flat(4), round_number(1)]
    ↓
LSTM Layers (hidden_size=128, num_layers=2)
    ↓
Three Heads:
├─ Policy Head: π(action|state)
├─ Opponent Prediction Head: p(opponent_cooperates)
└─ Value Head: V(state)
```

### Loss Function
The **ToM-RL Loss** balances two objectives:

```math
L = L_RL + α · L_Op
```

- **L_RL** (Policy Gradient): `-E[log π(a|s) · Advantage]`
- **L_Op** (Opponent Prediction): Binary cross-entropy on opponent's next action
- **α**: Weighting parameter balancing reward optimization vs. opponent modeling

For full mathematical details, see [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md).

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ (with CUDA for GPU acceleration)
- NumPy, Pandas, Matplotlib, Seaborn

### Setup
```bash
# Clone the repository
git clone https://github.com/nitayalon/Cognitive-Therapy-fo-AI-.git
cd Cognitive-Therapy-fo-AI-

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## 🚀 Quick Start

### 1. Basic Generalization Experiment
Train on one game/opponent set, test generalization:
```bash
python main_experiment.py \
  --experiment-mode basic \
  --train-game prisoners-dilemma \
  --train-opponents 0.1,0.3,0.5 \
  --test-game hawk-dove \
  --test-opponents 0.7,0.9
```

### 2. Generalization Matrix (Full Study)
Complete 4×4 game cross-training experiment:
```bash
python main_experiment.py \
  --experiment-mode generalization-matrix \
  --opponents 0.1,0.3,0.5,0.7,0.9 \
  --num-games 100
```

### 3. Single Game Training
Train and evaluate on single game:
```bash
python main_experiment.py \
  --game prisoners-dilemma \
  --opponents 0.1,0.3,0.5,0.7,0.9 \
  --num-games 100 \
  --adaptive-loss
```

### 4. Quick Test (5 minutes)
Verify installation:
```bash
python main_experiment.py --config config/quick_test_config.json
```

### 5. Using Configuration Files
```bash
python main_experiment.py --config config/default_config.json
```

## 🧪 Running Experiments

### SLURM Cluster (HPC)
```bash
# Vanilla RL baseline
sbatch run_vanilla_rl_experiment.sh

# Generalization matrix
sbatch run_generalization_matrix.sh

# Array jobs for parallel tasks
sbatch run_experiment_array.sh
```

### Local Execution
```bash
# Single experiment
bash run_experiment.sh

# Test run
bash run_experiment_test.sh
```

## 📁 Repository Structure

```
Cognitive-Therapy-fo-AI-/
│
├── src/cognitive_therapy_ai/     # Core framework
│   ├── games.py                  # Game implementations (PD, HD, BoS, SH)
│   ├── network.py                # GameLSTM architecture
│   ├── opponent.py               # Opponent strategies and factory
│   ├── trainer.py                # GameSession & GameTrainer
│   ├── tom_rl_loss.py            # ToM-RL loss function (ACTIVE)
│   ├── loss.py                   # Legacy 3-task loss (compatibility)
│   ├── config.py                 # Configuration dataclasses
│   ├── utils.py                  # Utility functions
│   ├── training_monitor.py       # Training diagnostics
│   └── testing_monitor.py        # Evaluation diagnostics
│
├── config/                       # Experiment configurations
│   ├── default_config.json       # Full experimental setup
│   ├── generalization_matrix_config.json
│   └── quick_test_config.json    # Fast validation
│
├── experiments/                  # Experiment outputs
│   ├── analysis_scripts/         # Data analysis tools
│   │   ├── analyze_action_distributions.py
│   │   ├── analyze_vanilla_comprehensive.py
│   │   ├── analyze_vanilla_vs_proto_tom_comprehensive.py
│   │   ├── aggregate_generalization_results.py
│   │   ├── task_generalization_analysis.py
│   │   ├── generalization_2x2_analysis.py
│   │   └── ...
│   ├── generalization_matrix_*/  # Full 4×4 game experiments
│   ├── vanilla_rl_array_*/       # Vanilla RL baselines
│   └── results/                  # Aggregated results
│
├── tests/                        # Unit and integration tests
│   ├── test_framework.py         # Core framework tests
│   ├── test_vanilla_rl.py        # Vanilla baseline tests
│   ├── test_generalization_*.py  # Generalization tests
│   ├── test_multi_game_*.py      # Multi-game experiments
│   └── ...
│
├── docs/                         # Documentation
│   ├── MODIFICATIONS.md          # Technical change log
│   ├── COMPLETE_MONITORING_SYSTEM.md
│   └── KEYERROR_FIXES.md
│
├── main_experiment.py            # Main experiment runner
├── setup.py                      # Package installation
├── requirements.txt              # Dependencies
│
├── CHANGELOG.md                  # High-level changes
├── EXPERIMENT_DESIGN.md          # Research design & methods
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## 🔬 Experiments & Data

### Completed Experiments

#### 1. Generalization Matrix Study
**Location**: `experiments/generalization_matrix_*/`
- **Design**: Train on each of 3 games, test on all 3 games (3×3 matrix)
- **Opponents**: 5 types per game (defection probabilities: 0.1, 0.3, 0.5, 0.7, 0.9)
- **Output**: 9 train-test combinations (3×3 matrix)
- **Key Metrics**: Cross-game generalization error, cooperation rates, opponent prediction accuracy

#### 2. Vanilla RL Baseline
**Location**: `experiments/vanilla_rl_array_*/`
- **Design**: Same as ToM-RL but without opponent prediction head
- **Purpose**: Isolate effect of Theory of Mind auxiliary task
- **Comparison**: Vanilla vs. ToM-RL performance across all conditions

#### 3. Array-Based Task Studies
**Location**: `experiments/array_run_*/`
- **Design**: SLURM array jobs for parallelized experiments
- **Tasks**: Individual game-opponent combinations
- **Scaling**: Efficient execution on HPC clusters

### Experiment Output Structure
Each experiment creates:
```
experiment_name_YYYYMMDD_HHMMSS/
├── checkpoints/                  # Model states (.pth)
│   ├── detailed_training_logs/   # Per-epoch metrics (CSV)
│   └── final_model.pth
├── logs/                         # Training logs
│   ├── training.log
│   └── evaluation.log
├── plots/                        # Visualizations
│   ├── training_curves.png
│   ├── cooperation_heatmap.png
│   └── generalization_matrix.png
├── results/                      # Pickled results
│   ├── task_*_results.pkl        # Per-task results
│   └── aggregated_results.json
└── experiment_config.json        # Full configuration
```

## 📊 Analysis Tools

### Core Analysis Scripts
Located in `experiments/analysis_scripts/`:

#### Performance Analysis
- **`analyze_vanilla_comprehensive.py`**: Vanilla RL detailed analysis
- **`analyze_vanilla_vs_proto_tom_comprehensive.py`**: Compare architectures
- **`analyze_vanilla_baseline_performance.py`**: Baseline metrics

#### Generalization Analysis
- **`task_generalization_analysis.py`**: Per-task generalization metrics
- **`generalization_2x2_analysis.py`**: Cross-game transfer
- **`aggregate_generalization_results.py`**: Summary statistics

#### Behavioral Analysis
- **`analyze_action_distributions.py`**: Cooperation/defection patterns
- **`within_task_opponent_analysis.py`**: Opponent-specific strategies

#### Data Integrity
- **`check_vanilla_data_integrity.py`**: Validate experiment outputs
- **`generate_summary_report.py`**: Comprehensive experiment summaries

### Running Analysis
```bash
cd experiments/analysis_scripts

# Full comparison
python analyze_vanilla_vs_proto_tom_comprehensive.py

# Generalization matrix
python generalization_2x2_analysis.py

# Action distributions
python analyze_action_distributions.py
```

## ⚙️ Configuration

### Network Configuration
```json
{
  "network": {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.1,
    "input_size": 5
  }
}
```

### Training Configuration
```json
{
  "training": {
    "num_games_per_partner": 100,
    "learning_rate": 0.001,
    "max_epochs": 500,
    "convergence_threshold": 1e-6,
    "patience": 50,
    "tom_loss_weight": 1.0
  }
}
```

### Experiment Configuration
```json
{
  "experiment": {
    "opponent_defection_probs": [0.1, 0.3, 0.5, 0.7, 0.9],
    "random_seed": 42,
    "save_checkpoints": true,
    "log_level": "INFO"
  }
}
```

See `config/default_config.json` for complete configuration options.

## 🎮 Demo Scripts

### ToM-RL Training Demo
```bash
python tom_rl_demo.py
```
Demonstrates:
- Network creation and initialization
- ToM-RL loss computation
- Training loop with opponent prediction
- Visualization of learned behaviors

### Detailed Training Demo
```bash
python detailed_training_demo.py
```
Shows:
- Single-game training workflow
- Multi-game training workflow  
- Checkpoint management
- Results analysis

### Complete Monitoring Demo
```bash
python complete_monitoring_demo.py
```
Illustrates:
- Training progress monitoring
- Real-time loss tracking
- Performance visualization
- Diagnostic outputs

## 🔧 Command Line Interface

### Main Experiment Options
```
python main_experiment.py [OPTIONS]

Core Options:
  --experiment-mode {basic,multi-game,generalization-matrix,segmented}
                        Experiment type
  --config PATH         JSON configuration file
  
Game Selection:
  --game NAME           Single game mode
  --train-game NAME     Training game (basic mode)
  --test-game NAME      Test game (basic/multi-game mode)
  --training-games LIST Training games (multi-game mode)
  
Opponent Configuration:
  --opponents PROBS     Comma-separated defection probabilities
  --train-opponents PROBS  Training opponents (basic mode)
  --test-opponents PROBS   Test opponents (basic mode)
  
Training Parameters:
  --num-games INT       Games per opponent session (default: 100)
  --max-epochs INT      Maximum training epochs (default: 500)
  --learning-rate FLOAT Adam learning rate (default: 0.001)
  --patience INT        Early stopping patience (default: 50)
  
Loss Configuration:
  --adaptive-loss       Use adaptive ToM loss weighting
  --tom-weight FLOAT    ToM loss weight α (default: 1.0)
  
Output & Logging:
  --output-dir PATH     Output directory (default: experiments/)
  --seed INT            Random seed (default: 42)
  --verbose             Enable verbose logging
  --device {cpu,cuda,auto}  Computation device (default: auto)
```

### Examples
```bash
# Basic generalization with specific opponents
python main_experiment.py \
  --experiment-mode basic \
  --train-game prisoners-dilemma \
  --train-opponents 0.1,0.5,0.9 \
  --test-game stag-hunt \
  --test-opponents 0.3,0.7

# Adaptive loss weighting
python main_experiment.py \
  --game hawk-dove \
  --opponents 0.2,0.4,0.6,0.8 \
  --adaptive-loss \
  --verbose

# Custom configuration with GPU
python main_experiment.py \
  --config my_config.json \
  --device cuda \
  --seed 123
```

## 🛠️ Extending the Framework

### Adding New Games

1. **Create game class** extending `MixedMotiveGame`:
```python
from cognitive_therapy_ai.games import MixedMotiveGame
import numpy as np

class CustomGame(MixedMotiveGame):
    def __init__(self, param_a=3.0, param_b=1.0):
        super().__init__("Custom-Game")
        self.param_a = param_a
        self.param_b = param_b
    
    def get_payoff_matrix(self):
        # Return [[CC, CD], [DC, DD]] payoff matrix
        return np.array([
            [self.param_a, 0],
            [self.param_b, self.param_b/2]
        ], dtype=np.float32)
```

2. **Register in GameFactory** (`src/cognitive_therapy_ai/games.py`):
```python
@staticmethod
def create_game(game_name: str, **kwargs):
    games = {
        'custom-game': CustomGame,
        # ... existing games
    }
    return games[game_name](**kwargs)
```

### Adding New Opponent Strategies

1. **Create strategy class** extending `OpponentStrategy`:
```python
from cognitive_therapy_ai.opponent import OpponentStrategy, Action

class AdaptiveOpponent(OpponentStrategy):
    """Adapts based on network's cooperation rate."""
    
    def __init__(self, adaptation_rate=0.1):
        self.adaptation_rate = adaptation_rate
        self.coop_history = []
    
    def choose_action(self, game_history, round_number):
        if round_number == 0:
            return Action.COOPERATE
        
        # Track opponent cooperation
        recent_coops = sum(1 for h in game_history[-5:] 
                          if h['opponent_action'] == Action.COOPERATE)
        coop_rate = recent_coops / min(5, len(game_history))
        
        # Defect if opponent defects often
        return Action.DEFECT if coop_rate < 0.3 else Action.COOPERATE
```

2. **Register in OpponentFactory** (`src/cognitive_therapy_ai/opponent.py`):
```python
@staticmethod
def create_opponent(opponent_type: str, **kwargs):
    strategies = {
        'adaptive': AdaptiveOpponent,
        # ... existing strategies
    }
    return Opponent(strategies[opponent_type](**kwargs))
```

### Custom Loss Functions

Extend `ToMRLLoss` for custom objectives:
```python
from cognitive_therapy_ai.tom_rl_loss import ToMRLLoss

class CustomToMLoss(ToMRLLoss):
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__(alpha=alpha)
        self.beta = beta
    
    def forward(self, policy_logits, opponent_coop_probs, 
                value_estimates, actions, rewards, opponent_actions):
        # Base ToM-RL loss
        base_loss = super().forward(
            policy_logits, opponent_coop_probs,
            value_estimates, actions, rewards, opponent_actions
        )
        
        # Add custom regularization
        policy_entropy = -torch.sum(
            torch.softmax(policy_logits, dim=1) * 
            torch.log_softmax(policy_logits, dim=1)
        )
        
        return base_loss - self.beta * policy_entropy
```

## 🔑 Key Implementation Details

### Session-Based Training
- **GameSession**: Plays T consecutive games with same opponent
- **Hidden State Persistence**: LSTM state maintained across all T games
- **State Reset**: Hidden state resets between sessions, not within

### State Representation
All games use **standardized 5-element vectors**:
```python
state = [
    payoff[0,0],  # Cooperate vs Cooperate
    payoff[0,1],  # Cooperate vs Defect
    payoff[1,0],  # Defect vs Cooperate
    payoff[1,1],  # Defect vs Defect
    round_number_normalized  # Normalized to [0, 1]
]
```

### Training Data Structure
ToM-RL expects batched tensors:
```python
training_data = {
    'policy_logits': torch.Tensor,          # (batch_size, 2)
    'opponent_coop_probs': torch.Tensor,    # (batch_size, 1)
    'value_estimates': torch.Tensor,        # (batch_size, 1)
    'actions': torch.Tensor,                # (batch_size,)
    'rewards': torch.Tensor,                # (batch_size,)
    'opponent_actions': torch.Tensor        # (batch_size,)
}
```

### Loss Normalization
ToM-RL normalizes losses for stable multi-task learning:
```python
L_RL_norm = L_RL / max(1e-8, L_RL.detach())
L_Op_norm = L_Op / max(1e-8, L_Op.detach())
Total_Loss = L_RL_norm + α * L_Op_norm
```

## 📖 Research Applications

### Completed Studies
1. **ToM-RL vs Vanilla RL**: Effect of opponent prediction on generalization
2. **Cross-Game Transfer**: Which games facilitate better transfer learning?
3. **Opponent Spectrum Analysis**: How training opponent diversity affects robustness
4. **Generalization Matrix**: Complete 4×4 game cross-training analysis

### Potential Extensions
- **Curriculum Learning**: Progressive opponent difficulty
- **Meta-Learning**: Fast adaptation to novel opponents
- **Multi-Agent ToM**: Simultaneous modeling of multiple opponents
- **Hierarchical Policies**: Abstract strategy selection
- **Continuous Games**: Beyond binary cooperation/defection

### Metrics for Analysis
- **Generalization Error**: Performance drop on test vs. train conditions
- **ToM Utilization**: How much opponent prediction influences policy
- **Strategy Diversity**: Entropy of learned policies
- **Robustness**: Performance variance across opponent types
- **Transfer Efficiency**: Learning speed on new games

## 📚 Documentation

- **[EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)**: Complete research design and methods
- **[CHANGELOG.md](CHANGELOG.md)**: Version history and notable changes
- **[docs/MODIFICATIONS.md](docs/MODIFICATIONS.md)**: Detailed technical modifications
- **[docs/COMPLETE_MONITORING_SYSTEM.md](docs/COMPLETE_MONITORING_SYSTEM.md)**: Monitoring system guide
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)**: Development guidelines

## 🧪 Testing

Run test suite:
```bash
# Framework validation
python tests/test_framework.py

# Generalization tests
python tests/test_generalization_same_game_new_opponents.py
python tests/test_generalization_new_game_same_opponents.py
python tests/test_generalization_new_game_new_opponents.py

# Vanilla RL baseline
python tests/test_vanilla_rl.py

# All tests
pytest tests/
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the existing code style
4. Add tests for new functionality
5. Update documentation (README, CHANGELOG, docs/)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

See [.github/copilot-instructions.md](.github/copilot-instructions.md) for development guidelines.

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@software{cognitive_therapy_ai_2026,
  title = {Cognitive Therapy for AI: Theory of Mind in Mixed-Motive Games},
  author = {Alon, Nitay and Contributors},
  year = {2026},
  url = {https://github.com/nitayalon/Cognitive-Therapy-fo-AI-},
  version = {1.0}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions, collaborations, or issues:
- **GitHub Issues**: [Report a bug](https://github.com/nitayalon/Cognitive-Therapy-fo-AI-/issues)
- **Email**: [Contact maintainers](mailto:nitay.alon@mail.huji.ac.il)

## 🙏 Acknowledgments

- Max Planck Institute for Human Development
- Hebrew University of Jerusalem
- PyTorch team for the deep learning framework
- All contributors and collaborators

---

**Version**: 1.0  
**Last Updated**: March 2026  
**Status**: Active Development
