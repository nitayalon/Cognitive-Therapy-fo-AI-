# Cognitive Therapy for AI: Mixed-Motive Game Training Framework

This repository implements a framework for training RRN (LSTM) networks on mixed-motive games to study representation failure (maladaptiveness) as a function of training game and opponent behavior. 
The core workflow trains on a **single game** against a **single spectrum of opponents**, then evaluates generalization across:
1. **Same game, new opponents**
2. **New game, same opponents**
3. **New game, new opponents**

## Project Overview

The framework implements a **Multi-Agent Reinforcement Learning (MARL)** approach with support for four classic mixed-motive games:
- **Hawk-Dove Game**: Resource competition with potential for costly conflict 
- **Prisoner's Dilemma**: Classic cooperation vs. defection dilemma
- **Battle of the Sexes**: Coordination game with conflicting preferences
- **Stag Hunt**: Coordination game balancing mutual benefit vs. individual security

### ToM-RL Architecture

The network uses a multi-task objective that combines two complementary loss functions to create Theory of Mind inductive bias:

1. **Reinforcement Learning Loss (LRL)**: Policy gradient with advantage estimation
   - `LRL = -E_t[log π(a_t|s_t) * Â_t]`
   - Where `Â_t = R_t - V_θ(s_t)` is the advantage estimate

2. **Opponent Prediction Loss (LOp)**: Binary cross-entropy for predicting opponent cooperation
   - `LOp = -Σ_t [o_t * log(p̂_t^d) + (1 - o_t) * log(1 - p̂_t^d)]`
   - Creates ToM inductive bias for understanding opponent behavior

**Total Loss**: `L = LRL + α * LOp`

The auxiliary prediction task encourages the agent to develop internal representations of opponent behavior patterns, corresponding to the computational role of Theory of Mind in social decision-making.

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/nitayalon/Cognitive-Therapy-fo-AI-.git
cd Cognitive-Therapy-fo-AI-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Basic Usage
Run the basic single-game generalization experiment:
```bash
python main_experiment.py --experiment-mode basic --train-game prisoners-dilemma --train-opponents 0.1,0.3 --test-game hawk-dove --test-opponents 0.7,0.9
```

### Single Game Training (In-Distribution)
Train on a specific game with a focused opponent spectrum:
```bash
python main_experiment.py --experiment-mode basic --train-game prisoners-dilemma --train-opponents 0.1,0.3
```

### Using Configuration Files
```bash
python main_experiment.py --config config/default_config.json
```

### Quick Test Run
For testing and debugging:
```bash
python main_experiment.py --config config/quick_test_config.json --verbose
```

### Multi-Game Experiment (Legacy)
```bash
python main_experiment.py --experiment-mode multi-game --training-games prisoners-dilemma,hawk-dove --test-game stag-hunt --opponents 0.1,0.3,0.5,0.7,0.9
```

## Framework Architecture

### Core Components

#### 1. Mixed-Motive Games (`cognitive_therapy_ai.games`)
- `MixedMotiveGame`: Abstract base class for all games
- `HawkDove`: Hawk-Dove game implementation
- `PrisonersDilemma`: Prisoner's Dilemma implementation
- `BattleOfSexes`: Battle of the Sexes implementation
- `StagHunt`: Stag Hunt game implementation
- `GameFactory`: Factory for creating game instances

#### 2. Opponent Models (`cognitive_therapy_ai.opponent`)
- `OpponentStrategy`: Abstract base for opponent strategies
- `ProbabilisticOpponent`: Main opponent with fixed defection probability
- `TitForTatOpponent`, `AlwaysCooperateOpponent`, etc.: Additional strategies
- `Opponent`: Wrapper class managing opponent behavior and statistics

#### 3. Neural Network (`cognitive_therapy_ai.network`)
- `GameLSTM`: ToM-RL LSTM with policy head, opponent prediction head, and value function
- `NetworkManager`: Utility class for network operations and checkpointing

#### 4. Loss Functions (`cognitive_therapy_ai.tom_rl_loss`)
- `ToMRLLoss`: Theory of Mind Reinforcement Learning loss combining RL and opponent prediction
- `AdaptiveToMRLLoss`: Automatically balances α parameter during training
- `LossAnalyzer`: Tracks and analyzes loss components and ToM contribution

#### 5. Training System (`cognitive_therapy_ai.trainer`) 
- `GameSession`: Manages T games between network and single opponent
- `GameTrainer`: Main training orchestrator handling multiple opponents and convergence
- Supports adaptive loss weighting and early stopping

#### 6. Legacy Loss Functions (`cognitive_therapy_ai.loss`)
- `CompositeLoss`: Original three-task loss (kept for backward compatibility)
- `AdaptiveLoss`: Original adaptive loss balancing
- `LossAnalyzer`: Legacy loss convergence analysis

## Configuration

### Network Configuration
```python
network_config = NetworkConfig(
    hidden_size=128,          # LSTM hidden size
    num_layers=2,             # Number of LSTM layers
    dropout=0.1               # Dropout probability
)
```

### Training Configuration
```python
training_config = TrainingConfig(
    num_games_per_partner=100,           # T parameter - games per opponent
    learning_rate=0.001,                 # Adam learning rate
    max_epochs=500,                      # Maximum training epochs
    convergence_threshold=1e-6,          # Convergence threshold for loss
    patience=50,                         # Early stopping patience
    reward_loss_weight=1.0,              # Policy gradient loss weight
    action_prediction_loss_weight=1.0,   # Action prediction loss weight
    type_prediction_loss_weight=1.0      # Type prediction loss weight
)
```

### Experiment Configuration
```python
experiment_config = ExperimentConfig(
    opponent_defection_probs=[0.1, 0.3, 0.5, 0.7, 0.9],  # p values to test
    random_seed=42,                      # Random seed for reproducibility
    save_checkpoints=True,               # Whether to save model checkpoints
    log_level="INFO"                     # Logging level
)
```

## Command Line Options

```
python main_experiment.py [OPTIONS]

Options:
    --experiment-mode {basic,multi-game,segmented}
                                                        Experiment mode (default: basic)
    --config PATH              Path to JSON configuration file
    --train-game NAME          Train game for basic mode
    --training-games LIST      Comma-separated training games for multi-game mode
    --test-game NAME           Test game (basic + multi-game)
    --train-opponents PROBS    Comma-separated training opponent defection probabilities (basic mode)
    --test-opponents PROBS     Comma-separated test opponent defection probabilities (basic mode)
    --opponents PROBS          Comma-separated defection probabilities (multi-game/segmented)
    --num-games INT            Games per partner - T parameter (default: 100)
    --max-epochs INT           Maximum training epochs (default: 500)
    --output-dir PATH          Output directory (default: experiments)
    --seed INT                 Random seed (default: 42)
    --device {cpu,cuda,auto}   Training device (default: auto)
    --adaptive-loss            Use adaptive loss weighting
    --verbose                  Enable verbose logging
```

## Output Structure

Experiments create organized output directories:
```
experiments/
└── basic_generalization_experiment_YYYYMMDD_HHMMSS/
    ├── checkpoints/          # Model checkpoints
    ├── logs/                 # Training/testing logs
    ├── plots/                # Visualization plots
    ├── results/              # Experiment results
    │   ├── basic_generalization_results.pkl
    │   └── basic_generalization_report.json
    └── experiment_config.json
```

## Results Analysis

The framework provides comprehensive results including:

### Training Metrics
- Loss convergence for all three components
- Training epochs and convergence status
- Learning curves and optimization statistics

### Game Performance
- Cumulative rewards against different opponent types
- Cooperation rates and strategy adaptation
- Opponent action prediction accuracy
- Opponent type estimation error

### Network Analysis
- Hidden state representations over training
- Policy evolution and strategy changes
- Multi-task learning balance and trade-offs

## Extending the Framework

### Adding New Games
```python
from cognitive_therapy_ai.games import MixedMotiveGame, Action

class CustomGame(MixedMotiveGame):
    def __init__(self, custom_param=1.0):
        super().__init__("Custom-Game")
        self.custom_param = custom_param
    
    def get_payoff_matrix(self):
        # Return 2x2 payoff matrix
        return np.array([[3, 0], [5, 1]], dtype=np.float32)
```

### Adding New Opponent Strategies
```python
from cognitive_therapy_ai.opponent import OpponentStrategy, Action

class CustomStrategy(OpponentStrategy):
    def choose_action(self, game_history, round_number):
        # Implement custom strategy logic
        return Action.COOPERATE if round_number % 2 == 0 else Action.DEFECT
```

### Custom Loss Functions
```python
from cognitive_therapy_ai.loss import CompositeLoss

class CustomLoss(CompositeLoss):
    def forward(self, *args, **kwargs):
        # Implement custom loss computation
        base_loss = super().forward(*args, **kwargs)
        # Add custom components
        return base_loss
```

## Research Applications

This framework supports various research directions:

### Cognitive Therapy Analogies
- **Bias Detection**: Networks learning to identify and correct opponent prediction biases
- **Cognitive Restructuring**: Adaptation of game strategies based on opponent type learning
- **Behavioral Intervention**: Studying how different loss weightings affect strategy evolution

### Game-Theoretic Analysis
- **Strategy Evolution**: How policies change with different opponent distributions
- **Representation Learning**: What internal representations emerge for different games
- **Transfer Learning**: How learning on one game affects performance on others

### Multi-Task Learning
- **Task Balance**: Optimal weighting of policy, prediction, and estimation objectives
- **Interference**: How auxiliary tasks affect primary policy learning
- **Generalization**: Network performance on unseen opponent types

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{cognitive-therapy-ai,
  title={Cognitive Therapy for AI: Mixed-Motive Game Training Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/nitayalon/Cognitive-Therapy-fo-AI-}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Contact

For questions or collaborations, please contact [your.email@example.com](mailto:your.email@example.com).
