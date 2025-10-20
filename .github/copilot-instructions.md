# Cognitive Therapy for AI - Development Guide

This framework implements **Theory of Mind Reinforcement Learning (ToM-RL)** for training LSTM networks on mixed-motive games. The core architecture combines policy learning with opponent behavior prediction to create Theory of Mind inductive bias.

## Architecture Overview

### Core Game Loop: Session-Based Training
- **GameSession**: Plays T consecutive games with same opponent (e.g., T=100)
- **GameTrainer**: Orchestrates multiple sessions across different opponent types
- Networks maintain LSTM hidden state across all T games within a session
- State resets between sessions but persists within sessions

### Multi-Task Loss Function (ToM-RL)
The framework uses a **dual-objective loss** combining:
1. **LRL** (Reinforcement Learning Loss): Policy gradient with advantage estimation
2. **LOp** (Opponent Prediction Loss): Binary cross-entropy for predicting opponent cooperation

```python
# Total Loss: L = LRL_norm + α * LOp_norm
# See tom_rl_loss.py - this is the main loss, not loss.py (legacy)
```

### State Representation Pattern
All games use **standardized 5-element state vectors**:
```python
# [payoff_matrix_flattened(4), round_number_normalized(1)]
# payoff_matrix: [[coop_vs_coop, coop_vs_defect], [defect_vs_coop, defect_vs_defect]]
```

## Key Development Patterns

### Game Implementation
All games extend `MixedMotiveGame` with standardized `get_payoff_matrix()`:
```python
# Example: PrisonersDilemma payoff matrix
return np.array([
    [self.R, self.S],  # Cooperate vs [Cooperate, Defect]
    [self.T, self.P]   # Defect vs [Cooperate, Defect]
], dtype=np.float32)
```

### Network Architecture (GameLSTM)
- **Multi-head output**: policy_logits, opponent_coop_prob, value_estimate
- **Input size**: Always 5 (payoff matrix + round number)
- **Hidden state persistence**: Maintained across T games within sessions
- Use `network.py` - this contains the main ToM-RL architecture

### Configuration System
- **Hierarchical configs**: NetworkConfig, TrainingConfig, ExperimentConfig
- **JSON-based**: See `config/default_config.json` for structure
- **Command-line override**: `main_experiment.py` supports both config files and CLI args

## Critical File Relationships

### Loss Function Evolution
```
loss.py          <- Legacy 3-task loss (kept for compatibility)
tom_rl_loss.py   <- Current ToM-RL implementation (ACTIVELY USED)
```

### Training Flow
```
main_experiment.py -> GameTrainer -> GameSession -> GameLSTM
                   -> ToMRLLoss (modernized implementation)
```

### Session Modes
```
GameSession(training_mode=True)  <- Gradients enabled for training
GameSession(training_mode=False) <- Gradients disabled for evaluation
```

## Development Commands

### Quick Testing
```bash
# Framework verification
python test_framework.py

# Quick experiment (5-min test)
python main_experiment.py --config config/quick_test_config.json

# Single game test
python main_experiment.py --game prisoners-dilemma --opponents 0.5 --num-games 10
```

### Experiment Patterns
```bash
# Full experiment suite (all 4 games, 5 opponent types)
python main_experiment.py --game all --opponents 0.1,0.3,0.5,0.7,0.9

# Custom configuration
python main_experiment.py --config my_config.json --adaptive-loss
```

### Output Structure
```
experiments/mixed_motive_experiment_YYYYMMDD_HHMMSS/
├── checkpoints/          # Model states (.pth files)
├── logs/                 # Training logs
├── plots/                # Matplotlib figures  
├── results/              # Pickled results per game
└── experiment_config.json
```

## Code Conventions

### Import Pattern
```python
# Always import from package root
from cognitive_therapy_ai import GameFactory, GameLSTM, GameTrainer
from cognitive_therapy_ai.tom_rl_loss import ToMRLLoss, AdaptiveToMRLLoss  # Current implementation
```

### Training Data Structure (ToM-RL)
```python
# Expected tensors for ToM-RL loss:
training_data = {
    'policy_logits': torch.Tensor,      # (batch_size, num_actions)
    'opponent_coop_probs': torch.Tensor, # (batch_size, 1)
    'value_estimates': torch.Tensor,     # (batch_size, 1)  
    'actions': torch.Tensor,            # (batch_size,)
    'rewards': torch.Tensor,            # (batch_size,)
    'opponent_actions': torch.Tensor    # (batch_size,)
}
```

### Device Handling
```python
# Framework auto-detects CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Opponent Specification
```python
# Opponents defined by defection probability (p ∈ [0,1])
opponent = OpponentFactory.create_opponent('probabilistic', defection_prob=0.3)
```

## Extension Points

### Adding New Games
1. Extend `MixedMotiveGame` 
2. Implement `get_payoff_matrix()` returning 2x2 numpy array
3. Register in `GameFactory.create_game()`
4. Add config section to default_config.json

### Custom Loss Functions
- Extend `ToMRLLoss` (not CompositeLoss)
- Maintain normalization for stable multi-task training
- Consider α parameter for RL vs ToM balance

### New Opponent Strategies  
1. Extend `OpponentStrategy`
2. Implement `choose_action(game_history, round_number)`
3. Register in `OpponentFactory`

## Debugging Notes

- **Training instability**: Check loss normalization in ToMRLLoss
- **Memory issues**: Reduce batch_size or hidden_size in configs
- **Import errors**: Verify `src/` is in Python path for development
- **Convergence issues**: Adjust learning rate, patience, or convergence_threshold
- **Experiment crashes**: Use `--config config/quick_test_config.json` for debugging

## Research Context

This framework studies **cognitive therapy analogies** in AI:
- **Bias Detection**: Networks learning opponent prediction
- **Cognitive Restructuring**: Strategy adaptation based on opponent understanding  
- **Multi-task Learning**: Balancing reward optimization with Theory of Mind development

The ToM-RL approach creates inductive bias for understanding opponent behavior patterns, analogous to Theory of Mind in cognitive science.