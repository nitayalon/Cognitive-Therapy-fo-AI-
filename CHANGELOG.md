# Cognitive Therapy AI - Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Changed
- **Loss Function Enhancement (2025-10-20)**: Replaced opponent action prediction with opponent policy prediction
  - Modified network to output opponent policy logits instead of cooperation probability
  - Replaced binary cross-entropy loss with KL divergence loss for opponent modeling
  - Enhanced Theory of Mind learning through policy-level understanding
  - Added temperature parameter for policy prediction sharpness (default: 0.1)

### Added
- True opponent policy extraction from opponent type parameters
- KL divergence loss for opponent policy prediction
- Temperature parameter for opponent policy softmax
- Documentation of normalization open issue in MODIFICATIONS.md

- **Training Procedure Enhancement (2025-10-20)**: Replaced probabilistic game sampling with deterministic all-games training
  - Removed `random.random() < game_weights[game_name]` probabilistic sampling
  - Each opponent now trains on ALL games every epoch (guaranteed coverage)
  - Increased training volume: 3x more sessions per epoch for complete environment exposure
  - Enhanced training stability and balanced learning across all games

### Technical Changes
- `GameLSTM.forward()`: Now returns `opponent_policy_logits` instead of `opponent_coop_prob`
- `ToMRLLoss.forward()`: Added `true_opponent_policy` parameter and `temperature` control
- `GameSession`: Enhanced training data to include true opponent policy tensors
- `GameTrainer._train_mixed_epoch()`: Deterministic game coverage instead of probabilistic sampling 

---

## [Previous Changes] - 2025-10-17

### Added
- Custom NumpyJSONEncoder for JSON serialization
- Multi-game simultaneous training capability
- Comprehensive safety checks for data structure handling

### Changed
- Migrated from legacy CompositeLoss to ToMRLLoss architecture
- Updated LossAnalyzer API compatibility
- Fixed gradient flow management with training_mode parameter

### Fixed
- **Key Naming Consistency (Bug Fix)**: Fixed critical KeyError issues across the training pipeline
  - **Loss function keys**: 'opponent_prediction_loss' → 'opponent_policy_loss' and normalized variants
  - **Training data keys**: Fixed mismatch between 'value_estimate' (data) vs 'value_estimates' (expected)
  - **Training data keys**: Fixed mismatch between 'opponent_type_pred' (data) vs 'opponent_type_preds' (expected)
  - **LossAnalyzer**: Updated history dictionary initialization and method key references
  - **Trainer methods**: Updated all loss function calls to use correct data keys
  - **Impact**: Resolves runtime crashes during training, loss calculation, and metrics collection
  - **Files affected**: trainer.py, tom_rl_loss.py
- JSON serialization errors with NumPy float32 types
- Data structure mismatches in experiment reporting
- Tensor conversion issues in advantages calculation

---

## Template for Future Changes

### [Version/Date] - YYYY-MM-DD

#### Loss Function Changes
- **Modified**: [Specific component]
- **Reason**: [Why the change was made]
- **Impact**: [Expected effects on training]
- **Files**: [List of modified files]

#### Training Procedure Changes
- **Modified**: [Training step/process]
- **Parameters**: [New/changed parameters]
- **Behavior**: [How training changes]
- **Files**: [List of modified files]

#### Opponent Changes
- **Added/Modified**: [Opponent strategy]
- **Strategy**: [Description of opponent behavior]
- **Purpose**: [Research objective]
- **Files**: [List of modified files]