# Cognitive Therapy AI - Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **Complete Training & Testing Documentation System (2025-11-13)**: Comprehensive monitoring pipeline
  - **Training Phase Monitoring**: 
    - New `TrainingMonitor` and `BatchedTrainingMonitor` classes for detailed logging
    - Logs all loss components (RL, opponent prediction, total) for each training iteration
    - Records network head outputs (policy logits, opponent policy logits, value estimates)
    - Tracks sampled actions from network policy and opponent actions
    - Documents rewards received by both agent and opponent
    - Real-time table updates saved as CSV and Excel files every 50-100 iterations
    - Automatic gradient norm tracking and logging
    - Integration with both single-game and multi-game training modes
    - Files: `src/cognitive_therapy_ai/training_monitor.py`, enhanced `trainer.py`
  
  - **Testing Phase Monitoring**:
    - New `TestingMonitor` class for comprehensive evaluation documentation
    - Logs network predictions vs true opponent behavior during testing
    - Records policy outputs, value estimates, and action distributions
    - Tracks sampled actions for both agent and opponent during evaluation
    - Documents rewards and game outcomes for each test step
    - Session-based organization with detailed game and opponent information
    - CSV and Excel output with comprehensive column schemas
    - Files: `src/cognitive_therapy_ai/testing_monitor.py`
  
  - **Network Serial ID System**:
    - UUID-based network identification for linking training and testing data
    - Automatic serial ID generation and tracking throughout training lifecycle
    - Linkage files connecting training logs to testing logs for analysis
    - Enhanced trainer with network identification methods
    - Complete data traceability from training through evaluation phases
  
  - **Enhanced Evaluation Methods**:
    - Updated `evaluate()` and `evaluate_on_multiple_games()` methods with monitoring
    - Detailed step-by-step logging during evaluation sessions
    - Automatic monitoring enabling when directories are provided
    - Comprehensive session management and result aggregation
  
  - **Demo and Configuration**:
    - New `complete_monitoring_demo.py` demonstrating full monitoring pipeline
    - Enhanced VS Code launch configurations for testing new features
    - Automatic monitoring activation when save directories are provided

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

- **Interim Training Reports (2025-11-02)**: Enhanced multi-game training with progress monitoring
  - Added interim summary reports every 100 epochs during multi-game training
  - Reports agent policy probabilities (averaged over last 10 epochs)
  - Reports predicted opponent policy probabilities (averaged over last 10 epochs)
  - Reports average reward performance (averaged over last 10 epochs)
  - All metrics logged to INFO level for training visibility

### Fixed
- **Legacy Code Cleanup (2025-11-02)**: Removed outdated opponent type prediction tensors
  - Removed `opponent_type_pred` tensor creation in GameSession (line 173)
  - Removed references to `opponent_type_preds` in session data compilation
  - Aligned code with documented policy-based opponent modeling approach
  - Training data now only includes `true_opponent_policy` as documented

- **Training Configuration (2025-11-02)**: Disabled early termination for full epoch training
  - Modified `_check_convergence()` to always return False
  - Training now runs for complete `config.max_epochs` without early stopping
  - Loss tracking still maintained for metrics, but no convergence-based termination
  - Ensures consistent training duration across all experiments

- **High Priority Opponent Policy Learning Fixes (2025-11-02)**:
  - **Data Pipeline Consistency**: Fixed opponent policy tensor format to match network output
    - Changed from `[1-p_d, p_d]` to `[p_d, 1-p_d]` format (defect, cooperate)
    - Added detailed consistency logging for debugging
  - **Enhanced Loss Monitoring**: Added comprehensive loss component analysis
    - Tracks RL vs opponent policy loss ratios and alpha contributions  
    - Detailed prediction vs ground truth logging for opponent policy
  - **Network Architecture Improvements**: Enhanced opponent policy head
    - Added LayerNorm and additional depth for better opponent modeling
    - Specialized weight initialization with reduced gains for stability
  - **Advanced Diagnostics**: Added prediction analysis tools and enhanced interim reports
    - Network analysis method with temperature sensitivity testing
    - Enhanced interim summaries with loss ratios and prediction entropy

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