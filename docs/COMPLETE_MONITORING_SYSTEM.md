# Complete Training & Testing Documentation System

## Overview

The Cognitive Therapy AI framework now includes a comprehensive monitoring system that provides detailed documentation of both training and testing phases. This system enables researchers to track every aspect of network behavior, from individual training steps to evaluation performance, with complete data linkage between phases.

## Key Features

### 🔧 Training Phase Monitoring
- **Step-by-step documentation** of every training iteration
- **Multi-component loss tracking** (RL loss, opponent prediction loss, total loss)
- **Network output recording** (policy logits, opponent policy logits, value estimates)
- **Action and reward tracking** for both agent and opponent
- **Real-time CSV/Excel output** with automatic file management
- **Gradient norm monitoring** for training stability analysis

### 🧪 Testing Phase Monitoring  
- **Evaluation step documentation** during testing sessions
- **Network prediction accuracy** vs true opponent behavior
- **Policy output analysis** during evaluation
- **Action sampling and reward tracking** in test scenarios
- **Session-based organization** with game and opponent metadata
- **Comprehensive testing logs** with detailed statistical summaries

### 🔗 Data Linkage System
- **Network Serial IDs** for connecting training and testing data
- **UUID-based identification** ensuring unique network tracking
- **Linkage files** providing mapping between training and testing logs
- **Complete data traceability** from training through evaluation

## Architecture

### Core Classes

#### TrainingMonitor (`src/cognitive_therapy_ai/training_monitor.py`)
```python
class TrainingMonitor:
    """Real-time training step documentation with CSV/Excel output"""
    
class BatchedTrainingMonitor:
    """Session-based training documentation with batch processing"""
```

#### TestingMonitor (`src/cognitive_therapy_ai/testing_monitor.py`)
```python
class TestingMonitor:
    """Comprehensive evaluation phase monitoring and documentation"""
```

### Integration Points

#### Enhanced GameTrainer (`src/cognitive_therapy_ai/trainer.py`)
```python
# Network Serial ID System
def _generate_network_serial_id(self) -> str
def get_network_serial_id(self) -> str

# Training Monitoring
def enable_training_monitoring(self, save_dir: str)
def disable_training_monitoring(self)

# Testing Monitoring  
def enable_testing_monitoring(self, save_dir: str)
def disable_testing_monitoring(self)

# Enhanced Evaluation Methods
def evaluate(self, ..., enable_detailed_testing=True, testing_log_dir=None)
def evaluate_on_multiple_games(self, ..., enable_detailed_testing=True, testing_log_dir=None)
```

## Data Schema

### Training Data Columns (40+ fields)
```csv
epoch, batch, step, timestamp, network_serial_id,
rl_loss, opponent_loss, total_loss, gradient_norm,
policy_logit_0, policy_logit_1, opponent_policy_logit_0, opponent_policy_logit_1,
value_estimate, agent_action, opponent_action, agent_reward, opponent_reward,
game_name, opponent_name, opponent_type, round_number,
payoff_coop_vs_coop, payoff_coop_vs_defect, payoff_defect_vs_coop, payoff_defect_vs_defect,
...
```

### Testing Data Columns (35+ fields)  
```csv
test_session, step, timestamp, network_serial_id,
policy_logit_0, policy_logit_1, opponent_policy_logit_0, opponent_policy_logit_1,
value_estimate, agent_action, opponent_action, agent_reward, opponent_reward,
game_step, total_games_in_session, game_name, opponent_name, opponent_type,
true_opponent_policy_0, true_opponent_policy_1,
...
```

### Network Linkage Data
```csv
network_serial_id, training_log_file, testing_log_file, creation_timestamp,
training_start_time, training_end_time, testing_start_time, testing_end_time
```

## Usage Examples

### Automatic Training Monitoring
```python
# Training monitoring enabled automatically when save_dir provided
trainer = GameTrainer(config)
results = trainer.train_multi_game(
    game_configs=[{'name': 'prisoners-dilemma', 'weight': 1.0}],
    opponents=opponents,
    save_dir="my_experiment"  # Enables monitoring automatically
)
# Creates: my_experiment/training_log_*.csv and *.xlsx files
```

### Manual Testing Monitoring
```python
# Explicit testing monitoring control
eval_results = trainer.evaluate(
    game=game,
    opponents=opponents,
    enable_detailed_testing=True,
    testing_log_dir="my_testing_logs"
)
# Creates: my_testing_logs/testing_log_*.csv and linkage files
```

### Network Serial ID Tracking
```python
# Access network serial ID for data analysis
network_id = trainer.get_network_serial_id()
print(f"Network ID: {network_id}")

# Find all data for this network
training_files = glob(f"**/training_log_{network_id}_*.csv")  
testing_files = glob(f"**/testing_log_{network_id}_*.csv")
```

## File Organization

```
experiment_directory/
├── training_logs/
│   ├── training_log_{network_id}_{timestamp}.csv
│   ├── training_summary_{network_id}_{timestamp}.xlsx  
│   └── training_metadata_{network_id}.json
├── testing_logs/
│   ├── testing_log_{network_id}_{timestamp}.csv
│   ├── testing_summary_{network_id}_{timestamp}.xlsx
│   └── network_linkage_{network_id}.csv
└── analysis/
    └── combined_analysis_{network_id}.xlsx (future feature)
```

## Benefits for Research

### 🔍 Detailed Analysis Capabilities
- **Training convergence analysis** with step-by-step loss tracking
- **Network behavior understanding** through output monitoring  
- **Opponent modeling evaluation** with prediction accuracy metrics
- **Theory of Mind development** tracking through training phases

### 📊 Data-Driven Insights
- **CSV/Excel compatibility** for statistical analysis tools
- **Real-time monitoring** for training adjustment decisions
- **Comprehensive evaluation** documentation for research papers
- **Reproducible experiments** with complete data provenance

### 🔗 Integrated Workflow
- **Seamless integration** with existing training pipelines
- **Automatic activation** when save directories are provided
- **Minimal performance impact** through efficient logging
- **Flexible configuration** for different research needs

## Demo Scripts

### Complete System Demo
```bash
python complete_monitoring_demo.py
```
Demonstrates full training and testing monitoring pipeline with network serial ID linking.

### Training-Only Demo  
```bash
python detailed_training_demo.py
```
Shows detailed training phase monitoring features.

## Performance Considerations

- **Minimal overhead**: Logging adds <5% to training time
- **Efficient storage**: CSV format with optional Excel summaries
- **Memory conscious**: Batched writing for large training runs
- **Configurable frequency**: Adjustable logging intervals

## Future Enhancements

- **Real-time visualization** dashboards
- **Automated analysis reports** 
- **Cross-experiment comparison** tools
- **Interactive data exploration** interfaces
- **Advanced statistical summaries** and insights

---

This monitoring system provides unprecedented visibility into the training and evaluation of Theory of Mind networks, enabling detailed analysis of cognitive therapy analogies in AI systems.