# FSM Representation Experiment: Data Collection Specification

## Overview

This document specifies all data collected during training and testing phases of the FSM representation experiments. The data collection system captures complete behavioral trajectories at the timestep level, enabling comprehensive analysis of agent learning, opponent modeling, and generalization.

---

## 🎯 Data Collection Requirements

### ✅ **Implemented Features**
1. ✅ Timestep-level behavioral data
2. ✅ Agent action probabilities  
3. ✅ Opponent action probabilities (for probabilistic opponents)
4. ✅ Both agent and opponent rewards
5. ✅ Value estimates from critic network
6. ✅ Complete observation vectors
7. ✅ Compressed storage (JSONL.GZ format)
8. ✅ Streaming save/load for memory efficiency
9. ✅ Subsampling support (save every N-th episode)

---

## 📊 Data Structure

### **Training Trajectories**

**File:** `experiments/fsm_train_{JOB_ID}/condition_{COND_ID}_seed_{SEED}/trajectories_train.jsonl.gz`

**Format:** One JSON object per line (JSONL), gzip compressed

**Fields Per Timestep:**
```json
{
  "episode": 0,                    // Episode number (0-indexed)
  "timestep": 0,                   // Timestep within episode (0-99)
  
  // Agent data
  "agent_action": 0,               // 0=Cooperate, 1=Defect
  "agent_action_prob": 0.505,      // P(action_taken | state)
  "agent_reward": 0.0,             // Agent's payoff this step
  
  // Opponent data
  "opponent_action": 1,            // 0=Cooperate, 1=Defect
  "opponent_action_prob": 0.700,   // Opponent's policy probability
  "opponent_reward": 5.0,          // Opponent's payoff this step
  
  // Agent internal state
  "value_estimate": -0.14,         // V(s_t) from value head
  "observation": [0.0, 0.0, ...],  // Full observation vector
  
  // Metadata
  "done": false                    // True if end of session (timestep 99)
}
```

**Size:**
- ~200 bytes per timestep (compressed)
- 10K episodes × 100 timesteps = 1M rows
- File size: ~50 MB compressed per training condition

---

### **Testing Trajectories**

**File:** `experiments/fsm_test_{JOB_ID}/model_{MODEL_ID}_test_{GAME}_{OPP}/trajectories_test.jsonl.gz`

**Format:** Identical to training trajectories

**Fields:** Same as training (see above)

**Size:**
- 100 test episodes × 100 timesteps = 10K rows per test condition  
- File size: ~500 KB compressed per test condition

---

## 📁 Complete File Structure

### **Training Phase Output**
```
experiments/fsm_train_{JOB_ID}/
└── condition_{COND_ID}_seed_{SEED}/
    ├── trajectories_train.jsonl.gz       # ← NEW: Timestep-level behavioral data
    ├── train_metrics.json                 # Episode-level summary
    ├── checkpoint.pth                     # Model weights + config
    ├── fsm_train.json                     # FSM structure (states, transitions)
    ├── fidelity_train.json                # FSM fidelity scores
    ├── attribution_profile.json           # Integrated gradients
    ├── best_response_alignment.json       # Nash equilibrium alignment
    └── config.json                        # Experiment configuration
```

### **Testing Phase Output**
```
experiments/fsm_test_{JOB_ID}/
└── model_{MODEL_ID}_test_{GAME}_{OPP}/
    ├── trajectories_test.jsonl.gz        # ← NEW: Test behavioral data
    ├── test_metrics.json                  # Episode-level test summary
    ├── fsm_test.json                      # FSM extracted from test data
    ├── fidelity_test.json                 # FSM fidelity on test
    ├── generalization_report.json         # Train vs test comparison
    └── test_config.json                   # Test condition metadata
```

---

## 🔬 Data Usage Examples

### **1. Cooperation Dynamics Analysis**
```python
from cognitive_therapy_ai.trajectory_utils import load_trajectories_jsonl

# Load training trajectories
traj_file = "experiments/.../trajectories_train.jsonl.gz"

# Compute cooperation rate over time
coop_by_episode = {}
for data in load_trajectories_jsonl(traj_file):
    ep = data['episode']
    if ep not in coop_by_episode:
        coop_by_episode[ep] = []
    coop_by_episode[ep].append(1 - data['agent_action'])  # 0=defect, 1=coop

# Average cooperation per episode
mean_coop = {ep: np.mean(actions) for ep, actions in coop_by_episode.items()}
```

### **2. Policy Confidence Analysis**
```python
# Analyze agent's confidence (action probabilities)
confidence_over_time = []

for data in load_trajectories_jsonl(traj_file, max_episodes=100):
    confidence_over_time.append(data['agent_action_prob'])

# Plot: Does confidence increase with training?
plt.plot(confidence_over_time)
plt.xlabel('Timestep')
plt.ylabel('P(action_taken)')
plt.title('Agent Confidence Over Training')
```

### **3. Reward Structure Analysis**
```python
# Compare agent vs opponent rewards
agent_rewards = []
opponent_rewards = []

for data in load_trajectories_jsonl(traj_file):
    agent_rewards.append(data['agent_reward'])
    opponent_rewards.append(data['opponent_reward'])

# Social welfare: sum of both players' payoffs
social_welfare = np.array(agent_rewards) + np.array(opponent_rewards)
```

### **4. Value Function Learning**
```python
# Track value estimates over training
value_estimates_by_episode = {}

for data in load_trajectories_jsonl(traj_file):
    ep = data['episode']
    if ep not in value_estimates_by_episode:
        value_estimates_by_episode[ep] = []
    value_estimates_by_episode[ep].append(data['value_estimate'])

# Does value function converge?
mean_values = {ep: np.mean(vals) for ep, vals in value_estimates_by_episode.items()}
```

### **5. Cross-Game Generalization**
```python
# Compare behavior on train vs test games
train_traj = load_trajectories_jsonl("trajectories_train.jsonl.gz")
test_traj = load_trajectories_jsonl("trajectories_test.jsonl.gz")

# Extract cooperation rates
train_coop = np.mean([1 - d['agent_action'] for d in train_traj])
test_coop = np.mean([1 - d['agent_action'] for d in test_traj])

print(f"Cooperation: Train={train_coop:.3f}, Test={test_coop:.3f}")
```

---

## 💾 Storage Optimization

### **Subsampling Option**
To reduce storage for large experiments:

```python
from cognitive_therapy_ai.trajectory_utils import save_episode_trajectories

# Save only every 10th episode
save_episode_trajectories(
    episode_trajectories,
    output_path="trajectories_train.jsonl.gz",
    compress=True,
    save_every_nth=10  # Reduces storage by 90%
)
```

### **Storage Estimates**

**Full Experiment (no subsampling):**
- Training: 42 conditions × 10 seeds × 50 MB = **21 GB**
- Testing: 1260 conditions × 500 KB = **630 MB**
- **Total: ~22 GB** (manageable with compression)

**With 10x Subsampling:**
- Training: 42 conditions × 10 seeds × 5 MB = **2.1 GB**
- Testing: 1260 conditions × 50 KB = **63 MB**
- **Total: ~2.2 GB** (recommended for cluster)

---

## 🚀 Integration Status

### ✅ **Implemented**
1. `SessionEnvironment.step()` - Returns opponent data
2. `TrajectoryStep` dataclass - Extended with opponent fields
3. `trajectory_utils.py` - Save/load/analyze utilities
4. `REINFORCETrainer.train_session_rl()` - Optional trajectory return
5. Validation test - `test_trajectory_collection.py` passes

### 🔄 **Next Steps**
1. Integrate into `run_fsm_experiment.py` (new script)
2. Add CLI flags: `--save-trajectories`, `--save-every-nth`
3. Update SLURM scripts to enable trajectory saving
4. Test on cluster with small job (1-2 conditions)

---

## 📖 API Reference

### **Core Functions**

#### `save_episode_trajectories()`
```python
def save_episode_trajectories(
    episode_trajectories: List[List[TrajectoryStep]],
    output_path: Path,
    compress: bool = True,
    save_every_nth: int = 1
) -> int
```
Save trajectories from multiple episodes with optional subsampling.

#### `load_trajectories_jsonl()`
```python
def load_trajectories_jsonl(
    file_path: Path,
    max_episodes: int = None
) -> Iterator[Dict[str, Any]]
```
Load trajectories from JSONL file (generator for memory efficiency).

#### `compute_trajectory_statistics()`
```python
def compute_trajectory_statistics(
    file_path: Path
) -> Dict[str, Any]
```
Compute summary statistics from trajectory file.

**Returns:**
```python
{
    'num_episodes': 100,
    'total_timesteps': 10000,
    'mean_episode_reward': 159.6,
    'std_episode_reward': 15.8,
    'mean_agent_cooperation': 0.502,
    'mean_opponent_cooperation': 0.314
}
```

---

## ✅ Validation Results

**Test:** `python test_trajectory_collection.py`

**Output:**
```
================================================================================
✅ ALL TESTS PASSED
================================================================================

Trajectory collection system is working correctly!
Ready to integrate into experiment scripts.

Test Results:
- 5 episodes × 100 timesteps = 500 datapoints collected
- All 11 required fields present
- Data types validated
- Opponent probabilities match expected distribution
- Compressed file size: 12.9 KB (25.8 bytes/timestep)
- Save/load roundtrip successful
- Statistics computation working
```

---

## 📚 Related Files

- **Implementation:** `src/cognitive_therapy_ai/trajectory_utils.py`
- **Core Infrastructure:** `src/cognitive_therapy_ai/reinforce_trainer.py`  
- **Validation Test:** `test_trajectory_collection.py`
- **Sample Data:** `results/test_trajectory_collection/trajectories_train.jsonl.gz`

---

## 🎓 Research Applications

This data enables analysis of:
1. **Learning Dynamics** - How cooperation/defection rates evolve
2. **Policy Confidence** - Action probability distributions over training
3. **Value Function Accuracy** - Comparison of V(s) to actual returns
4. **Opponent Modeling** - Implicit ToM via action selection patterns
5. **Generalization** - Behavioral shifts when tested on new opponents/games
6. **Social Dilemmas** - Joint payoff optimization vs individual gain
7. **FSM Fidelity Sources** - Why some policies are non-deterministic
8. **Attribution Validation** - Linking salient features to action probabilities

---

**Status:** ✅ **COMPLETE AND VALIDATED**  
**Next:** Integrate into `run_fsm_experiment.py` and deploy to cluster
