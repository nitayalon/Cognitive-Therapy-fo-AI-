# Cognitive Therapy for AI - Experiment Design & Methods Manual

**Version**: 1.0  
**Date**: February 22, 2026  
**Status**: Active Framework Design Document

---

## Table of Contents
1. [Research Overview](#1-research-overview)
2. [Theoretical Framework](#2-theoretical-framework)
3. [Research Questions](#3-research-questions)
4. [Hypotheses](#4-hypotheses)
5. [Experiment Design](#5-experiment-design)
6. [Methods](#6-methods)
7. [Metrics & Evaluation](#7-metrics--evaluation)
8. [Expected Outcomes](#8-expected-outcomes)
9. [Analysis Plan](#9-analysis-plan)
10. [Implementation Guidelines](#10-implementation-guidelines)

---

## 1. Research Overview

### 1.1 Project Goal
This project aims to explore the causal relation between train environment features and the (in)ability to generalize. In particular, this work models maladaptation as generalization error. Our paradigm trains RL agents in various well knoe mixed-motive games against well defined opponents, and then test (generalize) the learning to new games and/or new opponents.  

### 1.2 Key Innovation
This framework uses two (nested) multi-agent paradigms to explore the realtion between the complexity of the training environment and the learner's ability (or lack of it) to successfuly interact with other agents in unseen ineractions.
We compare between opponet unaware models (plain RRN, trained to maximize reward) and opponent aware (proto-Theory of Mind) models, in which the agents is also trained to predict others' future actinos and reward (learning their incentives).
Our analysis is focused on:
- Representation failure (maladaptiveness) as a function of the traning task charectaristics (complexity, number of Nash-Equillibria)
- Generalization across game structures and opponent types
- The role of proto Theory of Mind in social decision-making

### 1.3 Cognitive Therapy Analogy
[Explain the cognitive therapy metaphor and its relevance to AI systems]

**Core Analogies**:
- **Bias Detection**: Networks learning opponent prediction → Identifying cognitive distortions
- **Cognitive Restructuring**: Strategy adaptation based on opponent understanding → Reframing maladaptive beliefs
- **Behavioral Intervention**: Multi-task learning balancing policy and ToM → Therapeutic techniques

---

## 2. Theoretical Framework

### 2.1 Theory of Mind in AI
[Describe the computational role of Theory of Mind and how it's implemented]

**Definition**: Theory of Mind (ToM) is the capacity to attribute mental states (beliefs, intentions, desires) to others and use these attributions to predict behavior.

**Implementation**: The auxiliary opponent prediction task creates an inductive bias for developing internal representations of opponent behavioral patterns.

### 2.2 Agent Architectures: Vanilla RL vs Proto-ToM

This framework implements **two types of agents** to study the role of social inductive biases in generalization:

#### 2.2.1 Vanilla RL Agent (Baseline)

**Objective**: Pure reward maximization without social awareness

**Loss Function**:
```
L_vanilla = LRL
where LRL = -E_t[log π(a_t|s_t) * Â_t]
      Â_t = R_t - V_θ(s_t)  (advantage estimate)
```

**Characteristics**:
- No opponent modeling or prediction
- Pure policy gradient learning
- Opponent-unaware: treats environment as stationary
- Baseline for measuring ToM contribution

#### 2.2.2 Proto-ToM Agent (Multi-Task)

**Objective**: Reward maximization with social auxiliary tasks

**Primary Task (LRL)**: Policy gradient with advantage estimation (same as vanilla)

**Auxiliary Task 1 (LOp_action)**: Opponent action prediction
```
LOp_action = -Σ_t [o_t * log(p̂_t^action) + (1 - o_t) * log(1 - p̂_t^action)]
where o_t ∈ {0,1} is opponent's action (cooperate/defect)
```

**Auxiliary Task 2 (LOp_reward)**: Opponent reward prediction
```
LOp_reward = MSE(r̂_opponent, r_opponent)
where r_opponent is the actual reward received by opponent
```

**Purpose of Auxiliary Tasks**:
- **Opponent Action Prediction**: Encourages modeling opponent behavioral patterns
- **Opponent Reward Prediction**: Encourages learning causal mapping from game structure to incentives
- Together form **proto-Theory of Mind**: shallow mentalizing about opponent's decisions and goals

**Combined Loss**: 
```
L_proto-ToM = LRL_norm + α * LOp_action_norm + β * LOp_reward_norm
```

**Note**: Loss components are normalized to [0,1] range for balanced multi-task learning

### 2.3 Mixed-Motive Games as Test Environments

**Why Mixed-Motive Games**:
- Balance between cooperation and competition
- Opponent behavior critically affects optimal strategy
- Rich strategic space for studying adaptation

**Game Suite**:
1. **Prisoner's Dilemma**: Classic cooperation dilemma (dominant defection)
2. **Hawk-Dove**: Resource competition with costly conflict
3. **Stag Hunt**: Coordination with risk (assurance game)
4. **Battle of the Sexes**: Coordination with preference conflict

---

## 3. Research Questions

### 3.1 Primary Research Questions

**RQ1**: Does proto-Theory of Mind improve generalization performance compared to vanilla RL agents?
- Sub-question: Under which test conditions (same-game/new-opponents, new-game/same-opponents, new-game/new-opponents) does proto-ToM provide the largest advantage?

**RQ2**: What is the relative difficulty of game generalization vs. opponent generalization?
- For vanilla agents: [To be specified]
- For proto-ToM agents: [To be specified]
- Does proto-ToM differentially affect these two generalization axes?

**RQ3**: How does training distribution (game type × opponent range) affect out-of-distribution generalization?
- Which training conditions produce the most robust agents?
- Do certain game structures promote better transfer?
- Does opponent diversity during training improve generalization?

### 3.2 Secondary Research Questions

**RQ4**: What is the relationship between opponent prediction accuracy and task performance?
- Does better opponent modeling correlate with higher rewards?
- Is this relationship consistent across test conditions?

**RQ5**: How do the two auxiliary tasks (action prediction vs. reward prediction) contribute to proto-ToM agent performance?
- Which auxiliary task is more predictive of generalization success?
- Do they interact synergistically?

---

## 4. Hypotheses

### 4.1 Proto-ToM Advantage Hypotheses

**H1**: Proto-ToM agents will show smaller generalization error than vanilla RL agents across all out-of-distribution test conditions
- **Prediction**: Δ Performance_proto-ToM > Δ Performance_vanilla for OOD conditions
- **Rationale**: Opponent modeling provides inductive bias for understanding social interactions that transfers across contexts
- **Test**: Two-way ANOVA (agent_type × test_condition) on generalization error; main effect of agent_type

**H2**: Proto-ToM advantage will be largest for opponent generalization (same-game/new-opponents)
- **Prediction**: (Vanilla - Proto-ToM) generalization gap is largest when only opponents change
- **Rationale**: Auxiliary tasks directly target opponent understanding, providing maximal benefit when game structure is familiar
- **Test**: Interaction effect in ANOVA; post-hoc comparison of proto-ToM advantage across test conditions

### 4.2 Generalization Structure Hypotheses

**H3**: Opponent generalization is easier than game generalization for both agent types
- **Prediction**: Generalization_error(same-game/new-opponents) < Generalization_error(new-game/same-opponents)
- **Rationale**: Opponent strategies are simpler to model (stateless probabilistic) than game structures (different payoff matrices)
- **Test**: Paired t-test comparing the two generalization conditions within each agent type

**H4**: Proto-ToM agents will show reduced catastrophic interference compared to vanilla agents
- **Prediction**: Proto-ToM in-distribution performance remains stable; vanilla performance may degrade
- **Rationale**: Auxiliary tasks regularize representations, preventing overfitting to specific opponent types
- **Test**: Compare in-distribution performance variability between agent types

### 4.3 Auxiliary Task Contribution Hypotheses

**H5**: Opponent reward prediction will be more predictive of generalization success than opponent action prediction
- **Prediction**: Correlation(LOp_reward_accuracy, generalization_performance) > Correlation(LOp_action_accuracy, generalization_performance)
- **Rationale**: Learning incentive structures (reward prediction) captures deeper game understanding than surface behavior (action prediction)
- **Test**: Regression analysis predicting generalization from both auxiliary task accuracies; compare standardized coefficients

**H6**: Proto-ToM agents trained on high opponent diversity ranges will generalize better than those trained on narrow ranges
- **Prediction**: Agents trained on [0.1, 0.9] generalize better than agents trained on [0.4, 0.6]
- **Rationale**: Diverse training opponents provide richer signal for auxiliary task learning
- **Test**: Compare average generalization error across training conditions grouped by opponent range width

### 4.4 Training Distribution Hypotheses

**H7**: Game structure complexity affects proto-ToM advantage differentially
- **Prediction**: Proto-ToM advantage is larger when trained on games with multiple Nash equilibria (Stag Hunt, BoS)
- **Rationale**: Complex games require deeper opponent understanding, amplifying proto-ToM benefits
- **Test**: ANOVA with game_type as factor; examine proto-ToM advantage by training game 

---

## 5. Experiment Design

### 5.0 Experimental Paradigm: Two-Agent Comparison

**Core Design**: Each training condition is run with **both agent types**:

1. **Vanilla RL Agent**: Trained only to maximize reward (baseline)
2. **Proto-ToM Agent**: Trained with reward maximization + social auxiliary tasks

**Comparison Structure**:
```
For each training condition (game × opponent_range):
    Train Vanilla Agent → Test on all 16 conditions → Measure generalization
    Train Proto-ToM Agent → Test on all 16 conditions → Measure generalization
    Compare: Vanilla vs Proto-ToM performance across test conditions
```

**Result**: 
- 16 training conditions × 2 agent types = **32 trained agents**
- 32 agents × 16 test conditions = **512 generalization measurements**
- Direct comparison of vanilla vs proto-ToM generalization patterns

### 5.1 Training Phase: Single-Game, Single-Opponent-Range

**Training Paradigm**:
- **Single Game**: One game from {Prisoner's Dilemma, Hawk-Dove, Stag Hunt, Battle of Sexes}
- **Single Opponent Range**: Fixed probability range (e.g., [0.1, 0.3] = low cooperation opponents)
- **Session Structure**: T consecutive games with same opponent (T = 100 default)
- **LSTM State**: Persists across all T games within session, resets between opponents

**Training Schedule**:
```
For each epoch:
    For each opponent in training_set:
        Initialize LSTM hidden state
        Play T games maintaining hidden state
        Accumulate gradients from all T games
        Update network parameters
```

### 5.2 Test Phase: Four-Condition Generalization

**Baseline Condition**: In-Distribution Performance
- **Game**: Same as training
- **Opponents**: Same range as training
- **Purpose**: Measure in-distribution capability

**Condition 1**: Same Game, New Opponents (Opponent Generalization)
- **Game**: Same as training
- **Opponents**: Different range (e.g., [0.7, 0.9] if trained on [0.1, 0.3])
- **Purpose**: Isolate opponent-type generalization

**Condition 2**: New Game, Same Opponents (Game Generalization)
- **Game**: Different from training
- **Opponents**: Same range as training
- **Purpose**: Isolate game-structure generalization

**Condition 3**: New Game, New Opponents (Full Out-of-Distribution)
- **Game**: Different from training
- **Opponents**: Different range from training
- **Purpose**: Measure combined generalization challenge

### 5.3 Generalization Matrix: Full Factorial Design

**Matrix Structure**: 4 Games × 4 Opponent Ranges = 16 Training Conditions

**Games**:
1. Prisoner's Dilemma (PD)
2. Hawk-Dove (HD)
3. Stag Hunt (SH)
4. Battle of Sexes (BoS)

**Opponent Ranges**:
1. Low cooperation: [0.1, 0.3]
2. Mid-low cooperation: [0.3, 0.5]
3. Mid-high cooperation: [0.5, 0.7]
4. High cooperation: [0.7, 0.9]

**Training Conditions** (each gets one agent):
```
Condition  0: PD  + Low      → Test on all 16 game-opponent combinations
Condition  1: PD  + Mid-Low  → Test on all 16 game-opponent combinations
Condition  2: PD  + Mid-High → Test on all 16 game-opponent combinations
Condition  3: PD  + High     → Test on all 16 game-opponent combinations
Condition  4: HD  + Low      → Test on all 16 game-opponent combinations
...
Condition 15: BoS + High     → Test on all 16 game-opponent combinations
```

**Result**: 16 agents × 16 test conditions = 256 generalization measurements

### 5.4 Opponent Model

**Opponent Type**: Probabilistic strategy with fixed defection probability `p_d`

**Behavior**:
```python
p(Defect) = p_d
p(Cooperate) = 1 - p_d
```

**Characteristics**:
- Memoryless (stateless)
- Stochastic with fixed probabilities
- Type parameter `p_d` ∈ [0, 1] determines cooperation level

**Training Set**: Multiple opponents sampled from specified range
- Example: Range [0.1, 0.3] → 11 equally spaced opponents: {0.1, 0.12, 0.14, ..., 0.3}

---

## 6. Methods

### 6.1 Network Architecture

**Model**: GameLSTM (Recurrent Neural Network)

**Shared Components** (both agent types):

**Input** (5 dimensions):
```python
state = [
    payoff_coop_coop,     # Payoff when both cooperate
    payoff_coop_defect,   # Payoff when agent cooperates, opponent defects
    payoff_defect_coop,   # Payoff when agent defects, opponent cooperates
    payoff_defect_defect, # Payoff when both defect
    round_number_normalized  # Current round / T
]
```

**LSTM Core**:
```
Input (5) → LSTM (hidden_size × num_layers) → Hidden Representation
```

**Default Hyperparameters**:
- `hidden_size = 128`
- `num_layers = 2`
- `dropout = 0.1`

---

#### 6.1.1 Vanilla RL Agent Architecture

**Output Heads**:
```
LSTM Hidden → Policy Head (2 logits: cooperate/defect)
           └─> Value Head (1 value: state value estimate)
```

**No opponent modeling heads** - pure policy optimization

---

#### 6.1.2 Proto-ToM Agent Architecture

**Output Heads**:
```
LSTM Hidden → Policy Head (2 logits: cooperate/defect)
           ├─> Value Head (1 value: state value estimate)
           ├─> Opponent Action Head (2 logits: opponent cooperate/defect)
           └─> Opponent Reward Head (1 value: predicted opponent reward)
```

**Additional heads for social auxiliary tasks**:
- **Opponent Action Head**: Predicts opponent's next action
- **Opponent Reward Head**: Predicts opponent's expected reward (incentive learning)

### 6.2 Loss Functions

**File**: `src/cognitive_therapy_ai/tom_rl_loss.py`

---

#### 6.2.1 Vanilla RL Agent Loss

**Single-Task Objective**: Pure reward maximization

```python
# 1. Policy Gradient Loss
advantages = rewards - value_estimates  # or GAE if use_gae=True
policy_loss = -log_probs * advantages.detach()
LRL = policy_loss.mean()

# 2. Value Function Loss (for advantage estimation)
value_loss = MSE(value_estimates, rewards)

# Total Loss
L_vanilla = LRL + value_loss
```

**Characteristics**:
- No auxiliary tasks
- Standard actor-critic architecture
- Baseline for measuring ToM contribution

---

#### 6.2.2 Proto-ToM Agent Loss

**Multi-Task Objective**: Reward + Social Auxiliary Tasks

```python
# 1. Reinforcement Learning Loss (same as vanilla)
advantages = rewards - value_estimates
policy_loss = -log_probs * advantages.detach()
LRL = policy_loss.mean()

# 2. Opponent Action Prediction Loss
LOp_action = BCE(opponent_action_probs, opponent_actions)
# Binary cross-entropy for predicting opponent's next action

# 3. Opponent Reward Prediction Loss  
LOp_reward = MSE(predicted_opponent_rewards, actual_opponent_rewards)
# Mean squared error for learning opponent's incentive structure

# 4. Value Function Loss
value_loss = MSE(value_estimates, rewards)

# Normalize auxiliary losses to [0, 1] range for stable multi-task learning
L_total = LRL + LOp_action + LOp_reward
LRL_norm = LRL / L_total
LOp_action_norm = LOp_action / L_total
LOp_reward_norm = LOp_reward / L_total

# Combined Loss with weighting
L_proto-ToM = LRL_norm + α * LOp_action_norm + β * LOp_reward_norm + value_loss
```

**Loss Weighting Parameters**:
- **α (alpha)**: Weight for opponent action prediction (default: 1.0)
- **β (beta)**: Weight for opponent reward prediction (default: 1.0)
- **Adaptive mode**: α and β adjust dynamically based on relative loss magnitudes

**Design Rationale**:
- **Opponent Action Prediction**: Creates inductive bias for opponent behavior modeling
- **Opponent Reward Prediction**: Encourages learning game structure and opponent incentives
- **Normalization**: Ensures balanced multi-task learning across different loss scales

### 6.3 Training Protocol

**Optimizer**: Adam
- Learning rate: 0.001
- Default PyTorch β parameters

**Convergence Criteria**:
- **Loss threshold**: Change < 1e-6 between epochs
- **Patience**: 50 epochs without improvement
- **Max epochs**: 500

**Training Loop**:
```python
for epoch in range(max_epochs):
    for opponent in training_opponents:
        # Initialize hidden state
        hidden = network.init_hidden()
        
        # Play T games with this opponent
        for game_round in range(T):
            state = game.get_state()
            action, hidden = network(state, hidden)
            reward = game.step(action, opponent.act())
            
        # Accumulate loss
        loss = compute_tom_rl_loss(...)
    
    # Update parameters
    optimizer.step()
    
    # Check convergence
    if converged:
        break
```

**Batch Processing**: 
- Batch size: 32 (default)
- Batches created from collected game experiences

### 6.4 Evaluation Protocol

**Test Sessions**: 20 evaluation sessions per opponent (default)

**Metrics Collected** (both agent types):
- Average reward per game
- Cooperation rate (proportion of cooperative actions)
- Policy entropy (measure of exploration vs exploitation)
- Win rate vs opponent
- Mutual cooperation/defection rates

**Additional Metrics** (proto-ToM agents only):
- Opponent action prediction accuracy
- Opponent reward prediction error (MSE)
- Auxiliary task loss values

**Evaluation Procedure**:
```python
for test_condition in all_test_conditions:
    # Evaluate vanilla agent
    vanilla_results = evaluate(vanilla_agent, test_condition)
    
    # Evaluate proto-ToM agent  
    proto_tom_results = evaluate(proto_tom_agent, test_condition)
    
    # Compute proto-ToM advantage
    advantage = proto_tom_results - vanilla_results
```

**No Training**: Gradients disabled during evaluation for both agent types

---

## 7. Metrics & Evaluation

### 7.1 Performance Metrics

**Primary Metrics**:
- **Average Reward**: Mean cumulative reward across test sessions
- **Cooperation Rate**: Proportion of cooperative actions taken by agent

**Secondary Metrics**:
- **Win Rate**: Proportion of games where agent outscores opponent
- **Mutual Cooperation Rate**: Proportion of rounds with mutual cooperation
- **Mutual Defection Rate**: Proportion of rounds with mutual defection

### 7.2 Generalization Metrics

**Generalization Error**:
```python
GeneralizationError = Performance_baseline - Performance_OOD_condition
```

**Breakdown by Axis**:
- **Opponent Generalization Gap**: Baseline - Same_Game_New_Opponents
- **Game Generalization Gap**: Baseline - New_Game_Same_Opponents  
- **Full OOD Gap**: Baseline - New_Game_New_Opponents

### 7.3 Theory of Mind Metrics (Proto-ToM Agents Only)

**Opponent Action Prediction Accuracy**:
```python
Action_Accuracy = Σ(predicted_action == actual_action) / num_predictions
```

**Opponent Reward Prediction Error**:
```python
Reward_Error = MSE(predicted_opponent_rewards, actual_opponent_rewards)
```

**ToM Development Speed**: 
- Epochs required for LOp_action loss to converge
- Epochs required for LOp_reward loss to converge

**ToM-Performance Correlation**: 
- Correlation between action prediction accuracy and agent reward
- Correlation between reward prediction accuracy and agent reward

### 7.4 Loss Component Analysis

**Vanilla Agent Loss Tracking**:
- LRL (policy gradient loss)
- Value loss
- Total loss trajectory

**Proto-ToM Agent Loss Tracking**:
- LRL contribution (normalized policy gradient loss)
- LOp_action contribution (normalized opponent action prediction loss)
- LOp_reward contribution (normalized opponent reward prediction loss)
- Value loss
- Total loss trajectory

**Adaptive Weighting** (if enabled):
- α trajectory over training (opponent action weight)
- β trajectory over training (opponent reward weight)
- Loss balancing effectiveness

**Agent Comparison**:
- Training speed: epochs to convergence (vanilla vs proto-ToM)
- Final loss values comparison
- Convergence stability metrics

---

## 8. Expected Outcomes

### 8.1 Agent Comparison: Proto-ToM vs Vanilla

**Expected Performance Hierarchy** (best to worst):
1. **Proto-ToM in-distribution**: Best performance with full training support
2. **Vanilla in-distribution**: Good performance without auxiliary task overhead
3. **Proto-ToM OOD conditions**: Maintained performance through opponent understanding
4. **Vanilla OOD conditions**: Largest performance drop without social inductive bias

**Expected Proto-ToM Advantage**:
- **Largest** in same-game/new-opponents condition (direct benefit of opponent modeling)
- **Moderate** in new-game/same-opponents condition (opponent understanding partially transfers)
- **Smallest** in new-game/new-opponents condition (both axes require generalization)

**Rationale**: Proto-ToM auxiliary tasks create representations that capture opponent behavioral patterns, which should transfer more robustly than pure reward-based learning

### 8.2 Generalization Patterns

**Expected Ordering** (easiest to hardest) for both agent types:
1. **Baseline** (in-distribution): Best performance
2. **Same-game/new-opponents**: Moderate difficulty
3. **New-game/same-opponents**: Higher difficulty  
4. **New-game/new-opponents**: Highest difficulty (compounding challenges)

**Expected Proto-ToM Mitigation**: Proto-ToM agents should show **smaller performance drops** between conditions, particularly for opponent shifts

**Rationale**: Opponent strategies are memoryless and probabilistic (simpler to model) compared to game structure changes (different payoff matrices and equilibria)

### 8.3 Game-Specific Effects

**Game Difficulty Ranking** (predicted learning speed):
1. **Prisoner's Dilemma**: Simplest (dominant strategy)
2. **Hawk-Dove**: Moderate (anti-coordination)
3. **Battle of Sexes**: Complex (coordination with conflict)
4. **Stag Hunt**: Complex (multiple equilibria, risk-dominance)

**Transfer Matrix Predictions**:
- **Within coordination games**: BoS ↔ Stag Hunt should transfer well
- **Within conflict games**: PD ↔ Hawk-Dove moderate transfer
- **Cross-category**: Coordination → Conflict games should be hardest

**Proto-ToM Effect**: Should reduce game transfer difficulty by learning generalizable incentive structures

### 8.4 Auxiliary Task Predictions

**Opponent Action Prediction**:
- Expected accuracy: 70-85% after training
- Higher accuracy on consistent opponents (p_d close to 0 or 1)
- Lower accuracy on stochastic opponents (p_d ≈ 0.5)

**Opponent Reward Prediction**:
- More difficult than action prediction initially
- Should improve faster for games with clear payoff structures (PD, HD)
- Critical for understanding game-specific incentives

**Synergy Hypothesis**: Both auxiliary tasks together should outperform either alone

---

## 9. Analysis Plan

### 9.1 Training Phase Analysis

**Training Convergence Comparison**:
- Learning curves per training condition (vanilla vs proto-ToM)
- Convergence speed: epochs to reach threshold
- Final loss values: total loss comparison between agent types
- Training stability: loss variance across epochs

**Proto-ToM Agent Auxiliary Task Development**:
- Opponent action prediction accuracy trajectory over training
- Opponent reward prediction error reduction over training  
- Loss component balance: relative magnitudes of LRL, LOp_action, LOp_reward
- Correlation between auxiliary task performance and main task rewards

**Training Cost Analysis**:
- Computational time per epoch (vanilla vs proto-ToM)
- Memory requirements comparison
- Convergence efficiency (performance per epoch)

### 9.2 Generalization Analysis

**Primary Comparisons**:

1. **Agent Type Main Effect**:
   - Two-way ANOVA: agent_type (vanilla/proto-ToM) × test_condition
   - Main effect of agent_type on generalization error
   - Post-hoc tests for each test condition separately

2. **Generalization Matrix Visualization**:
   - **Two 16×16 heatmaps**: One for vanilla, one for proto-ToM
   - Rows: Training conditions, Columns: Test conditions
   - Metrics: Reward, cooperation rate, (proto-ToM only: prediction accuracies)
   - **Difference heatmap**: Proto-ToM − Vanilla performance

3. **Condition-Specific Analysis**:
   - **Same-game/new-opponents**: Opponent generalization ability
   - **New-game/same-opponents**: Game structure generalization ability
   - **New-game/new-opponents**: Combined generalization challenge
   - Compare vanilla vs proto-ToM within each condition

**Statistical Tests**:
- Paired t-tests: vanilla vs proto-ToM performance per test condition
- Effect size calculations (Cohen's d) for proto-ToM advantage  
- Interaction effects: Does proto-ToM advantage vary by test condition type?

### 9.3 Theory of Mind Analysis (Proto-ToM Agents)

**ToM-Performance Relationship**:
- Scatter plots: Opponent action accuracy vs agent reward
- Scatter plots: Opponent reward prediction error vs agent reward
- Multiple regression: Predict agent performance from both auxiliary tasks
- Partial correlations: Controlling for in-distribution performance

**ToM Generalization**:
- How action prediction accuracy transfers across test conditions
- How reward prediction accuracy transfers across test conditions
- Compare ToM robustness: variance in auxiliary task performance across tests

**Auxiliary Task Contribution**:
- Regression models predicting generalization from auxiliary task metrics
- Compare standardized coefficients: Which auxiliary task is more predictive?
- Interaction analysis: Do auxiliary tasks work synergistically?

### 9.4 Game and Opponent Effects

**Training Distribution Analysis**:

1. **Game Type Effects**:
   - ANOVA: training_game × agent_type on average generalization
   - Which games promote best transfer?
   - Does proto-ToM advantage vary by training game?

2. **Opponent Range Effects**:
   - ANOVA: opponent_range × agent_type on generalization
   - Diversity hypothesis: Wide ranges better than narrow?
   - Proto-ToM sensitivity to opponent training diversity

3. **Game Transition Difficulty**:
   - Transition matrix: From_game × To_game performance
   - Identify easiest and hardest game transitions
   - Compare vanilla vs proto-ToM transition patterns

### 9.5 Statistical Summary

**Descriptive Statistics**:
- Mean, std, min, max for all primary metrics
- Separate summaries for vanilla and proto-ToM agents
- Training condition breakdowns
- Test condition breakdowns

**Inferential Statistics**:
- **Main hypotheses tests** (see Section 4)
- Mixed-effects models: training_condition × test_condition × agent_type
- Post-hoc pairwise comparisons with FDR correction
- Effect size measures: η², Cohen's d, Cliff's delta

**Dimensionality Reduction**:
- PCA on generalization performance vectors (16-dimensional: one per test condition)
- Separate analysis for vanilla and proto-ToM agents
- Compare principal component structure between agent types
- Cluster analysis: Do training conditions cluster differently for vanilla vs proto-ToM?

### 9.6 Reporting Structure

**Primary Outputs**:

1. **Generalization Matrices** (`figures/`):
   - `vanilla_generalization_matrix.png`: Vanilla agent heatmap
   - `proto_tom_generalization_matrix.png`: Proto-ToM agent heatmap
   - `proto_tom_advantage_matrix.png`: Difference heatmap

2. **Statistical Summary** (`tables/`):
   - `descriptive_statistics.csv`: Means, SDs by agent type and condition
   - `hypothesis_tests.csv`: Results for H1-H7 with p-values and effect sizes
   - `agent_comparison_summary.csv`: Direct vanilla vs proto-ToM comparisons

3. **Auxiliary Task Analysis** (`figures/proto_tom/`):
   - `auxiliary_task_development.png`: Training trajectories
   - `tom_performance_correlations.png`: Scatter plots and regression lines
   - `auxiliary_task_generalization.png`: ToM metric transfers

4. **Executive Summary** (`report/`):
   - `executive_summary.pdf`: Key findings, visualizations, interpretations
   - Focus on proto-ToM advantage magnitude and patterns
   - Implications for social AI and multi-agent learning

**Detailed Reports**:
- Per-training-condition breakdowns (vanilla and proto-ToM)
- Per-test-condition analyses
- Game-specific deep dives
- Opponent range sensitivity analyses
- ToM development trajectories
- Loss component analyses
- Game-specific deep dives

---

## 10. Implementation Guidelines

### 10.1 Code Organization Principles

**Design Philosophy**:
- **Modularity**: Each component (game, opponent, network, loss) is independently testable
- **Agent Separation**: Vanilla and proto-ToM agents share core architecture but differ in output heads and loss
- **Extensibility**: New games, opponents, or loss functions can be added without modifying core code
- **Reproducibility**: All experiments use fixed random seeds and save complete configurations

**Key Files** (see [.github/copilot-instructions.md](.github/copilot-instructions.md) for details):
- `src/cognitive_therapy_ai/network.py`: GameLSTM architecture (shared by both agent types)
- `src/cognitive_therapy_ai/tom_rl_loss.py`: Loss functions for both vanilla and proto-ToM
- `src/cognitive_therapy_ai/trainer.py`: Training and evaluation loops
- `main_experiment.py`: Experiment orchestration and CLI
- `config/default_config.json`: Default hyperparameters

### 10.2 Implementing the Two-Agent Structure

**Network Configuration**:
```python
# Vanilla Agent
vanilla_network = GameLSTM(
    input_size=5,
    hidden_size=128,
    num_layers=2,
    output_heads=['policy', 'value']  # No auxiliary heads
)

# Proto-ToM Agent
proto_tom_network = GameLSTM(
    input_size=5,
    hidden_size=128,
    num_layers=2,
    output_heads=['policy', 'value', 'opponent_action', 'opponent_reward']
)
```

**Training Both Agents**:
```python
for training_condition in all_training_conditions:
    # Train vanilla agent
    vanilla_trainer = GameTrainer(
        network=vanilla_network,
        loss_fn=VanillaRLLoss(),
        ...
    )
    vanilla_results = vanilla_trainer.train_on_game(...)
    
    # Train proto-ToM agent
    proto_tom_trainer = GameTrainer(
        network=proto_tom_network,
        loss_fn=ProtoToMLoss(alpha=1.0, beta=1.0),
        ...
    )
    proto_tom_results = proto_tom_trainer.train_on_game(...)
    
    # Save both agents
    save_agent(vanilla_network, 'vanilla', training_condition)
    save_agent(proto_tom_network, 'proto_tom', training_condition)
```

### 10.2 Configuration Management

**Hierarchy**:
1. **Default config**: `config/default_config.json`
2. **Command-line overrides**: Arguments passed to `main_experiment.py`
3. **Custom configs**: User-provided JSON files

**Config Structure**:
```json
{
  "network_config": {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.1
  },
  "training_config": {
    "num_games_per_partner": 100,
    "learning_rate": 0.001,
    "batch_size": 32,
    "max_epochs": 500,
    "convergence_threshold": 1e-6,
    "patience": 50
  },
  "experiment_config": {
    "opponent_defection_probs": [0.1, 0.3, 0.5, 0.7, 0.9],
    "random_seed": 42,
    "save_checkpoints": true
  },
  "games": {
    "game-name": {"parameter": value}
  }
}
```

### 10.3 Experiment Modes

**Mode 1: Basic (Recommended) - Two Agent Comparison**
```bash
# Train and test both agent types
python main_experiment.py --experiment-mode basic \
    --agent-types vanilla,proto-tom \
    --train-game prisoners-dilemma --train-opponents 0.1,0.3 \
    --test-game hawk-dove --test-opponents 0.7,0.9
```

**Mode 2: Generalization Matrix (SLURM Array Jobs) - Both Agents**
```bash
# Each task trains both vanilla and proto-ToM agents
python main_experiment.py --experiment-mode generalization-matrix \
    --agent-types vanilla,proto-tom \
    --task-id $SLURM_ARRAY_TASK_ID \
    --matrix-config config/generalization_matrix_config.json
```

**Mode 3: Single Agent Type Testing**
```bash
# Train only vanilla or only proto-ToM for ablation studies
python main_experiment.py --experiment-mode basic \
    --agent-types vanilla \
    --train-game prisoners-dilemma --train-opponents 0.1,0.3
```

### 10.4 Output Structure

**Standard Experiment Directory**:
```
experiments/mixed_motive_experiment_YYYYMMDD_HHMMSS/
├── vanilla_agent/                # Vanilla RL agent results
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   └── final_model.pth
│   ├── logs/
│   │   ├── training.log
│   │   └── detailed_testing_logs/
│   ├── plots/
│   │   ├── training_curves.png
│   │   ├── cooperation_rates.png
│   │   └── loss_components.png
│   └── results/
│       ├── generalization_results.pkl
│       └── generalization_report.json
│
├── proto_tom_agent/             # Proto-ToM agent results
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   └── final_model.pth
│   ├── logs/
│   │   ├── training.log
│   │   └── detailed_testing_logs/
│   ├── plots/
│   │   ├── training_curves.png
│   │   ├── cooperation_rates.png
│   │   ├── loss_components.png
│   │   └── auxiliary_task_analysis.png
│   └── results/
│       ├── generalization_results.pkl
│       └── generalization_report.json
│
├── comparison/                  # Cross-agent comparison
│   ├── proto_tom_advantage_matrix.png
│   ├── agent_comparison_summary.csv
│   └── statistical_tests.csv
│
└── experiment_config.json       # Complete configuration snapshot
```

### 10.5 Modification Protocol

**When Changing Loss Functions**:
1. Document in `docs/MODIFICATIONS.md` with before/after code
2. Update `CHANGELOG.md` with high-level description
3. Create unit tests in `tests/`
4. Run comparison experiments (old vs new)

**When Adding Games**:
1. Extend `MixedMotiveGame` in `src/cognitive_therapy_ai/games/`
2. Implement `get_payoff_matrix()` returning 2×2 numpy array
3. Register in `GameFactory`
4. Add config section to `config/default_config.json`
5. Add tests

**When Adding Opponents**:
1. Extend `OpponentStrategy` in `src/cognitive_therapy_ai/opponent.py`
2. Implement `choose_action(game_history, round_number)`
3. Register in `OpponentFactory`
4. Document behavior and parameters

### 10.6 Testing Requirements

**Unit Tests**: All new components
**Integration Tests**: Full training pipelines
**Regression Tests**: Compare against baseline results
**Validation Tests**: Sanity checks on metrics

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-22 | [Name] | Initial document creation |

---

## References

[Add relevant literature, theoretical papers, or related work]

---

**Document Status**: This is a living document. Update as experimental design evolves or new insights emerge.
