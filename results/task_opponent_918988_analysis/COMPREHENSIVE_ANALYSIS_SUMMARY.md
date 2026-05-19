# Comprehensive Analysis Summary - Experiment 918988/918989
## Task-Opponent Setup with Reduced Network Architecture

**Date**: 2025-01-18  
**Experiment**: generalization_matrix_train_918988 / generalization_matrix_test_918989  
**Network**: 128 hidden units, 2 LSTM layers, 9-element input  
**Agent Type**: Vanilla RL (no ToM auxiliary tasks)  
**Seeds**: 5 (6431, 6441, 6451, 6461, 6471)  

---

## Experimental Design

### Task-Opponent Matrix (3 × 5 = 15 conditions)
- **Games**: Prisoner's Dilemma (PD), Hawk-Dove (HD), Stag-Hunt (SH)
- **Opponents**: p ∈ {0.1, 0.3, 0.5, 0.7, 0.9} (defection probability)
- **Training**: 75 models (15 conditions × 5 seeds)
- **Testing**: 72/75 models tested across all 15 conditions (models 72-74 missing)

### Data Coverage
- **Training Records**: 37,500 (75 models × 500 epochs)
- **Test Records**: 1,000 (72 models × 14 test conditions - no on-training testing)
- **All models hit max epoch limit** (500 epochs)

---

## Key Findings

### 1. Training Cooperation Patterns (3×5 Grid)

**Prisoner's Dilemma (PD):**
- **Near-zero cooperation across all opponent types**
- All models converged to pure defection strategy
- Stable from early epochs (~50-100)
- **No opponent-specific adaptation**

**Hawk-Dove (HD):**
- **Binary strategies** depending on opponent:
  - p=0.1, 0.3: ~0% cooperation (pure defection)
  - p=0.5: Variable cooperation (0-50%)
  - p=0.7, 0.9: ~100% cooperation (pure cooperation)
- **Rapid convergence** (< 100 epochs for most conditions)
- Clear opponent-dependent strategy selection

**Stag-Hunt (SH):**
- **High variance across seeds**:
  - p=0.5: ~50% cooperation with high variance
  - p=0.1, 0.3, 0.7, 0.9: Mixed outcomes (0% or 100%)
- **Unstable training dynamics**
- Some seeds show late-epoch fluctuations

---

### 2. Test Performance: Normalized Reward (3×3 Heatmap)

**Within-Game Performance (Diagonal):**
- **PD → PD**: 0.60-0.92 (strong performance, high diagonal values)
- **HD → HD**: 0.51-0.90 (good performance, some opponent mismatch penalties)
- **SH → SH**: 0.38-0.50 (moderate performance, high variance)

**Cross-Game Generalization:**
- **PD → HD**: 0.31-0.90 (variable, best when test opponent is low)
- **PD → SH**: ~0.50 (uniform, neutral outcome from defection)
- **HD → PD**: 0.28-0.92 (highly dependent on training opponent)
- **HD → SH**: 0.48-0.50 (neutral outcomes)
- **SH → PD**: 0.60-0.92 (good generalization due to cooperation)
- **SH → HD**: 0.50-0.90 (variable, cooperation works well in HD)

**Key Insight**: SH-trained agents generalize well to PD due to cooperative strategy, while PD-trained defectors struggle in coordination games.

---

### 3. Test Performance: Cooperation Heatmap (3×3 Grid)

**Prisoner's Dilemma:**
- **0% cooperation everywhere** (pure defection in training → pure defection in testing)
- No opponent adaptation in test phase
- Consistent across all test conditions

**Hawk-Dove:**
- **Binary outcomes**:
  - Agents trained on p=0.1-0.5: 0% test cooperation
  - Agents trained on p=0.7-0.9: 100% test cooperation
- Training strategy **fully determines** test behavior
- No on-the-fly adaptation

**Stag-Hunt:**
- **Mostly 100% cooperation** (p=0.1, 0.3, 0.7, 0.9 training)
- **48% cooperation** for p=0.5 trained agents (across all test conditions)
- Cooperative tendency persists across games

---

### 4. KL Divergence from Optimal Policy

**Definition**: KLD between test agent's cooperation rate and optimal policy (final training cooperation of agents trained on that test condition).

**Key Patterns**:

**PD-trained agents:**
- **Low KLD on PD** (near-optimal defection)
- **High KLD on SH** (defection is suboptimal)
- **Variable KLD on HD** (depends on opponent match)

**HD-trained agents:**
- **Highly opponent-specific**:
  - p=0.7, 0.9 trained: Low KLD on HD (correct cooperation)
  - p=0.1, 0.3 trained: Low KLD on PD (correct defection)
  - p=0.5 trained: High KLD on HD-0.5 (unstable optimal policy)
- **Poor cross-game transfer** (high KLD on PD and SH)

**SH-trained agents:**
- **Low KLD on SH** (cooperation is optimal)
- **High KLD on PD** (~23 max, cooperation is suboptimal)
- **Moderate KLD on HD** (cooperation sometimes works)

**Max KLD**: ~23 (SH-trained agents tested on PD - cooperation vs defection mismatch)

---

### 5. Cluster Analysis

**Agent Behavior Space** (Mean Cooperation × Mean Normalized Reward):

**Cluster 1: Zero Cooperation, Moderate-High Reward (x=0, y=0.5-0.55)**
- **Members**: All PD-trained (5 opponents), HD-trained p=0.1-0.3
- **Strategy**: Pure defection across all test conditions
- **Performance**: 0.50-0.55 normalized reward

**Cluster 2: Moderate Cooperation, Moderate Reward (x=0.48, y=0.47)**
- **Members**: SH-trained p=0.5 (single outlier)
- **Strategy**: ~48% cooperation rate
- **Performance**: Below-average reward (0.47)
- **Large error bars**: High variance across seeds

**Cluster 3: Full Cooperation, Moderate-High Reward (x=1.0, y=0.37-0.59)**
- **Members**: HD-trained p=0.7-0.9, SH-trained p=0.1/0.3/0.7/0.9
- **Strategy**: Pure cooperation across all test conditions
- **Performance**: Variable (0.37-0.59 depending on game)
  - **SH p=0.7, 0.9**: Lowest reward (0.37-0.39)
  - **HD p=0.7, 0.9**: Highest reward (0.42)
  - **SH p=0.1**: Highest reward in cluster (0.59)

**Key Insight**: Only **3 distinct behavioral clusters** despite 15 training conditions:
1. Always defect (PD + HD-low-opp)
2. Sometimes cooperate (SH-0.5)
3. Always cooperate (HD-high-opp + most SH)

---

### 6. Cross-Generalization Analysis

**Transfer Types**:
1. **Same game, different opponent** (Blue bars)
2. **Different game, same opponent** (Green bars)
3. **Different game, different opponent** (Red bars)

**HD-trained agents:**
- **p=0.1**: Same-game best (0.40), cross-game poor (0.70 green, 0.51 red)
- **p=0.3**: Same-game best (0.45), cross-game moderate (0.63 green, 0.53 red)
- **p=0.5**: Balanced performance (~0.50-0.55 across all types)
- **p=0.7**: Low same-game (0.46), moderate cross (0.25 green, 0.44 red)
- **p=0.9**: Good same-game (0.48), very poor diff-game/same-opp (0.08 green!)

**PD-trained agents:**
- **p=0.1**: Poor same-game (0.52), excellent cross-game (0.70 green, 0.70 red)
- **p=0.3**: Moderate same-game (0.56), excellent cross-game (0.60 green)
- **p=0.5**: Good same-game (0.60), moderate cross-game (~0.50)
- **p=0.7**: Best same-game (0.64), moderate cross-game (0.40 green)
- **p=0.9**: Excellent same-game (0.68), poor diff-game/same-opp (0.30 green)

**SH-trained agents:**
- **p=0.1**: Moderate same-game (0.40), excellent cross-game (0.56 green, 0.32 red)
- **p=0.3**: Poor same-game (0.45), moderate cross-game (~0.47)
- **p=0.5**: Balanced ~0.50 across all types
- **p=0.7**: Good same-game (0.50), poor cross-game (0.37 green)
- **p=0.9**: Balanced same-game (0.50), variable cross (0.24 green, 0.68 red!)
  - **Largest error bars**: High variance in cross-generalization

**Key Insight**: 
- **Opponent transfer harder than game transfer** for most conditions
- **PD agents generalize well** across games (defection is robust)
- **HD/SH high-opponent agents struggle** with opponent mismatch (green bars drop)

---

## Summary Statistics

### Training Metrics
- **Convergence**: 100% of models reached max epochs (500)
- **Average final loss**: 1.877
- **Cooperation variance**: High in SH (0.48 ± 0.04), low in PD (0.00)

### Test Metrics
- **Average normalized reward**: 0.506 ± 0.098
- **Cooperation rate range**: 0.00 to 1.00 (binary strategies)
- **KLD range**: 0.00 to 23.0 (max for cooperation-defection mismatch)

### Generalization Patterns
- **Within-game**: 0.38-0.92 normalized reward
- **Cross-game**: 0.24-0.90 normalized reward  
- **Best cross-game**: SH → PD (cooperation succeeds)
- **Worst cross-game**: HD p=0.9 → other games (cooperation fails)

---

## Comparison to Previous Task-Opponent Analysis

### Network Architecture Change
- **Previous**: 32 hidden units, 1 layer
- **Current**: 128 hidden units, 2 layers
- **Impact**: Increased capacity but **no qualitative change** in behavioral clusters

### Key Behavioral Consistency
- Same **3 behavioral clusters** (defect, sometimes-cooperate, cooperate)
- **PD still induces pure defection** regardless of network size
- **HD still shows binary strategies** based on opponent
- **SH still exhibits high variance**

### Potential Differences to Investigate
- Compare cluster positions (x, y coordinates)
- Compare KLD magnitudes
- Compare cross-generalization bar heights
- Compare training convergence speed

---

## Output Files

### Data Files (CSV)
- `task_opponent_training_cooperation.csv` - 37,500 training records
- `task_opponent_test_results.csv` - 1,000 test records
- `task_opponent_kld_from_optimal.csv` - 210 KLD computations
- `task_opponent_cluster_analysis.csv` - 15 aggregated conditions
- `task_opponent_cross_generalization.csv` - 15 conditions × 3 transfer types

### Plots (PNG)
1. `cooperation_vs_epoch_3x5.png` - Training dynamics
2. `normalized_reward_heatmap_3x3.png` - Test performance by game
3. `cooperation_heatmap_3x3.png` - Test cooperation by game
4. `metric_3_4_kld_from_optimal.png` - Policy divergence analysis
5. `metric_3_5_cluster_analysis.png` - Behavioral clustering
6. `metric_3_6a_cross_generalization_analysis.png` - Transfer analysis

**Output Location**: `results/task_opponent_918988_analysis/`

---

## Research Implications

### Network Capacity Does Not Improve Generalization
- Larger network (128 vs 32 units) still converges to **fixed strategies**
- No evidence of **on-the-fly adaptation** during testing
- Behavioral diversity remains **limited to 3 clusters**

### Game-Specific Strategy Learning
- **PD**: Universal defection (Nash equilibrium)
- **HD**: Opponent-dependent binary choice
- **SH**: Cooperative bias with high variance

### Generalization Asymmetry
- **Cooperation transfers better** than defection across games
- **Game structure matters more** than opponent probability for cross-transfer
- **Opponent mismatch is penalized** more than game mismatch

### Implications for ToM-RL Comparison
- **Baseline established** for vanilla RL with reduced network
- **Next step**: Compare against ToM-RL with same architecture
- **Hypothesis**: ToM auxiliary task should improve:
  1. Opponent adaptation (reduce binary strategies)
  2. Cross-generalization (green bars should increase)
  3. Cluster diversity (more than 3 clusters?)

---

**Analysis Script**: `analysis/run_task_opponent_918988_analysis.py`  
**Author**: AI Assistant  
**Date**: 2025-01-18
