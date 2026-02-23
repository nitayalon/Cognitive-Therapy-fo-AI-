# Generalization Matrix Analysis Plan

## Overview
This document outlines the comprehensive analyses to be conducted on the generalization matrix experiment results located in `experiments/generalization_matrix_834222/`.

**Experiment Structure:**
- 16 tasks (task_0 through task_15)
- Each task represents a unique training configuration (game type × opponent type)
- Each trained agent is tested for generalization across all game-opponent combinations

---

## 1. Training Performance Analysis

### 1.1 Learning Curves by Task
- **Metric**: Cooperation rate, average reward, loss components over epochs
- **Visualization**: Line plots per task showing convergence patterns
- **Output**: `training_curves_by_task.png`

### 1.2 Cross-Task Comparison
- **Metric**: Final training performance across all 16 tasks
- **Visualization**: Bar chart or heatmap comparing final metrics
- **Output**: `training_performance_comparison.png`

### 1.3 Convergence Analysis
- **Metric**: Epochs to convergence, final loss values
- **Statistical Test**: ANOVA across task types
- **Output**: `convergence_statistics.csv`

---

## 2. Generalization Performance Analysis

### 2.1 Generalization Matrix Heatmap
- **Rows**: Training conditions (16 tasks)
- **Columns**: Test conditions (all game-opponent combinations)
- **Metric**: Cooperation rate, reward, opponent prediction accuracy
- **Output**: `generalization_matrix_cooperation.png`, `generalization_matrix_reward.png`

### 2.2 Within-Game vs Cross-Game Generalization
- **Comparison**: Performance on same game (different opponents) vs different games
- **Visualization**: Box plots or violin plots
- **Statistical Test**: Paired t-test
- **Output**: `within_vs_cross_game_generalization.png`

### 2.3 Opponent Generalization Patterns
- **Analysis**: How well agents generalize to opponents with different cooperation levels
- **Metric**: Performance gradient across opponent cooperation rates
- **Output**: `opponent_generalization_patterns.png`

---

## 3. Theory of Mind (ToM) Analysis

### 3.1 Opponent Prediction Accuracy
- **Metric**: Opponent cooperation prediction accuracy during training and testing
- **Visualization**: Heatmap showing prediction accuracy across all conditions
- **Output**: `tom_prediction_accuracy_matrix.png`

### 3.2 ToM Development Over Training
- **Metric**: Opponent prediction loss trajectory
- **Analysis**: Compare ToM development speed across different training conditions
- **Output**: `tom_development_curves.png`

### 3.3 ToM-Performance Correlation
- **Analysis**: Correlation between opponent prediction accuracy and cooperation/reward
- **Statistical Test**: Pearson/Spearman correlation
- **Output**: `tom_performance_correlation.png`, `correlation_statistics.csv`

---

## 4. Policy Adaptation Analysis

### 4.1 Action Distribution Shifts
- **Metric**: Cooperation probability distribution across different test conditions
- **Visualization**: Density plots or histograms
- **Output**: `action_distribution_shifts.png`

### 4.2 Strategy Profiles
- **Analysis**: Identify distinct behavioral strategies learned by different agents
- **Method**: Clustering analysis on cooperation patterns
- **Output**: `strategy_clusters.png`, `strategy_profiles.csv`

### 4.3 Behavioral Flexibility
- **Metric**: Variance in cooperation rates across test conditions
- **Interpretation**: High variance = more adaptive to context
- **Output**: `behavioral_flexibility_scores.csv`

---

## 5. Game-Specific Analyses

### 5.1 Performance by Game Type
- **Comparison**: Prisoner's Dilemma, Stag Hunt, Battle of Sexes, Hawk-Dove
- **Metric**: Training success rate, generalization performance
- **Output**: `performance_by_game.png`

### 5.2 Game Transition Analysis
- **Analysis**: Which game transitions are easiest/hardest for generalization
- **Metric**: Performance drop when switching games
- **Output**: `game_transition_difficulty_matrix.png`

### 5.3 Payoff Structure Sensitivity
- **Analysis**: How payoff matrix differences affect generalization
- **Output**: `payoff_sensitivity_analysis.png`

---

## 6. Multi-Task Learning Analysis

### 6.1 Transfer Learning Effects
- **Analysis**: Do agents trained on certain games/opponents transfer better?
- **Metric**: Average generalization performance per training condition
- **Output**: `transfer_learning_rankings.csv`

### 6.2 Catastrophic Forgetting Assessment
- **Analysis**: Performance on training condition when tested later
- **Metric**: Training performance vs post-training test performance on same condition
- **Output**: `catastrophic_forgetting_analysis.png`

---

## 7. Statistical Summary

### 7.1 Descriptive Statistics
- **Metrics**: Mean, std, min, max for all key metrics across all conditions
- **Output**: `descriptive_statistics.csv`

### 7.2 Inferential Statistics
- **Tests**: 
  - ANOVA for comparing task groups
  - Post-hoc pairwise comparisons
  - Effect size calculations (Cohen's d, eta-squared)
- **Output**: `inferential_statistics.csv`

### 7.3 Dimensionality Reduction
- **Method**: PCA or t-SNE on performance vectors
- **Purpose**: Visualize relationships between training conditions
- **Output**: `performance_space_visualization.png`

---

## 8. Loss Component Analysis (ToM-RL Specific)

### 8.1 RL Loss vs ToM Loss Trade-off
- **Metric**: Normalized L_RL and L_Op over training
- **Analysis**: Balance between policy optimization and opponent modeling
- **Output**: `loss_component_tradeoff.png`

### 8.2 Adaptive Loss Weighting (if used)
- **Analysis**: How α parameter evolved during training
- **Output**: `alpha_evolution.png`

---

## 9. Consolidated Report

### 9.1 Executive Summary
- **Content**: Key findings, main conclusions, surprising results
- **Format**: PDF report with embedded figures
- **Output**: `generalization_matrix_executive_summary.pdf`

### 9.2 Complete Data Export
- **Format**: Consolidated CSV with all metrics
- **Output**: `complete_analysis_results.csv`

---

## Output Directory Structure

```
experiments/results/
├── figures/
│   ├── training/
│   ├── generalization/
│   ├── tom_analysis/
│   ├── policy_adaptation/
│   └── statistical/
├── tables/
│   ├── descriptive_stats/
│   ├── inferential_stats/
│   └── rankings/
├── data/
│   └── processed_metrics.csv
└── report/
    └── executive_summary.md
```

---

## Analysis Execution Notes

- **Data Loading**: Read from each task's `results/` directory
- **Checkpoint Analysis**: Use checkpoints to analyze intermediate training states
- **Log Parsing**: Extract metrics from training logs
- **Statistical Power**: Ensure sufficient data for statistical tests
- **Reproducibility**: Set random seeds for clustering/dimensionality reduction

---

## Research Questions to Address

1. Does training on specific opponent types improve generalization to similar opponents?
2. Which games serve as better "foundation" for multi-game generalization?
3. Is ToM prediction accuracy necessary for successful cooperation?
4. Do agents develop game-specific or opponent-specific strategies?
5. What is the relationship between training diversity and generalization capability?

---

## Next Steps

1. Review and edit this analysis plan
2. Prioritize analyses based on research questions
3. Execute analysis script
4. Generate all figures and tables
5. Write up findings
