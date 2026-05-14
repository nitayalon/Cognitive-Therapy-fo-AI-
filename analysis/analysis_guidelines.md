# Analysis Guidelines

This document outlines the guidelines and best practices for conducting analyses in the Cognitive Therapy for AI project.

## Directory Structure

- **`analysis/`**: Contains all analysis scripts and notebooks
  - Analysis scripts should be organized by analysis type
  - Use descriptive filenames that indicate the analysis purpose
  
- **`Results/`**: Contains outputs from analyses
  - Plots, figures, and visualizations
  - Tables and summary statistics
  - Analysis reports and summaries
  
- **`experiments/`**: Contains only raw experimental data
  - Training checkpoints
  - Logs
  - Raw results (CSV, JSON, PKL files)
  - No analysis scripts or generated plots

## Analysis Workflow

1. **Data Collection**: Experiments run on cluster, data saved to `experiments/[experiment_name]/`
2. **Analysis Execution**: Run analysis scripts from `analysis/` directory
3. **Results Storage**: Save all outputs (plots, tables, reports) to `Results/` directory
   - `Results/task_opponent_setup/`: Outputs for generalization matrix (task-opponent) setup
   - `Results/task_setup/`: Outputs for whole population (task) setup
4. **Documentation**: Update relevant markdown files with findings

## Best Practices

- Keep analysis scripts modular and reusable
- Document analysis parameters and assumptions
- Use consistent naming conventions for output files
- Version control analysis scripts but not large result files
- Create summary reports for each major analysis

## Script Organization

- `analyze_*.py`: Individual analysis scripts for specific tasks
- `plot_*.py`: Visualization and plotting scripts
- `*.ipynb`: Jupyter notebooks for exploratory analysis
- `check_*.py`: Data validation and integrity checking scripts

## Section 1: Analysis
Section 1 of the paper includes a baseline analysis of maladaptive behavior. In this phase, networks are trained across multiple game-theoretic frames and various opponents with different pro-sociality tendencies. The end goal is to characterize maladaptive behavior in preparation for future therapy.

We simulate two setups (as of 01/05/2026):
1. **Task-opponent setup** (Generalization Matrix), in which each agent is trained on a specific game and a fixed opponent
   - Training: `experiments/generalization_matrix_train_913243/` (75 models: 3 games × 5 opponents × 5 seeds)
   - Testing: `experiments/generalization_matrix_test_913245/` (1,050 test results: 75 models × 14 test conditions)
   - Architecture: input_size=9 with separate embeddings (May 8, 2026)
   
2. **Task setup** (Whole Population), in which each agent is trained on a specific game and a wide range of opponents
   - Training: `experiments/whole_population_train_913310/` (15 models: 3 games × 5 seeds, trained on all 5 opponents)
   - Testing: `experiments/whole_population_test_912631/` (225 test results: 15 models × 15 test conditions)
   - Architecture: input_size=9 with separate embeddings (May 9, 2026)
   - Note: `whole_population_train_911034/` uses older input_size=6 architecture (May 6, 2026) - not used for current analysis

In each task, the trained agent is then tested on all game-opponent setups. This sets the *generalization* metric of each agent.

### Data File Structure

**Whole Population Training:**
- Location: `experiments/whole_population_train_913310/training/whole_population_task_X_*/`
- Training logs: `checkpoints/detailed_training_log.csv` (epoch-level metrics)
- Results: `results/training_task_X_results.pkl`, `results/training_task_X_metrics.json`

**Whole Population Testing:**
- Location: `experiments/whole_population_test_912631/testing/whole_population_task_X_*/`
- Test results: `results/eval_model_X_<game>_on_<game>_opp_<prob>.csv` (episode-level data)
- Metadata: `results/eval_model_X_task_X_results.pkl`, `results/eval_model_X_task_X_report.json`

**Generalization Matrix Training:**
- Location: `experiments/generalization_matrix_train_913243/training/condition_X_seed_X/generalization_matrix_task_X_*/`
- Training logs: `checkpoints/detailed_training_log.csv`
- Results: `results/*.pkl`

**Generalization Matrix Testing:**
- Location: `experiments/generalization_matrix_test_913245/testing/` (structure TBD - data being downloaded)

### Analysis
For each setup, we run the following analysis protocol:
1. Data verification - verify that the entire data is available 
2. For each setup compute the following metrics. The computation is performed in two steps:
    2.1. Data unification (ETL) - collect raw data, unify and export the data for the metric computation into a csv file (with name indicating the setup and metric)
    2.2. Analysis and plotting script - consumes the csv file, run the analysis and output plots. Plots are saved in the results directory, in the output directory with name describing setup and metric
3. All plots are produces using Seaborn library in publication ready condition. Use information axis and variable names

### Behavioral Metrics (Section 3)

3.1. **Probability to cooperate as a function of training duration**
   - Line plot (x-axis: epoch, y-axis: P(cooperate), color: setup)
   - Shows learning dynamics during training
   
3.2. **Normalized generalization heatmap**
   - Per-agent task-wise normalized reward (normalize within task to [0,1])
   - Heatmap: y-axis=tasks, x-axis=opponents
   - Color gradient: red to green (RdYlGn colormap)
   - Training setup cell empty (testing generalization only)
   
3.3. **Probability to cooperate heatmap**
   - Same layout as 3.2
   - Shows cooperation patterns across test conditions
   - RdYlGn colormap: red=defection, green=cooperation
   
3.4. **KLD from optimal policy**
   - Two comparison plots:
     - KLD from task-only optimal policies
     - KLD from task-opponent optimal policies
   - For task-opponent setup: 3 rows (one per training game), normalized y-axis
   - For task setup: All training games on same plot, color-coded
   - X-axis: Test conditions grouped by (game, opponent), sorted alphabetically
   - Shows policy divergence from optimal behavior
   
3.5. **Cluster analysis of agents based on behavior**
   - Scatter plot: x-axis=mean cooperation probability, y-axis=mean normalized reward
   - Color: Training opponent
   - Shape: Training game
   - Visual clustering (no computational clustering algorithm)
   - Legends positioned in upper right corner

### Embedding Analysis (Metric 4)

**Research Question:** Do agents integrate opponent behavior (social learning) or only learn task structure (Nash equilibrium)?

**Architecture Background:**
- Input: 9 elements [payoff_matrix(4), round_number(1), opponent_action(1), agent_action(1), agent_reward(1), opponent_reward(1)]
- Separate embeddings for each component (6 pathways):
  - Environmental: payoff_matrix_embed, round_number_embed
  - Social: opponent_action_embed, agent_action_embed, agent_reward_embed, opponent_reward_embed
- Each embedding: Linear(input_dim, embed_dim) → ReLU → LayerNorm
- embed_dim = hidden_size / 6 (e.g., 128 → 21 per pathway)

**Core Hypothesis:**
- Social learners: Large weights in social embeddings (enable reciprocity)
- Nash learners: Small weights in social embeddings (ignore opponent history)

#### Analysis Methods:

**Method 1: Weight Magnitude Analysis**
- Compute L2 norm of each embedding layer's weights
- Larger weights = more learned structure = embedding is used
- Metrics:
  - `env_payoff_magnitude`, `env_round_magnitude`, `env_total`
  - `soc_opp_action`, `soc_agent_action`, `soc_agent_reward`, `soc_opp_reward`, `soc_total`
  - `social_ratio = soc_total / (env_total + soc_total)` ∈ [0,1]
- Interpretation:
  - social_ratio < 0.3: Nash learner (ignores social inputs)
  - social_ratio > 0.5: Social learner (uses opponent history)

**Method 2: Activation Variance Analysis**
- Measure variance of embedding outputs across diverse test states
- High variance = embedding discriminates inputs
- Low variance = embedding produces constant output (unused)
- Computed for each embedding pathway separately

**Method 3: Ablation Analysis**
- Zero out each embedding, measure policy change via KL divergence
- Large KLD = embedding is critical for policy
- Small KLD = embedding is not used
- Computed on subset of test states (1000 samples) for efficiency

**Method 4: Embedding Visualization (t-SNE/PCA)**
- Extract embedding activations across test episodes
- Project to 2D using t-SNE or PCA
- Color by test opponent cooperation rate
- Interpretation:
  - Clear clustering = embedding learns social patterns
  - Random scatter = embedding doesn't discriminate

#### Implementation:

**Scripts:**
- `analysis/metric_4_embedding_analysis.py` - Core analysis functions
- `analysis/run_task_opponent_embedding_analysis.py` - Task-opponent setup
- `analysis/run_task_embedding_analysis.py` - Task setup

**Data Generation:**
Generate 5000 diverse test states covering:
- All 3 games (random payoff matrices)
- Random rounds (0-99, normalized to 0-1)
- Random actions (0=cooperate, 1=defect)
- Computed rewards based on payoff matrices

**Execution:**
```bash
# Task-opponent setup
python analysis/run_task_opponent_embedding_analysis.py

# Task setup
python analysis/run_task_embedding_analysis.py
```

#### Visualizations:

**Task-Opponent Setup:**
- 4.1.1: Stacked bar chart of weight magnitudes (3 rows, one per game)
- 4.1.2: Heatmap of social ratios (rows=games, columns=opponents)
- 4.2.1: Box plots of activation variance by embedding type (3 panels per game)
- 4.2.2: Scatter of social vs environmental variance (color=opponent, shape=game)
- 4.3.1: Heatmap of ablation KL divergences (rows=models, columns=embeddings)
- 4.3.2: Average importance by condition (3×5 grid, one per game-opponent)

**Task Setup:**
- 4.1.1: Bar chart comparing env vs social weights across games
- 4.1.2: Box plot of social ratio distribution by game
- 4.3: Ablation KLD comparison (3 panels, one per game)

#### Output Structure:
```
Results/
├── task_opponent_setup/
│   └── embedding_analysis/
│       ├── unified_data/
│       │   └── embedding_analysis_results.csv
│       └── plots/
│           ├── metric_4.1.1_weight_magnitude_bars.png
│           ├── metric_4.1.2_social_ratio_heatmap.png
│           ├── metric_4.2.1_variance_boxplots.png
│           ├── metric_4.2.2_social_vs_env_variance_scatter.png
│           ├── metric_4.3.1_ablation_heatmap.png
│           └── metric_4.3.2_importance_by_condition.png
└── task_setup/
    └── embedding_analysis/
        ├── unified_data/
        │   └── embedding_analysis_results.csv
        └── plots/
            ├── metric_4.1.1_weight_magnitude_comparison.png
            ├── metric_4.1.2_social_ratio_by_game.png
            └── metric_4.3_ablation_comparison.png
```

#### Expected Findings:

**Task-Opponent Setup:**
- High social ratio for extreme opponents (0.1, 0.9) - need to adapt to specific behavior
- Low social ratio for PD-trained models - Nash dominates (always defect)
- Medium social ratio for HD/SH with moderate opponents

**Task Setup:**
- Lower social ratios overall - no specific opponent to adapt to
- Higher environmental weights - must learn general game structure
- PD may show lowest social usage (Nash most attractive)

#### Reproducibility Notes:

1. **Device:** Analysis runs on CUDA if available, else CPU
2. **Model Loading:** Uses `best_model.pth` from checkpoint directory
3. **Test States:** Generated randomly but with fixed seed (42) for reproducibility in visualizations
4. **Error Handling:** Skips models with missing checkpoints, logs warnings
5. **Memory:** Ablation uses subset of 1000 states to manage memory on GPU

#### Key Differentiators:

- **Metrics 3.1-3.5:** What agents do (observable behavior)
- **Metric 4:** How agents decide (internal representations)

This provides mechanistic insight into whether observed behaviors result from social reasoning or task-only optimization (Nash equilibrium).

---

### Reciprocity-Representation Coupling Analysis (Metric 5)

**Research Question:** How do task complexity, opponent complexity, learned representations, and behavioral reciprocity relate to each other?

**Core Hypotheses:**
1. **Task Complexity → Representation Complexity**: More complex games (e.g., Stag-Hunt coordination) require richer internal representations
2. **Opponent Complexity → Social Learning**: Adaptive opponents require reciprocity mechanisms, reflected in social embedding usage
3. **Representation-Behavior Coupling**: Similar behaviors can emerge from different representations (many-to-one mapping)
4. **Generalization from Coupling**: Agents with similar representation structure show similar generalization patterns

#### 5.1 Complexity Metrics

**Task Complexity Metrics:**
- `nash_equilibrium_count`: Number of pure strategy Nash equilibria (1=simple, >1=complex)
- `payoff_variance`: Variance in payoff matrix values (coordination games have higher variance)
- `social_dilemma_strength`: T - R for PD-like games, or max(T,R) - min(S,P) generalized
- `coordination_bonus`: R - max(S, T) for coordination games (negative for PD, positive for SH)
- `risk_dominance`: Measure of risk in coordination: (R-S) vs (T-P)

Game-specific values:
- **Prisoner's Dilemma**: Simple (1 NE), strong dilemma, no coordination
- **Hawk-Dove**: Medium complexity (1 NE in mixed strategies), anti-coordination
- **Stag-Hunt**: Complex (2 NE), weak dilemma, strong coordination bonus

**Opponent Complexity Metrics:**
- `stationarity`: 1.0 for probabilistic opponents (fixed strategy)
- `predictability`: 1 - entropy of opponent action distribution
  - High for extreme opponents (0.1, 0.9): predictable
  - Low for moderate opponents (0.5): unpredictable
- `adaptation_requirement`: How much agent must adapt behavior across rounds
  - Computed as variance in optimal response over opponent history
- `cooperation_rate`: Mean opponent cooperation (0.1 → 0.9, 0.9 → 0.1)

Opponent-specific values:
- **0.1 (Very Cooperative)**: High predictability, high cooperation
- **0.5 (Neutral)**: Low predictability, medium cooperation
- **0.9 (Very Defective)**: High predictability, low cooperation

**Behavioral Complexity Metrics:**
- `reciprocity_strength`: P(coop|opp_coop_t-1) - P(coop|opp_defect_t-1) ∈ [-1, 1]
  - Positive: reciprocal (tit-for-tat like)
  - Near zero: non-reciprocal (stationary strategy)
  - Negative: anti-reciprocal (contrarian)
- `policy_entropy`: H(π) = -Σ π(a) log π(a), measures strategy randomness
- `behavioral_variability`: Standard deviation of cooperation rate across games
- `temporal_consistency`: Autocorrelation in action sequences (0=random, 1=deterministic)
- `conditional_strategy_complexity`: Number of distinct P(coop|context) patterns

**Representational Complexity Metrics:**
- `social_ratio`: soc_total / (env_total + soc_total) from embedding weights
- `weight_l2_norm`: Total magnitude of all network weights
- `effective_dimensionality`: Intrinsic dimensionality of hidden state manifold
  - Computed via PCA: number of components explaining 95% variance
- `activation_sparsity`: Fraction of near-zero activations (L1/L2 ratio)
- `representational_similarity`: CKA score comparing representations across conditions
- `embedding_specialization`: Variance in weight magnitudes across embedding pathways
  - High: some embeddings strongly used, others ignored
  - Low: all embeddings used roughly equally

#### 5.2 Analysis Components

**Component 1: Task Complexity → Representation Structure**

*Hypothesis*: Complex games (SH) require higher-dimensional representations than simple games (PD)

Metrics:
- Effective dimensionality by training game
- Embedding specialization by training game
- Weight L2 norm by training game

Visualizations:
- Bar chart: effective_dim vs training_game (3 bars: PD, HD, SH)
- Heatmap: embedding_pathway_magnitude × training_game (6 pathways × 3 games)
- Scatter: payoff_variance (x) vs effective_dim (y), color by game

Expected Pattern:
- PD: Low dimensionality (simple dominant strategy), high specialization (ignore social)
- HD: Medium dimensionality (mixed strategy), medium specialization
- SH: High dimensionality (coordination), low specialization (all inputs relevant)

**Component 2: Opponent Complexity → Social Learning**

*Hypothesis*: Extreme opponents (predictable) require less social learning than moderate opponents (unpredictable)

Metrics:
- Social ratio by opponent defection probability
- Reciprocity strength by opponent defection probability
- Social embedding ablation importance by opponent

Visualizations:
- Line plot: opponent_defect_prob (x) vs social_ratio (y), separate line per game
- Line plot: opponent_defect_prob (x) vs reciprocity_strength (y), separate line per game
- Heatmap: training_condition × social_embedding_importance (15 conditions × 4 social embeddings)

Expected Pattern:
- Extreme opponents (0.1, 0.9): Low social ratio (predictable → no need to track)
- Moderate opponents (0.5): High social ratio (unpredictable → must track history)
- Reciprocity strength follows inverted-U: highest for moderate opponents

**Component 3: Representation-Behavior Coupling**

*Hypothesis*: Similar behaviors can emerge from different representations (degeneracy)

Analysis:
1. Cluster agents by behavior: (mean_coop, reciprocity_strength)
2. For each behavioral cluster, measure representation diversity:
   - Within-cluster weight distance (should be high if degenerate)
   - Between-cluster weight distance (for comparison)
3. Test: within_cluster_distance / between_cluster_distance
   - Ratio > 0.5: High degeneracy (same behavior, different representations)
   - Ratio < 0.3: Low degeneracy (same behavior, same representation)

Metrics:
- Behavioral clusters via K-means (k=5) on (mean_coop, reciprocity_strength)
- Weight-space distance: L2 norm of flattened parameter vectors
- Representational similarity: CKA on hidden state activations

Visualizations:
- Scatter: mean_coop (x) vs reciprocity_strength (y), color=cluster_id, marker=game
- Box plot: weight_distance, grouped by (within_cluster vs between_cluster)
- Heatmap: CKA similarity matrix (75×75 for task-opponent, 15×15 for task)

Expected Pattern:
- High within-cluster diversity for defector cluster (all achieve via different paths)
- Lower within-cluster diversity for reciprocator cluster (similar mechanisms)

**Component 4: Generalization from Representation Structure**

*Hypothesis*: Agents with similar representation structure show similar generalization patterns

Analysis:
1. Compute representation similarity matrix (CKA on hidden states)
2. Compute generalization similarity matrix (correlation of test performance vectors)
3. Test correlation: representational_similarity ~ generalization_similarity

Metrics:
- Representation similarity: CKA(model_i, model_j) ∈ [0,1]
- Generalization similarity: Pearson correlation of normalized test rewards
  - Each model has 14-dim vector (task-opponent) or 15-dim (task setup)
- Mantel test: correlation between similarity matrices

Visualizations:
- Scatter: representation_similarity (x) vs generalization_similarity (y)
  - Each point = model pair (i,j)
  - Color by: same_training_game (yes/no)
- Joint heatmap: Upper triangle = CKA, lower triangle = generalization correlation
- Regression plot with confidence bands

Expected Pattern:
- Positive correlation: similar representations → similar generalization
- Stronger correlation within same game (shared task structure)
- Weaker correlation across games (different task demands)

**Component 5: Integrated Complexity Analysis**

*Hypothesis*: Complexity metrics form a coherent structure predicting agent behavior

Analysis:
1. Compute all 4 complexity domains for each agent:
   - Task: nash_equilibria, payoff_variance, coordination_bonus
   - Opponent: predictability, cooperation_rate
   - Behavior: reciprocity_strength, policy_entropy
   - Representation: social_ratio, effective_dim, weight_l2_norm
2. Correlation analysis across domains
3. Dimensionality reduction (PCA) to find latent complexity factors
4. Regression: behavior ~ task_complexity + opponent_complexity + representation_complexity

Visualizations:
- Correlation matrix heatmap (all complexity metrics)
- PCA biplot: PC1 vs PC2, color=training_game, marker=opponent
- Feature importance: regression coefficients for predicting reciprocity_strength
- Path diagram: task_complexity → representation_complexity → behavioral_complexity

Expected Pattern:
- Task complexity correlates with representation complexity (mediation)
- Opponent complexity directly affects behavioral complexity
- Representation complexity mediates task → behavior relationship

#### 5.3 Implementation Plan

**Script Structure:**
```
analysis/
├── complexity_metrics.py              # Compute all 4 complexity domains
├── reciprocity_representation_coupling.py  # Main analysis script
└── run_complexity_analysis.py         # Entry point for both setups
```

**Key Functions:**

```python
# complexity_metrics.py
def compute_task_complexity(game_name: str, payoff_matrix: np.ndarray) -> Dict[str, float]
def compute_opponent_complexity(opponent_defect_prob: float, action_history: np.ndarray) -> Dict[str, float]
def compute_behavioral_complexity(test_data: pd.DataFrame) -> Dict[str, float]
def compute_representational_complexity(model: GameLSTM, test_states: torch.Tensor) -> Dict[str, float]

# reciprocity_representation_coupling.py
def analyze_task_representation_link(complexity_df: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]
def analyze_opponent_social_learning_link(complexity_df: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]
def analyze_representation_behavior_coupling(complexity_df: pd.DataFrame, models: Dict) -> Tuple[pd.DataFrame, plt.Figure]
def analyze_generalization_from_representation(complexity_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]
def integrated_complexity_analysis(complexity_df: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]
```

**Execution:**
```bash
# Task-opponent setup (75 models → 15 aggregated)
python analysis/run_complexity_analysis.py --setup task-opponent

# Task setup (15 models → 3 aggregated)
python analysis/run_complexity_analysis.py --setup task

# Both setups
python analysis/run_complexity_analysis.py --setup all
```

#### 5.4 Output Structure

```
Results/
├── task_opponent_setup/
│   └── complexity_analysis/
│       ├── unified_data/
│       │   ├── complexity_metrics.csv
│       │   ├── representation_behavior_coupling.csv
│       │   ├── generalization_similarity.csv
│       │   └── integrated_complexity.csv
│       └── plots/
│           ├── metric_5.1_task_representation_link.png
│           ├── metric_5.2_opponent_social_learning.png
│           ├── metric_5.3_representation_behavior_coupling.png
│           ├── metric_5.4_generalization_similarity.png
│           ├── metric_5.5_integrated_complexity.png
│           └── complexity_correlation_matrix.png
└── task_setup/
    └── complexity_analysis/
        └── [same structure]
```

#### 5.5 Statistical Tests

**Hypothesis Tests:**
1. Task complexity → Representation complexity:
   - One-way ANOVA: effective_dim ~ training_game
   - Post-hoc: Tukey HSD for pairwise comparisons

2. Opponent complexity → Social learning:
   - Spearman correlation: opponent_predictability ~ social_ratio
   - Quadratic regression: reciprocity_strength ~ opponent_defect_prob + opponent_defect_prob²

3. Representation-behavior coupling:
   - Permutation test: within_cluster_diversity vs between_cluster_diversity
   - Effect size: Cohen's d

4. Generalization from representation:
   - Mantel test: CKA_matrix ~ generalization_correlation_matrix
   - Linear regression: generalization_similarity ~ representation_similarity

5. Integrated complexity:
   - Mediation analysis: task_complexity → representation_complexity → behavioral_complexity
   - Structural equation modeling (SEM) if enough power

**Correction for Multiple Comparisons:**
- Bonferroni correction within each component (5 components)
- Report both corrected and uncorrected p-values
- Focus on effect sizes over p-values

#### 5.6 Expected Findings

**Task-Opponent Setup (15 conditions):**
- PD agents: High weight norms, low social ratio, low reciprocity (Nash dominates)
- SH agents: Lower weight norms, higher social ratio, moderate reciprocity (coordination requires tracking)
- HD agents: Medium on all metrics (anti-coordination)
- Extreme opponents: Lower social ratios regardless of game (predictable)
- Moderate opponents: Higher social ratios, higher reciprocity (must adapt)

**Task Setup (3 conditions):**
- Lower social ratios overall (no specific opponent to track)
- Higher representational dimensionality (must handle multiple opponent types)
- More degenerate representation-behavior mapping (one representation for multiple behaviors)
- Stronger generalization-representation coupling (general representations transfer better)

**Cross-Setup Comparison:**
- Task-opponent: Specialist representations (high social ratio for specific opponents)
- Task: Generalist representations (lower social ratio, higher dimensionality)
- Trade-off: specialization (task-opponent) vs generalization (task)

#### 5.7 Integration with Existing Metrics

**How Metric 5 Extends Previous Analyses:**

- **Metrics 3.1-3.5 (Behavioral)**: Provided WHAT agents do
- **Metric 4 (Embeddings)**: Provided HOW agents encode inputs
- **Metric 5 (Coupling)**: Provides WHY certain representations emerge and HOW they relate to behavior

**Cross-metric Insights:**
- Metric 3.4 (KLD from optimal) + Metric 5.1 → Do complex representations lead to better optimality?
- Metric 3.5 (Clustering) + Metric 5.3 → Are behavioral clusters homogeneous in representation?
- Metric 4.3 (Ablation) + Metric 5.2 → Does ablation importance predict reciprocity strength?
- Metric 3.6b (Cross-task ratio) + Metric 5.4 → Does representation similarity predict generalization?

**Unified Framework:**
```
Task Complexity → Representation Structure → Behavioral Patterns → Generalization
      ↑                    ↑                        ↑                    ↓
Opponent Complexity → Social Learning → Reciprocity → Test Performance
```

This positions Metric 5 as the integrative analysis that connects all previous findings into a coherent mechanistic story.