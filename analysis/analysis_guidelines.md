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