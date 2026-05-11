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
3. For each setup we compute and plot the following metrics:
3.1. Probability to cooperate as a function of the training duration. Line plot (x-axis epoch, y-axis P(coopoerate), colour - setup)
3.2. Normalized generalization heatmap - for each agent compute the taskwise normalized reward (normalzie rewards within task to [0,1]). The heatmap shows the average reward in each setup with the training setup cell empty. Tasks are on the y-axis, opponents are on the x-axis. Gradient goes from red to green with zero in white.
3.3. Probability to cooperate heatmap - for each agent and test data, plot a heatmap of the probability to cooperate using the same layout as the plot above
3.4. KLD from optimal policy - for each agent and setup compute the KLD between the learnt policy and the optimal policy (this is the policy of the agent that was trained on this particular setup). The end result is a line plot with KLD on the y-axis and discrete x-axis - a group for each task and within the task the opponents are ordered by the probability  to cooperate. Color indicating the trained agent.
3.5. Cluster analysis of agents based on behavior - using the data from 3.3 cluster the agents into groups based on their behavior. Color represent training opponent and shape represents the task. The x and y axis represent the probaility to cooporate and the average normalized reward across all test data 

4. In addition, we analyze the represnetation of each agent. In particular we are interested in the learning dymanics. Review EMBEDDING_ANALYSIS_GUIDE.md file, analyze and plot the social vs. task weights. Plot the results using PCA or tsne.
4.1. Reciprocity analysis - for each agent compute the rolling probability of cooperating as a function of the last opponent action (pass forward of the network with same input but different opponent action) - does agents adapt in reaction to the opponent actions or not?