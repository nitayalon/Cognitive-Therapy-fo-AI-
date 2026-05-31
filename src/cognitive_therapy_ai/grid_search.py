"""
Grid search over hidden sizes H for capacity-performance analysis.

Key features:
1. Sweep H ∈ {2,4,8,16,32} across input conditions and seeds
2. Track metrics: final return, BR accuracy, convergence speed, parameter count
3. Generate capacity-performance curves
4. Automated knee detection → select smallest H at saturation
5. Export results: JSON + markdown report

Mathematical basis for knee detection:
    - Fit power law: performance ~ log(capacity) or performance ~ capacity^α
    - Detect saturation: ∂²performance/∂capacity² → 0
    - Select minimal H where performance ≥ (max_performance - threshold)
"""

import torch
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import yaml
from datetime import datetime

from .representation_agent import RepresentationAgent
from .reinforce_trainer import REINFORCETrainer, SessionEnvironment, train_to_convergence
from .encoding import ObservationEncoder
from .games import MixedMotiveGame, PrisonersDilemma, StagHunt, HawkDove
from .opponent import ProbabilisticOpponent


@dataclass
class GridSearchResult:
    """Results for a single (H, game, opponent, seed) configuration."""
    hidden_size: int
    game_name: str
    p_coop: float
    seed: int
    input_condition: str
    
    # Metrics
    final_return: float
    mean_return: float
    br_accuracy: float
    episodes_to_convergence: int
    parameter_count: int
    
    # Training info
    training_mode: str
    converged: bool
    max_episodes: int


@dataclass
class CapacityAnalysis:
    """Analysis of capacity-performance relationship."""
    input_condition: str
    game_name: str
    p_coop: float
    
    # Results by H
    H_values: List[int]
    parameter_counts: List[int]
    mean_returns: List[float]
    std_returns: List[float]
    mean_br_accuracies: List[float]
    std_br_accuracies: List[float]
    
    # Knee detection
    optimal_H: int
    optimal_H_param_count: int
    saturation_threshold: float
    justification: str


def create_game_from_config(config: Dict, game_name: str) -> MixedMotiveGame:
    """Create game instance from config."""
    payoff = config['games'][game_name]['payoff']
    R, S = payoff[0][0], payoff[0][1]
    T, P = payoff[1][0], payoff[1][1]
    
    if game_name == 'PD':
        return PrisonersDilemma(T=T, R=R, P=P, S=S)
    elif game_name == 'SH':
        # StagHunt uses stag_payoff, hare_payoff, stag_failure
        # R = stag_payoff, T = stag_failure (shouldn't be used but included for config),
        # P = hare_payoff, S = stag_failure
        return StagHunt(stag_payoff=R, hare_payoff=P, stag_failure=S)
    elif game_name == 'HD':
        # HawkDove has resource_value and cost_of_conflict
        # We need to back-calculate from payoff matrix
        # T = resource_value, P = (resource_value - cost_of_conflict)/2
        resource_value = T
        cost_of_conflict = 2 * (T - P)
        return HawkDove(resource_value=resource_value, cost_of_conflict=cost_of_conflict)
    else:
        raise ValueError(f"Unknown game: {game_name}")


class GridSearchRunner:
    """
    Run grid search over hidden sizes.
    
    Sweeps:
    - H ∈ {2, 4, 8, 16, 32}
    - Games: PD, SH, HD
    - Opponents: [0.1, 0.3, 0.5, 0.7, 0.9]
    - Seeds: Multiple random seeds for robustness
    - Input conditions: no_game (specialist), game_tag (generalist)
    """
    
    def __init__(
        self,
        config_path: Path,
        output_dir: Path,
        device: Optional[torch.device] = None
    ):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device if device is not None else torch.device('cpu')
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results: List[GridSearchResult] = []
    
    def run_single_experiment(
        self,
        H: int,
        game_name: str,
        p_coop: float,
        seed: int,
        input_condition: str,
        training_mode: str = "rl",
        max_episodes: int = 500,
        verbose: bool = False
    ) -> GridSearchResult:
        """
        Run single experiment: train agent and evaluate.
        
        Args:
            H: Hidden size
            game_name: "PD", "SH", or "HD"
            p_coop: Opponent cooperation probability
            seed: Random seed
            input_condition: "no_game" or "game_tag"
            training_mode: "rl" or "bc"
            max_episodes: Maximum training episodes
            verbose: Print progress
        
        Returns:
            GridSearchResult with metrics
        """
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create game and opponent
        game = create_game_from_config(self.config, game_name)
        p_defect = 1.0 - p_coop
        opponent = ProbabilisticOpponent(defection_probability=p_defect)
        
        # Create encoder and environment
        encoder = ObservationEncoder(input_condition)
        T = self.config['train']['T']
        env = SessionEnvironment(
            game, opponent, encoder, T=T,
            game_name=game_name if input_condition == "game_tag" else None
        )
        
        # Create agent
        input_dim = encoder.get_input_dim()
        agent = RepresentationAgent(
            input_dim=input_dim,
            hidden_size=H,
            device=self.device
        )
        
        # Create optimizer and trainer
        lr = self.config['train']['lr']
        optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
        trainer = REINFORCETrainer(
            agent, optimizer,
            gamma=self.config['train']['gamma'],
            gae_lambda=self.config['train']['gae_lambda'],
            device=self.device
        )
        
        # Train
        if verbose:
            print(f"Training H={H}, {game_name}, p_coop={p_coop}, seed={seed}")
        
        history = train_to_convergence(
            trainer, env,
            max_episodes=max_episodes,
            convergence_window=self.config['train'].get('early_stop_patience', 50),
            convergence_threshold=self.config['train'].get('convergence_tol', 1e-6),
            mode=training_mode,
            verbose=verbose
        )
        
        # Evaluate final policy
        final_stats = trainer.evaluate_session(env, max_steps=T, deterministic=True)
        
        # Compute metrics
        episodes_trained = len(history['returns'])
        converged = episodes_trained < max_episodes
        mean_return = np.mean(history['returns'][-50:])  # Average last 50 episodes
        
        param_count = agent.count_parameters()['total_params']
        
        result = GridSearchResult(
            hidden_size=H,
            game_name=game_name,
            p_coop=p_coop,
            seed=seed,
            input_condition=input_condition,
            final_return=final_stats.total_return,
            mean_return=mean_return,
            br_accuracy=final_stats.br_accuracy,
            episodes_to_convergence=episodes_trained,
            parameter_count=param_count,
            training_mode=training_mode,
            converged=converged,
            max_episodes=max_episodes
        )
        
        self.results.append(result)
        return result
    
    def run_grid_search(
        self,
        games: List[str],
        opponent_bins: List[float],
        H_grid: List[int],
        seeds: List[int],
        input_condition: str,
        training_mode: str = "rl",
        max_episodes: int = 500,
        verbose: bool = True
    ):
        """
        Run full grid search.
        
        Args:
            games: List of game names
            opponent_bins: List of p_coop values
            H_grid: List of hidden sizes
            seeds: List of random seeds
            input_condition: "no_game" or "game_tag"
            training_mode: "rl" or "bc"
            max_episodes: Maximum episodes per run
            verbose: Print progress
        """
        total_runs = len(games) * len(opponent_bins) * len(H_grid) * len(seeds)
        run_count = 0
        
        for game_name in games:
            for p_coop in opponent_bins:
                for H in H_grid:
                    for seed in seeds:
                        run_count += 1
                        if verbose:
                            print(f"\n=== Run {run_count}/{total_runs} ===")
                        
                        self.run_single_experiment(
                            H=H,
                            game_name=game_name,
                            p_coop=p_coop,
                            seed=seed,
                            input_condition=input_condition,
                            training_mode=training_mode,
                            max_episodes=max_episodes,
                            verbose=verbose
                        )
        
        if verbose:
            print(f"\n✅ Grid search complete: {len(self.results)} experiments")
    
    def analyze_capacity(
        self,
        game_name: str,
        p_coop: float,
        input_condition: str,
        saturation_threshold: float = 0.05
    ) -> CapacityAnalysis:
        """
        Analyze capacity-performance relationship for specific (game, opponent).
        
        Args:
            game_name: Game to analyze
            p_coop: Opponent cooperation probability
            input_condition: Input encoding condition
            saturation_threshold: Threshold for detecting saturation (fraction of max)
        
        Returns:
            CapacityAnalysis with knee detection
        """
        # Filter results
        filtered = [
            r for r in self.results
            if r.game_name == game_name
            and abs(r.p_coop - p_coop) < 0.01
            and r.input_condition == input_condition
        ]
        
        if not filtered:
            raise ValueError(f"No results for {game_name}, p_coop={p_coop}, {input_condition}")
        
        # Group by H
        H_values = sorted(set(r.hidden_size for r in filtered))
        
        parameter_counts = []
        mean_returns = []
        std_returns = []
        mean_br_accuracies = []
        std_br_accuracies = []
        
        for H in H_values:
            H_results = [r for r in filtered if r.hidden_size == H]
            
            # Aggregate metrics across seeds
            returns = [r.final_return for r in H_results]
            br_accs = [r.br_accuracy for r in H_results]
            param_count = H_results[0].parameter_count
            
            parameter_counts.append(param_count)
            mean_returns.append(np.mean(returns))
            std_returns.append(np.std(returns))
            mean_br_accuracies.append(np.mean(br_accs))
            std_br_accuracies.append(np.std(br_accs))
        
        # Knee detection: find smallest H where BR accuracy ≥ (max - threshold)
        max_br_acc = max(mean_br_accuracies)
        saturation_value = max_br_acc - saturation_threshold
        
        optimal_H = None
        for i, (H, br_acc) in enumerate(zip(H_values, mean_br_accuracies)):
            if br_acc >= saturation_value:
                optimal_H = H
                optimal_H_param_count = parameter_counts[i]
                break
        
        if optimal_H is None:
            # If no saturation, use largest H
            optimal_H = H_values[-1]
            optimal_H_param_count = parameter_counts[-1]
        
        # Generate justification
        justification = (
            f"Selected H={optimal_H} ({optimal_H_param_count} params) for "
            f"{game_name} with p_coop={p_coop}.\n"
            f"BR accuracy: {mean_br_accuracies[H_values.index(optimal_H)]:.3f} "
            f"(max: {max_br_acc:.3f}, threshold: {saturation_value:.3f}).\n"
            f"This is the smallest network achieving near-maximal performance."
        )
        
        return CapacityAnalysis(
            input_condition=input_condition,
            game_name=game_name,
            p_coop=p_coop,
            H_values=H_values,
            parameter_counts=parameter_counts,
            mean_returns=mean_returns,
            std_returns=std_returns,
            mean_br_accuracies=mean_br_accuracies,
            std_br_accuracies=std_br_accuracies,
            optimal_H=optimal_H,
            optimal_H_param_count=optimal_H_param_count,
            saturation_threshold=saturation_threshold,
            justification=justification
        )
    
    def save_results(self, filename: str = "grid_search_results.json"):
        """Save all results to JSON."""
        output_path = self.output_dir / filename
        
        results_dict = [asdict(r) for r in self.results]
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def generate_report(
        self,
        analyses: List[CapacityAnalysis],
        filename: str = "network_size_justification.md"
    ):
        """
        Generate markdown report with capacity analysis and recommendations.
        
        Args:
            analyses: List of CapacityAnalysis results
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            f.write("# Network Size Justification\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write("This report presents capacity-performance analysis across games and opponents.\n")
            f.write("For each condition, we identify the smallest network achieving near-maximal performance.\n\n")
            
            f.write("## Capacity Analysis by Condition\n\n")
            
            for analysis in analyses:
                f.write(f"### {analysis.game_name} (p_coop={analysis.p_coop}) — {analysis.input_condition}\n\n")
                f.write(f"**Optimal H**: {analysis.optimal_H} ({analysis.optimal_H_param_count} parameters)\n\n")
                f.write(f"{analysis.justification}\n\n")
                
                f.write("**Capacity-Performance Curve**:\n\n")
                f.write("| H | Params | Return (mean±std) | BR Acc (mean±std) |\n")
                f.write("|---|--------|-------------------|-------------------|\n")
                
                for i, H in enumerate(analysis.H_values):
                    marker = "**" if H == analysis.optimal_H else ""
                    f.write(
                        f"| {marker}{H}{marker} | "
                        f"{analysis.parameter_counts[i]} | "
                        f"{analysis.mean_returns[i]:.2f}±{analysis.std_returns[i]:.2f} | "
                        f"{analysis.mean_br_accuracies[i]:.3f}±{analysis.std_br_accuracies[i]:.3f} |\n"
                    )
                
                f.write("\n")
            
            f.write("## Recommendations\n\n")
            
            # Aggregate recommendations
            optimal_H_values = [a.optimal_H for a in analyses]
            most_common_H = max(set(optimal_H_values), key=optimal_H_values.count)
            
            f.write(f"**Most common optimal H**: {most_common_H}\n\n")
            f.write(f"This value appeared in {optimal_H_values.count(most_common_H)}/{len(analyses)} conditions.\n\n")
            
            f.write("**Interpretation**:\n")
            f.write("- Small H values indicate that the task does not require large representational capacity\n")
            f.write("- Saturation at low H suggests FSM extraction will be tractable\n")
            f.write("- Consistent H across conditions validates the choice\n")
        
        print(f"Report saved to {output_path}")


def detect_knee_point(
    x_values: np.ndarray,
    y_values: np.ndarray,
    threshold: float = 0.05
) -> int:
    """
    Detect knee point in curve using threshold method.
    
    Args:
        x_values: X-axis values (e.g., parameter counts)
        y_values: Y-axis values (e.g., BR accuracies)
        threshold: Threshold below max for saturation
    
    Returns:
        Index of knee point
    """
    # Normalize y values
    y_max = np.max(y_values)
    saturation_value = y_max - threshold
    
    # Find first point exceeding threshold
    for i, y in enumerate(y_values):
        if y >= saturation_value:
            return i
    
    # If no saturation, return last point
    return len(y_values) - 1
