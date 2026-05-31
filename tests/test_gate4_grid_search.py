"""
Gate 4 Tests: Grid Search and Capacity Analysis

Verifies:
1. Single experiment runs successfully
2. Grid search executes across H values
3. Results are tracked correctly
4. Capacity analysis computes statistics
5. Knee detection identifies optimal H
6. Report generation produces valid markdown
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import yaml
import json
import tempfile
import shutil

from cognitive_therapy_ai.grid_search import (
    GridSearchRunner,
    GridSearchResult,
    CapacityAnalysis,
    detect_knee_point,
    create_game_from_config
)


def load_config():
    """Load base configuration."""
    config_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class TestSingleExperiment:
    """Test running single experiments."""
    
    def test_single_experiment_completes(self, tmp_path):
        """Test that a single experiment runs without errors."""
        config_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
        runner = GridSearchRunner(config_path, output_dir=tmp_path)
        
        result = runner.run_single_experiment(
            H=4,
            game_name='PD',
            p_coop=0.5,
            seed=42,
            input_condition='no_game',
            training_mode='bc',
            max_episodes=20,
            verbose=False
        )
        
        # Check result fields
        assert result.hidden_size == 4
        assert result.game_name == 'PD'
        assert result.p_coop == 0.5
        assert result.seed == 42
        assert result.input_condition == 'no_game'
        assert isinstance(result.final_return, float)
        assert isinstance(result.br_accuracy, float)
        assert 0.0 <= result.br_accuracy <= 1.0
        assert result.parameter_count > 0
    
    def test_experiment_parameter_count_matches_H(self, tmp_path):
        """Test that parameter count varies with H."""
        config_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
        runner = GridSearchRunner(config_path, output_dir=tmp_path)
        
        H_values = [2, 4, 8]
        param_counts = []
        
        for H in H_values:
            result = runner.run_single_experiment(
                H=H,
                game_name='PD',
                p_coop=0.5,
                seed=42,
                input_condition='no_game',
                training_mode='bc',
                max_episodes=10,
                verbose=False
            )
            param_counts.append(result.parameter_count)
        
        # Parameter count should increase with H
        for i in range(len(param_counts) - 1):
            assert param_counts[i] < param_counts[i+1], \
                f"Parameter count should increase with H: {param_counts}"
    
    def test_experiment_with_different_input_conditions(self, tmp_path):
        """Test experiments with no_game vs game_tag."""
        config_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
        runner = GridSearchRunner(config_path, output_dir=tmp_path)
        
        result_no_game = runner.run_single_experiment(
            H=4,
            game_name='PD',
            p_coop=0.5,
            seed=42,
            input_condition='no_game',
            training_mode='bc',
            max_episodes=10,
            verbose=False
        )
        
        result_game_tag = runner.run_single_experiment(
            H=4,
            game_name='PD',
            p_coop=0.5,
            seed=42,
            input_condition='game_tag',
            training_mode='bc',
            max_episodes=10,
            verbose=False
        )
        
        # Game_tag should have more parameters (larger input)
        assert result_game_tag.parameter_count > result_no_game.parameter_count


class TestGridSearch:
    """Test grid search execution."""
    
    def test_grid_search_runs_all_combinations(self, tmp_path):
        """Test that grid search runs all H × game × opponent × seed combinations."""
        config_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
        runner = GridSearchRunner(config_path, output_dir=tmp_path)
        
        games = ['PD']
        opponent_bins = [0.3, 0.7]
        H_grid = [2, 4]
        seeds = [1, 2]
        
        runner.run_grid_search(
            games=games,
            opponent_bins=opponent_bins,
            H_grid=H_grid,
            seeds=seeds,
            input_condition='no_game',
            training_mode='bc',
            max_episodes=10,
            verbose=False
        )
        
        # Should have 1*2*2*2 = 8 results
        expected_count = len(games) * len(opponent_bins) * len(H_grid) * len(seeds)
        assert len(runner.results) == expected_count
    
    def test_grid_search_saves_results(self, tmp_path):
        """Test that results are saved to JSON."""
        config_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
        runner = GridSearchRunner(config_path, output_dir=tmp_path)
        
        runner.run_grid_search(
            games=['PD'],
            opponent_bins=[0.5],
            H_grid=[4, 8],
            seeds=[1],
            input_condition='no_game',
            training_mode='bc',
            max_episodes=10,
            verbose=False
        )
        
        runner.save_results('test_results.json')
        
        # Check file exists
        results_path = tmp_path / 'test_results.json'
        assert results_path.exists()
        
        # Load and verify JSON
        with open(results_path, 'r') as f:
            loaded = json.load(f)
        
        assert len(loaded) == 2  # 2 H values × 1 game × 1 opponent × 1 seed
        assert all(isinstance(r, dict) for r in loaded)


class TestCapacityAnalysis:
    """Test capacity-performance analysis."""
    
    def test_analyze_capacity_computes_statistics(self, tmp_path):
        """Test that capacity analysis computes mean and std correctly."""
        config_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
        runner = GridSearchRunner(config_path, output_dir=tmp_path)
        
        # Run experiments with multiple seeds
        runner.run_grid_search(
            games=['PD'],
            opponent_bins=[0.5],
            H_grid=[4, 8],
            seeds=[1, 2, 3],
            input_condition='no_game',
            training_mode='bc',
            max_episodes=15,
            verbose=False
        )
        
        analysis = runner.analyze_capacity(
            game_name='PD',
            p_coop=0.5,
            input_condition='no_game',
            saturation_threshold=0.05
        )
        
        # Check analysis fields
        assert analysis.game_name == 'PD'
        assert analysis.p_coop == 0.5
        assert analysis.input_condition == 'no_game'
        assert len(analysis.H_values) == 2
        assert len(analysis.mean_returns) == 2
        assert len(analysis.std_returns) == 2
        assert len(analysis.mean_br_accuracies) == 2
        assert len(analysis.std_br_accuracies) == 2
    
    def test_capacity_analysis_selects_optimal_H(self, tmp_path):
        """Test that knee detection selects reasonable optimal H."""
        config_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
        runner = GridSearchRunner(config_path, output_dir=tmp_path)
        
        # Run with increasing H
        runner.run_grid_search(
            games=['PD'],
            opponent_bins=[0.5],
            H_grid=[2, 4, 8, 16],
            seeds=[1, 2],
            input_condition='no_game',
            training_mode='bc',
            max_episodes=20,
            verbose=False
        )
        
        analysis = runner.analyze_capacity(
            game_name='PD',
            p_coop=0.5,
            input_condition='no_game',
            saturation_threshold=0.05
        )
        
        # Optimal H should be in the grid
        assert analysis.optimal_H in [2, 4, 8, 16]
        assert analysis.optimal_H_param_count > 0
        assert len(analysis.justification) > 0


class TestKneeDetection:
    """Test knee detection algorithm."""
    
    def test_knee_detection_simple_case(self):
        """Test knee detection on simple monotonic curve."""
        # Simulate saturation curve: performance increases then plateaus
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([0.5, 0.8, 0.92, 0.95, 0.96])  # Saturates around 0.95
        
        knee_idx = detect_knee_point(x, y, threshold=0.05)
        
        # Should detect knee around index 2-3 (where y ≥ 0.91)
        assert knee_idx in [2, 3], f"Expected knee at 2-3, got {knee_idx}"
    
    def test_knee_detection_no_saturation(self):
        """Test knee detection when curve doesn't saturate."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Linear, no saturation
        
        knee_idx = detect_knee_point(x, y, threshold=0.05)
        
        # Should return last index when no saturation
        assert knee_idx == len(y) - 1


class TestReportGeneration:
    """Test report generation."""
    
    def test_report_generates_markdown(self, tmp_path):
        """Test that report generation creates valid markdown file."""
        config_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
        runner = GridSearchRunner(config_path, output_dir=tmp_path)
        
        # Run small grid search
        runner.run_grid_search(
            games=['PD'],
            opponent_bins=[0.5],
            H_grid=[4, 8],
            seeds=[1],
            input_condition='no_game',
            training_mode='bc',
            max_episodes=10,
            verbose=False
        )
        
        # Generate analysis
        analysis = runner.analyze_capacity(
            game_name='PD',
            p_coop=0.5,
            input_condition='no_game'
        )
        
        # Generate report
        runner.generate_report([analysis], filename='test_report.md')
        
        # Check file exists
        report_path = tmp_path / 'test_report.md'
        assert report_path.exists()
        
        # Check content
        content = report_path.read_text()
        assert '# Network Size Justification' in content
        assert 'Optimal H' in content
        assert 'PD' in content
        assert 'Capacity-Performance Curve' in content


class TestGameCreation:
    """Test game creation from config."""
    
    def test_create_all_games_from_config(self):
        """Test creating PD, SH, HD from config."""
        config = load_config()
        
        for game_name in ['PD', 'SH', 'HD']:
            game = create_game_from_config(config, game_name)
            assert game is not None
            
            # Check payoff matrix
            payoff = game.get_payoff_matrix()
            assert payoff.shape == (2, 2)


def test_gate4_integration(tmp_path):
    """
    Gate 4 integration test: Full grid search pipeline.
    
    Tests:
    1. Run mini grid search (2 H values, 1 game, 1 opponent, 2 seeds)
    2. Analyze capacity
    3. Generate report
    4. Verify outputs exist
    """
    print("\n[Gate 4 Integration Test]")
    
    config_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
    runner = GridSearchRunner(config_path, output_dir=tmp_path)
    
    # Mini grid search
    print("Running mini grid search...")
    runner.run_grid_search(
        games=['PD'],
        opponent_bins=[0.5],
        H_grid=[4, 8],
        seeds=[1, 2],
        input_condition='no_game',
        training_mode='bc',
        max_episodes=20,
        verbose=True
    )
    
    assert len(runner.results) == 4  # 2 H × 1 game × 1 opponent × 2 seeds
    
    # Save results
    print("\nSaving results...")
    runner.save_results('integration_results.json')
    assert (tmp_path / 'integration_results.json').exists()
    
    # Analyze capacity
    print("\nAnalyzing capacity...")
    analysis = runner.analyze_capacity(
        game_name='PD',
        p_coop=0.5,
        input_condition='no_game',
        saturation_threshold=0.05
    )
    
    print(f"Optimal H: {analysis.optimal_H}")
    print(f"Parameter count: {analysis.optimal_H_param_count}")
    assert analysis.optimal_H in [4, 8]
    
    # Generate report
    print("\nGenerating report...")
    runner.generate_report([analysis], filename='integration_report.md')
    assert (tmp_path / 'integration_report.md').exists()
    
    # Verify report content
    report_content = (tmp_path / 'integration_report.md').read_text()
    assert 'Network Size Justification' in report_content
    assert f'Optimal H**: {analysis.optimal_H}' in report_content
    
    print("\n✅ Gate 4 integration test passed!")


if __name__ == "__main__":
    # Run integration test
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_gate4_integration(Path(tmpdir))
    print("\n✅ All Gate 4 tests complete!")
