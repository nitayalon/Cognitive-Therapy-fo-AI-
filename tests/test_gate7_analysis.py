"""
Tests for Gate 7: Analysis Pipeline

Comprehensive testing of:
- AnalysisPipeline orchestration
- Plotting utilities
- Table generation
- Report writing
- Full integration pipeline
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json

from cognitive_therapy_ai.analysis import (
    ExperimentConfig,
    ExperimentResult,
    AnalysisPipeline
)
from cognitive_therapy_ai.representation_agent import RepresentationAgent
from cognitive_therapy_ai.fsm_extraction import FSM
from cognitive_therapy_ai.attribution import FeatureImportance


class TestExperimentConfig:
    """Test ExperimentConfig data structure."""
    
    def test_create_config(self):
        """Test creating experiment configuration."""
        config = ExperimentConfig(
            game_name='prisoners-dilemma',
            opponent_coop_prob=0.5,
            hidden_size=16,
            learning_rate=0.001,
            num_episodes=100,
            seed=42
        )
        
        assert config.game_name == 'prisoners-dilemma'
        assert config.opponent_coop_prob == 0.5
        assert config.hidden_size == 16
        assert config.seed == 42
    
    def test_config_str(self):
        """Test config string representation."""
        config = ExperimentConfig(
            game_name='stag-hunt',
            opponent_coop_prob=0.7,
            hidden_size=32
        )
        
        config_str = str(config)
        assert 'stag-hunt' in config_str
        assert '0.7' in config_str
        assert '32' in config_str


class TestExperimentResult:
    """Test ExperimentResult data structure."""
    
    def test_create_result(self):
        """Test creating experiment result."""
        config = ExperimentConfig(
            game_name='prisoners-dilemma',
            opponent_coop_prob=0.5,
            hidden_size=16
        )
        
        agent = RepresentationAgent(input_dim=8, hidden_size=16)
        
        result = ExperimentResult(
            config=config,
            agent=agent,
            training_rewards=[1.0, 1.5, 2.0],
            training_losses=[0.5, 0.3, 0.1],
            final_reward=2.0,
            convergence_episode=50
        )
        
        assert result.config == config
        assert result.agent == agent
        assert len(result.training_rewards) == 3
        assert result.final_reward == 2.0
        assert result.convergence_episode == 50
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        config = ExperimentConfig(
            game_name='prisoners-dilemma',
            opponent_coop_prob=0.5,
            hidden_size=16
        )
        
        agent = RepresentationAgent(input_dim=8, hidden_size=16)
        
        # Create mock FSM
        fsm = FSM(
            states={0, 1, 2, 3},
            alphabet=["START", "CC", "CD", "DC", "DD"],
            transitions={(0, "START"): (1, 0), (1, "CC"): (2, 0)},
            initial_state=0,
            n_states=4
        )
        
        result = ExperimentResult(
            config=config,
            agent=agent,
            training_rewards=[1.0, 2.0],
            training_losses=[0.5, 0.2],
            final_reward=2.0,
            convergence_episode=50,
            fsm=fsm,
            fsm_accuracy=0.92
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['config']['game_name'] == 'prisoners-dilemma'
        assert result_dict['final_reward'] == 2.0
        assert result_dict['num_states'] == 4
        assert result_dict['accuracy'] == 0.92


class TestAnalysisPipeline:
    """Test AnalysisPipeline functionality."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_pipeline_initialization(self, temp_output_dir):
        """Test pipeline initialization creates directories."""
        pipeline = AnalysisPipeline(output_dir=temp_output_dir)
        
        assert pipeline.output_dir.exists()
        assert pipeline.plots_dir.exists()
        assert pipeline.tables_dir.exists()
        assert pipeline.reports_dir.exists()
        assert len(pipeline.results) == 0
    
    def test_run_single_experiment(self, temp_output_dir):
        """Test running single experiment (quick version)."""
        pipeline = AnalysisPipeline(output_dir=temp_output_dir)
        
        config = ExperimentConfig(
            game_name='prisoners-dilemma',
            opponent_coop_prob=0.5,
            hidden_size=8,
            num_episodes=50,  # Quick test
            episode_length=10
        )
        
        result = pipeline.run_experiment(config)
        
        assert result.config == config
        assert result.agent is not None
        assert len(result.training_rewards) == 50
        assert len(result.training_losses) == 50
        assert result.final_reward is not None
        assert result.fsm_model is not None
        assert result.fsm_validation is not None
        assert result.feature_importance is not None
        assert len(pipeline.results) == 1
    
    def test_run_capacity_analysis(self, temp_output_dir):
        """Test capacity analysis with multiple hidden sizes."""
        pipeline = AnalysisPipeline(output_dir=temp_output_dir)
        
        results = pipeline.run_capacity_analysis(
            game_name='prisoners-dilemma',
            opponent_coop_prob=0.5,
            hidden_sizes=[4, 8],  # Small for speed
            num_seeds=2
        )
        
        assert len(results) == 4  # 2 sizes × 2 seeds
        assert all(r.config.game_name == 'prisoners-dilemma' for r in results)
        assert all(r.config.opponent_coop_prob == 0.5 for r in results)
        
        # Check different hidden sizes
        hidden_sizes = [r.config.hidden_size for r in results]
        assert 4 in hidden_sizes
        assert 8 in hidden_sizes


class TestPlotting:
    """Test plotting functions."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_results(self):
        """Create sample results for plotting."""
        results = []
        for h in [8, 16]:
            for seed in range(2):
                config = ExperimentConfig(
                    game_name='prisoners-dilemma',
                    opponent_coop_prob=0.5,
                    hidden_size=h,
                    seed=seed
                )
                
                agent = RepresentationAgent(input_dim=8, hidden_size=h)
                
                fsm = FSM(
                    states=set(range(h // 2)),
                    alphabet=["START", "CC", "CD", "DC", "DD"],
                    transitions={(0, "START"): (1, 0)},
                    initial_state=0,
                    n_states=h // 2
                )
                
                feature_importance = FeatureImportance(
                    feature_names=['f1', 'f2', 'f3'],
                    mean_attributions=np.array([0.5, 0.3, 0.2]),
                    std_attributions=np.array([0.1, 0.1, 0.1]),
                    total_observations=100
                )
                
                result = ExperimentResult(
                    config=config,
                    agent=agent,
                    training_rewards=list(np.random.rand(50)),
                    training_losses=list(np.random.rand(50)),
                    final_reward=1.5 + seed * 0.1,
                    convergence_episode=30,
                    fsm=fsm,
                    fsm_accuracy=0.90,
                    feature_importance=feature_importance
                )
                results.append(result)
        
        return results
    
    def test_plot_capacity_curve(self, temp_output_dir, sample_results):
        """Test capacity curve plotting."""
        pipeline = AnalysisPipeline(output_dir=temp_output_dir)
        pipeline.results = sample_results
        
        save_path = Path(temp_output_dir) / 'test_capacity.png'
        pipeline.plot_capacity_curve(sample_results, save_path=str(save_path))
        
        assert save_path.exists()
    
    def test_plot_training_curves(self, temp_output_dir, sample_results):
        """Test training curves plotting."""
        pipeline = AnalysisPipeline(output_dir=temp_output_dir)
        
        save_path = Path(temp_output_dir) / 'test_training.png'
        pipeline.plot_training_curves(sample_results[0], save_path=str(save_path))
        
        assert save_path.exists()
    
    def test_plot_attribution_heatmap(self, temp_output_dir, sample_results):
        """Test attribution heatmap plotting."""
        pipeline = AnalysisPipeline(output_dir=temp_output_dir)
        
        save_path = Path(temp_output_dir) / 'test_attribution.png'
        pipeline.plot_attribution_heatmap(sample_results[0], save_path=str(save_path))
        
        assert save_path.exists()
    
    def test_plot_attribution_heatmap_no_importance(self, temp_output_dir):
        """Test attribution heatmap with no feature importance."""
        pipeline = AnalysisPipeline(output_dir=temp_output_dir)
        
        config = ExperimentConfig(
            game_name='prisoners-dilemma',
            opponent_coop_prob=0.5,
            hidden_size=8
        )
        agent = RepresentationAgent(input_dim=8, hidden_size=8)
        
        result = ExperimentResult(
            config=config,
            agent=agent,
            training_rewards=[1.0],
            training_losses=[0.5],
            final_reward=1.0,
            convergence_episode=10,
            feature_importance=None  # No importance
        )
        
        save_path = Path(temp_output_dir) / 'test_no_attribution.png'
        pipeline.plot_attribution_heatmap(result, save_path=str(save_path))
        
        # Should not create file when no feature importance
        assert not save_path.exists()


class TestTables:
    """Test table generation functions."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_results(self):
        """Create sample results for tables."""
        results = []
        for game in ['prisoners-dilemma', 'stag-hunt']:
            for opp in [0.3, 0.7]:
                config = ExperimentConfig(
                    game_name=game,
                    opponent_coop_prob=opp,
                    hidden_size=16
                )
                
                agent = RepresentationAgent(input_dim=8, hidden_size=16)
                
                fsm = FSM(
                    states={0, 1, 2, 3, 4, 5, 6, 7},
                    alphabet=["START", "CC", "CD", "DC", "DD"],
                    transitions={(0, "START"): (1, 0)},
                    initial_state=0,
                    n_states=8
                )
                
                result = ExperimentResult(
                    config=config,
                    agent=agent,
                    training_rewards=[1.0, 2.0],
                    training_losses=[0.5, 0.2],
                    final_reward=1.8,
                    convergence_episode=40,
                    fsm=fsm,
                    fsm_accuracy=0.88
                )
                results.append(result)
        
        return results
    
    def test_generate_performance_table(self, temp_output_dir, sample_results):
        """Test performance table generation."""
        pipeline = AnalysisPipeline(output_dir=temp_output_dir)
        
        save_path = Path(temp_output_dir) / 'perf_table.md'
        table_md = pipeline.generate_performance_table(
            sample_results,
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        assert 'prisoners-dilemma' in table_md
        assert 'stag-hunt' in table_md
        assert '0.3' in table_md
        assert '0.7' in table_md
        assert 'Final Reward' in table_md
    
    def test_generate_parameter_table(self, temp_output_dir, sample_results):
        """Test parameter table generation."""
        pipeline = AnalysisPipeline(output_dir=temp_output_dir)
        
        save_path = Path(temp_output_dir) / 'param_table.md'
        table_md = pipeline.generate_parameter_table(
            sample_results,
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        assert 'Hidden Size' in table_md
        assert 'Total Params' in table_md
        assert '16' in table_md


class TestReports:
    """Test report generation functions."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_write_network_size_justification(self, temp_output_dir):
        """Test network size justification report."""
        pipeline = AnalysisPipeline(output_dir=temp_output_dir)
        
        save_path = Path(temp_output_dir) / 'network_size.md'
        report = pipeline.write_network_size_justification(save_path=str(save_path))
        
        assert save_path.exists()
        assert 'Network Size Justification' in report
        assert 'LSTM' in report
        assert 'Capacity Analysis' in report
    
    def test_write_extraction_results(self, temp_output_dir):
        """Test FSM extraction results report."""
        pipeline = AnalysisPipeline(output_dir=temp_output_dir)
        
        save_path = Path(temp_output_dir) / 'extraction.md'
        report = pipeline.write_extraction_results(save_path=str(save_path))
        
        assert save_path.exists()
        assert 'FSM Extraction Results' in report
        assert 'L*' in report
        assert 'Hopcroft' in report
    
    def test_write_attribution_analysis(self, temp_output_dir):
        """Test attribution analysis report."""
        pipeline = AnalysisPipeline(output_dir=temp_output_dir)
        
        save_path = Path(temp_output_dir) / 'attribution.md'
        report = pipeline.write_attribution_analysis(save_path=str(save_path))
        
        assert save_path.exists()
        assert 'Attribution Analysis Results' in report
        assert 'Integrated Gradients' in report
        assert 'Feature Importance' in report
    
    def test_write_full_report(self, temp_output_dir):
        """Test full comprehensive report."""
        pipeline = AnalysisPipeline(output_dir=temp_output_dir)
        
        save_path = Path(temp_output_dir) / 'full_report.md'
        report = pipeline.write_full_report(save_path=str(save_path))
        
        assert save_path.exists()
        assert 'Complete Analysis Report' in report
        assert 'Nature Communications' in report
        assert 'Executive Summary' in report
        assert 'Methodology' in report
        assert 'Results' in report
        assert 'Discussion' in report
        assert 'Conclusion' in report


class TestResultsSaving:
    """Test saving and loading results."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_save_results(self, temp_output_dir):
        """Test saving results to JSON."""
        pipeline = AnalysisPipeline(output_dir=temp_output_dir)
        
        # Add sample result
        config = ExperimentConfig(
            game_name='prisoners-dilemma',
            opponent_coop_prob=0.5,
            hidden_size=16
        )
        agent = RepresentationAgent(input_dim=8, hidden_size=16)
        
        result = ExperimentResult(
            config=config,
            agent=agent,
            training_rewards=[1.0, 2.0],
            training_losses=[0.5, 0.2],
            final_reward=2.0,
            convergence_episode=50
        )
        
        pipeline.results.append(result)
        
        # Save results
        pipeline.save_results('test_results.json')
        
        save_path = Path(temp_output_dir) / 'test_results.json'
        assert save_path.exists()
        
        # Load and verify
        with open(save_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert len(loaded_data) == 1
        assert loaded_data[0]['config']['game_name'] == 'prisoners-dilemma'
        assert loaded_data[0]['final_reward'] == 2.0


def test_gate7_integration():
    """
    Integration test: Run complete analysis pipeline.
    
    Tests full workflow:
    1. Run capacity analysis (small scale)
    2. Generate all plots
    3. Generate all tables
    4. Generate all reports
    5. Save results
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        pipeline = AnalysisPipeline(output_dir=temp_dir)
        
        # Run capacity analysis (minimal for speed)
        results = pipeline.run_capacity_analysis(
            game_name='prisoners-dilemma',
            opponent_coop_prob=0.5,
            hidden_sizes=[4, 8],
            num_seeds=2
        )
        
        assert len(results) == 4
        
        # Generate plots
        pipeline.plot_capacity_curve(results)
        pipeline.plot_training_curves(results[0])
        pipeline.plot_attribution_heatmap(results[0])
        
        # Check plots created
        plots_dir = Path(temp_dir) / 'plots'
        assert (plots_dir / 'capacity_curve.png').exists()
        assert len(list(plots_dir.glob('training_curves_*.png'))) > 0
        assert len(list(plots_dir.glob('attribution_heatmap_*.png'))) > 0
        
        # Generate tables
        pipeline.generate_performance_table(results)
        pipeline.generate_parameter_table(results)
        
        # Check tables created
        tables_dir = Path(temp_dir) / 'tables'
        assert (tables_dir / 'performance_table.md').exists()
        assert (tables_dir / 'parameter_table.md').exists()
        
        # Generate reports
        pipeline.write_network_size_justification()
        pipeline.write_extraction_results()
        pipeline.write_attribution_analysis()
        pipeline.write_full_report()
        
        # Check reports created
        reports_dir = Path(temp_dir) / 'reports'
        assert (reports_dir / 'network_size_justification.md').exists()
        assert (reports_dir / 'extraction_results.md').exists()
        assert (reports_dir / 'attribution_analysis.md').exists()
        assert (reports_dir / 'full_analysis_report.md').exists()
        
        # Save results
        pipeline.save_results()
        assert (Path(temp_dir) / 'experiment_results.json').exists()
        
        print("\n" + "="*60)
        print("GATE 7 INTEGRATION TEST PASSED")
        print("="*60)
        print(f"✓ Ran {len(results)} experiments")
        print(f"✓ Generated {len(list(plots_dir.glob('*.png')))} plots")
        print(f"✓ Generated {len(list(tables_dir.glob('*.md')))} tables")
        print(f"✓ Generated {len(list(reports_dir.glob('*.md')))} reports")
        print(f"✓ Saved results to JSON")
        print("="*60)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
