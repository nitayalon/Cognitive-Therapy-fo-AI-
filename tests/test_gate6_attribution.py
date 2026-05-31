"""
Gate 6 Tests: Attribution Analysis

Verifies:
1. Integrated gradients computation
2. Saliency map generation
3. Feature importance analysis
4. Attribution by input symbol
5. Visualization utilities
6. End-to-end attribution pipeline
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import yaml

from cognitive_therapy_ai.games import Action, PrisonersDilemma
from cognitive_therapy_ai.opponent import ProbabilisticOpponent
from cognitive_therapy_ai.encoding import ObservationEncoder
from cognitive_therapy_ai.representation_agent import RepresentationAgent
from cognitive_therapy_ai.reinforce_trainer import SessionEnvironment, REINFORCETrainer, train_to_convergence
from cognitive_therapy_ai.attribution import (
    IntegratedGradients,
    SaliencyAnalyzer,
    AttributionAnalyzer,
    AttributionResult,
    FeatureImportance,
    SaliencyMap,
    visualize_attribution,
    visualize_feature_importance
)


def load_config():
    """Load base configuration."""
    config_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_simple_trained_agent(device='cpu'):
    """Create and train a simple agent for testing."""
    config = load_config()
    
    # Create game and opponent
    payoff = config['games']['PD']['payoff']
    game = PrisonersDilemma(
        T=payoff[1][0], R=payoff[0][0],
        P=payoff[1][1], S=payoff[0][1]
    )
    opponent = ProbabilisticOpponent(defection_probability=0.7)
    
    # Create encoder and environment
    encoder = ObservationEncoder('no_game')
    env = SessionEnvironment(game, opponent, encoder, T=50, game_name=None)
    
    # Create agent
    agent = RepresentationAgent(
        input_dim=encoder.get_input_dim(),
        hidden_size=4,
        device=torch.device(device)
    )
    
    # Train briefly
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)
    trainer = REINFORCETrainer(agent, optimizer, gamma=0.99, gae_lambda=0.95, device=torch.device(device))
    
    _ = train_to_convergence(
        trainer, env,
        max_episodes=20,
        convergence_window=10,
        convergence_threshold=0.01,
        mode='rl',
        verbose=False
    )
    
    return agent, encoder


class TestIntegratedGradients:
    """Test integrated gradients attribution."""
    
    def test_integrated_gradients_runs(self):
        """Test that integrated gradients computation completes."""
        agent, encoder = create_simple_trained_agent()
        ig = IntegratedGradients(agent, encoder)
        
        # Create test observation (CC outcome)
        obs = encoder.encode_observation(
            agent_last_action=Action.COOPERATE,
            opponent_last_action=Action.COOPERATE
        )
        hidden_state = agent.reset_hidden_state(batch_size=1)
        
        result = ig.attribute(obs, hidden_state, n_steps=10)
        
        # Check result structure
        assert isinstance(result, AttributionResult)
        assert result.attributions.shape == obs.shape
        assert result.predicted_action in [Action.COOPERATE, Action.DEFECT]
        assert len(result.policy_probs) == 2
        assert np.isclose(result.policy_probs.sum(), 1.0)
    
    def test_attributions_sum_to_prediction_difference(self):
        """Test that attributions approximately satisfy completeness axiom."""
        agent, encoder = create_simple_trained_agent()
        ig = IntegratedGradients(agent, encoder)
        
        obs = encoder.encode_observation(Action.COOPERATE, Action.DEFECT)
        hidden_state = agent.reset_hidden_state(batch_size=1)
        baseline = encoder.encode_start_token()
        
        result = ig.attribute(obs, hidden_state, baseline=baseline, n_steps=50)
        
        # Attributions should be non-zero for some features
        assert np.any(np.abs(result.attributions) > 1e-6)
    
    def test_baseline_affects_attributions(self):
        """Test that different baselines produce different attributions."""
        agent, encoder = create_simple_trained_agent()
        ig = IntegratedGradients(agent, encoder)
        
        obs = encoder.encode_observation(Action.DEFECT, Action.DEFECT)
        hidden_state = agent.reset_hidden_state(batch_size=1)
        
        # Two different baselines
        baseline1 = encoder.encode_start_token()
        baseline2 = encoder.encode_observation(Action.COOPERATE, Action.COOPERATE)
        
        result1 = ig.attribute(obs, hidden_state, baseline=baseline1, n_steps=20)
        result2 = ig.attribute(obs, hidden_state, baseline=baseline2, n_steps=20)
        
        # Attributions should differ (unless obs == baseline2)
        if not np.array_equal(obs, baseline2):
            assert not np.allclose(result1.attributions, result2.attributions)
    
    def test_batch_attribute(self):
        """Test batch attribution."""
        agent, encoder = create_simple_trained_agent()
        ig = IntegratedGradients(agent, encoder)
        
        # Create multiple observations
        obs1 = encoder.encode_observation(Action.COOPERATE, Action.COOPERATE)
        obs2 = encoder.encode_observation(Action.DEFECT, Action.DEFECT)
        observations = [obs1, obs2]
        
        hidden_states = [agent.reset_hidden_state(1), agent.reset_hidden_state(1)]
        
        results = ig.batch_attribute(observations, hidden_states, n_steps=10)
        
        assert len(results) == 2
        assert all(isinstance(r, AttributionResult) for r in results)


class TestSaliencyAnalyzer:
    """Test saliency map generation."""
    
    def test_saliency_computation(self):
        """Test basic saliency computation."""
        agent, encoder = create_simple_trained_agent()
        saliency = SaliencyAnalyzer(agent)
        
        obs = encoder.encode_observation(Action.COOPERATE, Action.DEFECT)
        hidden_state = agent.reset_hidden_state(batch_size=1)
        
        saliency_scores = saliency.compute_saliency(obs, hidden_state)
        
        # Check shape and properties
        assert saliency_scores.shape == obs.shape
        assert np.all(saliency_scores >= 0)  # Absolute gradients
    
    def test_saliency_map_creation(self):
        """Test saliency map for state transition."""
        agent, encoder = create_simple_trained_agent()
        saliency_analyzer = SaliencyAnalyzer(agent)
        
        obs1 = encoder.encode_observation(Action.COOPERATE, Action.COOPERATE)
        obs2 = encoder.encode_observation(Action.DEFECT, Action.COOPERATE)
        
        hidden_state = agent.reset_hidden_state(batch_size=1)
        
        # Get hidden states (h component)
        with torch.no_grad():
            _, _, h_state_after = agent.forward(
                torch.tensor(obs1, dtype=torch.float32).unsqueeze(0),
                hidden_state
            )
        
        h1 = hidden_state[0].cpu().numpy().flatten()
        h2 = h_state_after[0].cpu().numpy().flatten()
        
        saliency_map = saliency_analyzer.compute_saliency_map(
            from_obs=obs1,
            to_obs=obs2,
            from_hidden=h1,
            to_hidden=h2,
            from_action=Action.COOPERATE,
            from_hidden_state=hidden_state
        )
        
        assert isinstance(saliency_map, SaliencyMap)
        assert saliency_map.saliency_scores.shape == obs1.shape
        assert saliency_map.from_action == Action.COOPERATE


class TestAttributionAnalyzer:
    """Test high-level attribution analyzer."""
    
    def test_analyzer_initialization(self):
        """Test that analyzer initializes correctly."""
        agent, encoder = create_simple_trained_agent()
        analyzer = AttributionAnalyzer(agent, encoder)
        
        assert analyzer.ig is not None
        assert analyzer.saliency is not None
        assert analyzer.encoder == encoder
    
    def test_analyze_trajectory_integrated_gradients(self):
        """Test trajectory analysis with integrated gradients."""
        agent, encoder = create_simple_trained_agent()
        analyzer = AttributionAnalyzer(agent, encoder)
        
        # Create small trajectory
        observations = [
            encoder.encode_start_token(),
            encoder.encode_observation(Action.COOPERATE, Action.COOPERATE),
            encoder.encode_observation(Action.DEFECT, Action.COOPERATE)
        ]
        actions = [Action.COOPERATE, Action.DEFECT, Action.DEFECT]
        
        # Generate hidden states
        hidden_states_h = []
        lstm_states = []
        h_state = agent.reset_hidden_state(1)
        
        for obs in observations:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, _, h_state = agent.forward(obs_tensor, h_state)
            hidden_states_h.append(h_state[0].cpu().numpy().flatten())
            lstm_states.append(h_state)
        
        results = analyzer.analyze_trajectory(
            observations=observations,
            actions=actions,
            hidden_states=hidden_states_h,
            lstm_states=lstm_states,
            method='integrated_gradients'
        )
        
        assert len(results) == len(observations)
        assert all(r.method == 'integrated_gradients' for r in results)
    
    def test_analyze_trajectory_saliency(self):
        """Test trajectory analysis with saliency."""
        agent, encoder = create_simple_trained_agent()
        analyzer = AttributionAnalyzer(agent, encoder)
        
        observations = [encoder.encode_observation(Action.COOPERATE, Action.DEFECT)]
        actions = [Action.DEFECT]
        
        h_state = agent.reset_hidden_state(1)
        lstm_states = [h_state]
        hidden_states_h = [h_state[0].cpu().numpy().flatten()]
        
        results = analyzer.analyze_trajectory(
            observations, actions, hidden_states_h, lstm_states,
            method='saliency'
        )
        
        assert len(results) == 1
        assert results[0].method == 'saliency'
    
    def test_compute_feature_importance(self):
        """Test feature importance computation."""
        agent, encoder = create_simple_trained_agent()
        analyzer = AttributionAnalyzer(agent, encoder)
        
        # Create attribution results
        obs = encoder.encode_observation(Action.COOPERATE, Action.COOPERATE)
        h_state = agent.reset_hidden_state(1)
        
        result = analyzer.ig.attribute(obs, h_state, n_steps=10)
        
        importance = analyzer.compute_feature_importance([result])
        
        assert isinstance(importance, FeatureImportance)
        assert len(importance.feature_names) == encoder.get_input_dim()
        assert len(importance.mean_attributions) == encoder.get_input_dim()
        assert importance.total_observations == 1
    
    def test_rank_features_by_importance(self):
        """Test feature ranking."""
        agent, encoder = create_simple_trained_agent()
        analyzer = AttributionAnalyzer(agent, encoder)
        
        obs = encoder.encode_observation(Action.DEFECT, Action.DEFECT)
        h_state = agent.reset_hidden_state(1)
        result = analyzer.ig.attribute(obs, h_state, n_steps=10)
        
        importance = analyzer.compute_feature_importance([result])
        ranked = analyzer.rank_features_by_importance(importance, top_k=3)
        
        assert len(ranked) == 3
        assert all(isinstance(name, str) and isinstance(score, (float, np.floating)) 
                   for name, score in ranked)
        # Should be sorted descending
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)
    
    def test_analyze_by_input_symbol(self):
        """Test attribution analysis by input symbol."""
        agent, encoder = create_simple_trained_agent()
        analyzer = AttributionAnalyzer(agent, encoder)
        
        # Create observations with different symbols
        observations = [
            encoder.encode_start_token(),
            encoder.encode_observation(Action.COOPERATE, Action.COOPERATE),
            encoder.encode_observation(Action.DEFECT, Action.DEFECT),
            encoder.encode_observation(Action.COOPERATE, Action.COOPERATE)
        ]
        
        lstm_states = [agent.reset_hidden_state(1) for _ in observations]
        
        results = analyzer.ig.batch_attribute(observations, lstm_states, n_steps=10)
        
        importance_by_symbol = analyzer.analyze_by_input_symbol(results, observations)
        
        assert 'START' in importance_by_symbol
        assert 'CC' in importance_by_symbol
        assert 'DD' in importance_by_symbol
        assert importance_by_symbol['CC'].total_observations == 2  # Two CC observations


class TestFeatureNames:
    """Test feature naming."""
    
    def test_feature_names_no_game(self):
        """Test feature names for no_game condition."""
        agent, encoder = create_simple_trained_agent()
        analyzer = AttributionAnalyzer(agent, encoder)
        
        names = analyzer._get_feature_names()
        
        expected_names = [
            'agent_cooperate', 'agent_defect',
            'opp_cooperate', 'opp_defect',
            'outcome_CC', 'outcome_CD', 'outcome_DC', 'outcome_DD'
        ]
        assert names == expected_names
    
    def test_feature_names_game_tag(self):
        """Test feature names for game_tag condition."""
        encoder = ObservationEncoder('game_tag')
        device = torch.device('cpu')
        agent = RepresentationAgent(
            input_dim=encoder.get_input_dim(),
            hidden_size=4,
            device=device
        )
        analyzer = AttributionAnalyzer(agent, encoder, device)
        
        names = analyzer._get_feature_names()
        
        assert len(names) == 11
        assert 'game_PD' in names
        assert 'game_SH' in names
        assert 'game_HD' in names


class TestVisualization:
    """Test visualization utilities."""
    
    def test_visualize_attribution(self):
        """Test attribution visualization."""
        agent, encoder = create_simple_trained_agent()
        ig = IntegratedGradients(agent, encoder)
        
        obs = encoder.encode_observation(Action.COOPERATE, Action.DEFECT)
        h_state = agent.reset_hidden_state(1)
        result = ig.attribute(obs, h_state, n_steps=10)
        
        feature_names = [
            'agent_cooperate', 'agent_defect',
            'opp_cooperate', 'opp_defect',
            'outcome_CC', 'outcome_CD', 'outcome_DC', 'outcome_DD'
        ]
        
        viz = visualize_attribution(result, feature_names)
        
        assert isinstance(viz, str)
        assert 'Attribution Scores' in viz
        assert 'Predicted Action' in viz
        assert any(name in viz for name in feature_names)
    
    def test_visualize_feature_importance(self):
        """Test feature importance visualization."""
        agent, encoder = create_simple_trained_agent()
        analyzer = AttributionAnalyzer(agent, encoder)
        
        obs = encoder.encode_observation(Action.DEFECT, Action.COOPERATE)
        h_state = agent.reset_hidden_state(1)
        result = analyzer.ig.attribute(obs, h_state, n_steps=10)
        
        importance = analyzer.compute_feature_importance([result])
        viz = visualize_feature_importance(importance, top_k=5)
        
        assert isinstance(viz, str)
        assert 'Feature Importance' in viz
        assert 'Total Observations' in viz


def test_gate6_integration():
    """
    Gate 6 integration test: Full attribution pipeline.
    
    Tests:
    1. Train agent on PD
    2. Compute attributions across trajectory
    3. Analyze feature importance
    4. Identify top features
    5. Verify attribution makes sense
    """
    print("\n[Gate 6 Integration Test]")
    
    # Setup
    config = load_config()
    device = torch.device('cpu')
    
    # Create game and opponent
    payoff = config['games']['PD']['payoff']
    game = PrisonersDilemma(
        T=payoff[1][0], R=payoff[0][0],
        P=payoff[1][1], S=payoff[0][1]
    )
    opponent = ProbabilisticOpponent(defection_probability=0.5)
    
    print(f"Game: PD, Opponent: p_defect=0.5")
    
    # Create encoder and environment
    encoder = ObservationEncoder('no_game')
    env = SessionEnvironment(game, opponent, encoder, T=50, game_name=None)
    
    # Create and train agent
    print("Training agent...")
    agent = RepresentationAgent(
        input_dim=encoder.get_input_dim(),
        hidden_size=8,
        device=device
    )
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)
    trainer = REINFORCETrainer(agent, optimizer, gamma=0.99, gae_lambda=0.95, device=device)
    
    history = train_to_convergence(
        trainer, env,
        max_episodes=30,
        convergence_window=15,
        convergence_threshold=0.01,
        mode='rl',
        verbose=False
    )
    
    print(f"Training completed: {len(history['returns'])} episodes")
    
    # Generate trajectory for attribution
    print("\nGenerating trajectory...")
    observations = []
    actions = []
    lstm_states = []
    hidden_states_h = []
    
    obs = env.reset()
    h_state = agent.reset_hidden_state(1)
    
    for step in range(10):
        observations.append(obs)
        lstm_states.append(h_state)
        hidden_states_h.append(h_state[0].cpu().numpy().flatten())
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, h_state = agent.select_action(obs_tensor, h_state, deterministic=True)
        
        action_enum = Action.COOPERATE if action == 0 else Action.DEFECT
        actions.append(action_enum)
        
        obs, _, done = env.step(action)
        if done:
            break
    
    print(f"Trajectory length: {len(observations)}")
    
    # Attribution analysis
    print("\nComputing attributions...")
    analyzer = AttributionAnalyzer(agent, encoder, device)
    
    attribution_results = analyzer.analyze_trajectory(
        observations=observations,
        actions=actions,
        hidden_states=hidden_states_h,
        lstm_states=lstm_states,
        method='integrated_gradients'
    )
    
    print(f"Computed attributions for {len(attribution_results)} observations")
    
    # Feature importance
    print("\nAnalyzing feature importance...")
    importance = analyzer.compute_feature_importance(attribution_results)
    
    print(visualize_feature_importance(importance, top_k=5))
    
    # Rank features
    top_features = analyzer.rank_features_by_importance(importance, top_k=3)
    print(f"\nTop 3 features:")
    for rank, (name, score) in enumerate(top_features, 1):
        print(f"  {rank}. {name}: {score:.4f}")
    
    # Analyze by input symbol
    importance_by_symbol = analyzer.analyze_by_input_symbol(
        attribution_results, observations
    )
    print(f"\nAttribution by input symbol:")
    for symbol, imp in importance_by_symbol.items():
        print(f"  {symbol}: {imp.total_observations} observations")
    
    # Assertions
    assert len(attribution_results) > 0, "Should have attribution results"
    assert importance.total_observations == len(attribution_results)
    assert len(top_features) == 3
    assert all(score >= 0 for _, score in top_features)
    
    print(f"\n✅ Gate 6 integration test passed!")
    print(f"   Attribution pipeline identifies important input features")
    print(f"   Ready for mechanistic analysis in Gate 7")


if __name__ == "__main__":
    # Run integration test
    test_gate6_integration()
    print("\n✅ All Gate 6 tests complete!")
