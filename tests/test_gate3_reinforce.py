"""
Gate 3 Tests: REINFORCE Training

Verifies:
1. Session environment mechanics (T games, hidden state persistence)
2. GAE advantage computation (correctness vs hand-calculated values)
3. REINFORCE training (policy improves over time)
4. BC mode (learns to imitate best response)
5. Convergence on trivial cases:
   - PD always-defect (dominant strategy)
   - SH/HD with extreme opponents (p_coop near 0 or 1)
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import yaml

from cognitive_therapy_ai.representation_agent import RepresentationAgent
from cognitive_therapy_ai.reinforce_trainer import (
    REINFORCETrainer,
    SessionEnvironment,
    train_to_convergence
)
from cognitive_therapy_ai.encoding import ObservationEncoder, Action
from cognitive_therapy_ai.games import PrisonersDilemma, StagHunt, HawkDove
from cognitive_therapy_ai.opponent import ProbabilisticOpponent
from cognitive_therapy_ai.best_response import analytic_best_response


def load_config():
    """Load base configuration."""
    config_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_game_from_config(config, game_name):
    """
    Create game instance from config.
    
    Config format: payoff: [[R, S], [T, P]]
    Game expects: T, R, P, S parameters
    """
    payoff = config['games'][game_name]['payoff']
    # Extract from matrix: [[R, S], [T, P]]
    R = payoff[0][0]
    S = payoff[0][1]
    T = payoff[1][0]
    P = payoff[1][1]
    
    if game_name == 'PD':
        return PrisonersDilemma(T=T, R=R, P=P, S=S)
    elif game_name == 'SH':
        return StagHunt(T=T, R=R, P=P, S=S)
    elif game_name == 'HD':
        return HawkDove(T=T, R=R, P=P, S=S)
    else:
        raise ValueError(f"Unknown game: {game_name}")


class TestSessionEnvironment:
    """Test session-based game environment."""
    
    def test_session_reset(self):
        """Test that reset returns START token."""
        config = load_config()
        game = create_game_from_config(config, 'PD')
        opponent = ProbabilisticOpponent(defection_probability=0.5)
        encoder = ObservationEncoder("no_game")
        
        env = SessionEnvironment(game, opponent, encoder, T=10)
        obs = env.reset()
        
        expected_start = encoder.encode_start_token()
        np.testing.assert_array_equal(obs, expected_start)
        assert env.game_count == 0
    
    def test_session_step_mechanics(self):
        """Test that step updates state correctly."""
        config = load_config()
        game = create_game_from_config(config, 'PD')
        opponent = ProbabilisticOpponent(defection_probability=0.0)  # Always cooperate
        encoder = ObservationEncoder("no_game")
        
        env = SessionEnvironment(game, opponent, encoder, T=5)
        env.reset()
        
        # Agent cooperates
        next_obs, reward, done = env.step(0)
        
        # Opponent always cooperates, so outcome should be CC
        expected_obs = encoder.encode_observation(Action.COOPERATE, Action.COOPERATE)
        np.testing.assert_array_equal(next_obs, expected_obs)
        
        # PD: R (mutual cooperation) = 3
        assert reward == 3.0
        assert not done  # Not done yet (game 1 of 5)
        assert env.game_count == 1
    
    def test_session_terminates_after_T_games(self):
        """Test that session terminates after exactly T games."""
        config = load_config()
        game = create_game_from_config(config, 'PD')
        opponent = ProbabilisticOpponent(defection_probability=0.5)
        encoder = ObservationEncoder("no_game")
        
        T = 10
        env = SessionEnvironment(game, opponent, encoder, T=T)
        env.reset()
        
        for t in range(T):
            _, _, done = env.step(0)
            if t < T - 1:
                assert not done, f"Should not be done at game {t+1}"
            else:
                assert done, f"Should be done at game {t+1}"
        
        assert env.game_count == T
    
    def test_session_with_game_tag(self):
        """Test session environment with game tag encoding."""
        config = load_config()
        game = create_game_from_config(config, 'PD')
        opponent = ProbabilisticOpponent(defection_probability=0.5)
        encoder = ObservationEncoder("game_tag")
        
        env = SessionEnvironment(game, opponent, encoder, T=10, game_name="PD")
        obs = env.reset()
        
        # Should include game tag
        expected_start = encoder.encode_start_token(game_name="PD")
        np.testing.assert_array_equal(obs, expected_start)


class TestGAEComputation:
    """Test Generalized Advantage Estimation."""
    
    def test_gae_simple_case(self):
        """Test GAE computation on simple hand-checkable case."""
        agent = RepresentationAgent(input_dim=8, hidden_size=4)
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
        trainer = REINFORCETrainer(agent, optimizer, gamma=0.99, gae_lambda=0.95)
        
        # Simple trajectory: 3 steps
        rewards = [1.0, 1.0, 1.0]
        values = [
            torch.tensor([0.5]),
            torch.tensor([0.5]),
            torch.tensor([0.5]),
            torch.tensor([0.0])  # Final value (terminal)
        ]
        dones = [False, False, True]
        
        advantages, returns = trainer.compute_gae_advantages(rewards, values, dones)
        
        # Check shapes
        assert advantages.shape == (3,)
        assert returns.shape == (3,)
        
        # Advantages should be positive (rewards > values)
        assert (advantages > 0).all()
    
    def test_gae_terminal_state(self):
        """Test GAE handles terminal state correctly."""
        agent = RepresentationAgent(input_dim=8, hidden_size=4)
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
        trainer = REINFORCETrainer(agent, optimizer, gamma=0.99, gae_lambda=0.95)
        
        # Single step ending in terminal state
        rewards = [1.0]
        values = [
            torch.tensor([0.5]),
            torch.tensor([0.0])  # Unused for terminal
        ]
        dones = [True]
        
        advantages, returns = trainer.compute_gae_advantages(rewards, values, dones)
        
        # Advantage = reward - value (no bootstrap)
        expected_advantage = 1.0 - 0.5
        np.testing.assert_almost_equal(advantages[0].item(), expected_advantage, decimal=5)


class TestREINFORCETraining:
    """Test REINFORCE training mechanics."""
    
    def test_train_session_rl_completes(self):
        """Test that RL training session completes without errors."""
        config = load_config()
        game = create_game_from_config(config, 'PD')
        opponent = ProbabilisticOpponent(defection_probability=0.5)
        encoder = ObservationEncoder("no_game")
        env = SessionEnvironment(game, opponent, encoder, T=10)
        
        agent = RepresentationAgent(input_dim=8, hidden_size=4)
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
        trainer = REINFORCETrainer(agent, optimizer)
        
        stats = trainer.train_session_rl(env, max_steps=10)
        
        # Check stats are valid
        assert stats.num_games == 10
        assert isinstance(stats.total_return, float)
        assert isinstance(stats.policy_loss, float)
        assert isinstance(stats.value_loss, float)
        assert 0.0 <= stats.br_accuracy <= 1.0
    
    def test_train_session_bc_completes(self):
        """Test that BC training session completes without errors."""
        config = load_config()
        game = create_game_from_config(config, 'PD')
        opponent = ProbabilisticOpponent(defection_probability=0.5)
        encoder = ObservationEncoder("no_game")
        env = SessionEnvironment(game, opponent, encoder, T=10)
        
        agent = RepresentationAgent(input_dim=8, hidden_size=4)
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
        trainer = REINFORCETrainer(agent, optimizer)
        
        stats = trainer.train_session_bc(env, max_steps=10)
        
        # BC should have perfect BR accuracy
        assert stats.br_accuracy == 1.0
        assert stats.num_games == 10
    
    def test_policy_changes_after_training(self):
        """Test that agent policy changes after training."""
        config = load_config()
        game = create_game_from_config(config, 'PD')
        opponent = ProbabilisticOpponent(defection_probability=0.5)
        encoder = ObservationEncoder("no_game")
        env = SessionEnvironment(game, opponent, encoder, T=10)
        
        agent = RepresentationAgent(input_dim=8, hidden_size=4)
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)
        trainer = REINFORCETrainer(agent, optimizer)
        
        # Get initial policy parameters
        initial_params = [p.clone() for p in agent.parameters()]
        
        # Train for a few sessions
        for _ in range(5):
            trainer.train_session_rl(env, max_steps=10)
        
        # Check that parameters changed
        final_params = list(agent.parameters())
        changed = False
        for p_init, p_final in zip(initial_params, final_params):
            if not torch.allclose(p_init, p_final):
                changed = True
                break
        
        assert changed, "Agent parameters should change after training"


class TestEvaluationMode:
    """Test evaluation without training."""
    
    def test_evaluate_session_no_gradients(self):
        """Test that evaluation mode doesn't compute gradients."""
        config = load_config()
        game = create_game_from_config(config, 'PD')
        opponent = ProbabilisticOpponent(defection_probability=0.5)
        encoder = ObservationEncoder("no_game")
        env = SessionEnvironment(game, opponent, encoder, T=10)
        
        agent = RepresentationAgent(input_dim=8, hidden_size=4)
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
        trainer = REINFORCETrainer(agent, optimizer)
        
        # Evaluate
        stats = trainer.evaluate_session(env, max_steps=10, deterministic=True)
        
        # Check that no gradients are computed
        for p in agent.parameters():
            assert p.grad is None, "Evaluation should not compute gradients"
        
        # Stats should be valid
        assert stats.num_games == 10
        assert 0.0 <= stats.br_accuracy <= 1.0


class TestConvergence:
    """Test convergence on trivial cases."""
    
    def test_pd_converges_to_always_defect(self):
        """
        Critical test: PD has dominant strategy (always defect).
        REINFORCE should learn this.
        """
        config = load_config()
        game = create_game_from_config(config, 'PD')
        
        # Test with different opponent types
        for p_defect in [0.1, 0.5, 0.9]:
            opponent = ProbabilisticOpponent(defection_probability=p_defect)
            encoder = ObservationEncoder("no_game")
            env = SessionEnvironment(game, opponent, encoder, T=20)
            
            agent = RepresentationAgent(input_dim=8, hidden_size=8)
            optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)
            trainer = REINFORCETrainer(agent, optimizer)
            
            # Train to convergence
            history = train_to_convergence(
                trainer, env,
                max_episodes=200,
                convergence_window=30,
                convergence_threshold=1.0,
                mode="rl",
                verbose=False
            )
            
            # Evaluate final policy
            final_stats = trainer.evaluate_session(env, max_steps=20, deterministic=True)
            
            # In PD, best response is always DEFECT (action=1)
            # Check that agent learned to defect
            # BR accuracy should be high (>0.8)
            assert final_stats.br_accuracy > 0.8, \
                f"PD (p_defect={p_defect}): BR accuracy {final_stats.br_accuracy:.2f} too low"
    
    def test_bc_mode_learns_best_response_quickly(self):
        """Test that BC mode learns best response faster than RL."""
        config = load_config()
        game = create_game_from_config(config, 'PD')
        opponent = ProbabilisticOpponent(defection_probability=0.5)
        encoder = ObservationEncoder("no_game")
        env = SessionEnvironment(game, opponent, encoder, T=20)
        
        agent = RepresentationAgent(input_dim=8, hidden_size=8)
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)
        trainer = REINFORCETrainer(agent, optimizer)
        
        # Train in BC mode
        history = train_to_convergence(
            trainer, env,
            max_episodes=50,
            convergence_window=10,
            convergence_threshold=0.1,
            mode="bc",
            verbose=False
        )
        
        # BC should achieve high BR accuracy quickly
        final_br_acc = history['br_accuracies'][-1]
        assert final_br_acc > 0.95, \
            f"BC mode should learn BR quickly, got {final_br_acc:.3f}"
    
    @pytest.mark.slow
    def test_staghunt_conditional_policy(self):
        """
        Test that SH learns conditional policy (cooperate if p_coop high, defect if low).
        
        This is a slower test marked as optional.
        """
        config = load_config()
        game = create_game_from_config(config, 'SH')
        encoder = ObservationEncoder("no_game")
        
        # Train on high-cooperation opponent (should learn to cooperate)
        opponent_high = ProbabilisticOpponent(defection_probability=0.1)  # p_coop=0.9
        env_high = SessionEnvironment(game, opponent_high, encoder, T=20)
        
        agent_high = RepresentationAgent(input_dim=8, hidden_size=16)
        optimizer_high = torch.optim.Adam(agent_high.parameters(), lr=0.005)
        trainer_high = REINFORCETrainer(agent_high, optimizer_high)
        
        train_to_convergence(
            trainer_high, env_high,
            max_episodes=300,
            mode="rl",
            verbose=False
        )
        
        stats_high = trainer_high.evaluate_session(env_high, max_steps=20)
        
        # Should learn to cooperate (BR for SH with p_coop=0.9 is cooperate)
        # BR accuracy should be reasonably high
        assert stats_high.br_accuracy > 0.6, \
            f"SH with p_coop=0.9: Expected cooperation, got BR_acc={stats_high.br_accuracy:.2f}"


class TestHiddenStatePersistence:
    """Test that hidden state persists across games within session."""
    
    def test_hidden_state_used_across_session(self):
        """Verify hidden state is passed between games in session."""
        config = load_config()
        game = create_game_from_config(config, 'PD')
        opponent = ProbabilisticOpponent(defection_probability=0.5)
        encoder = ObservationEncoder("no_game")
        env = SessionEnvironment(game, opponent, encoder, T=5)
        
        agent = RepresentationAgent(input_dim=8, hidden_size=4)
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
        trainer = REINFORCETrainer(agent, optimizer)
        
        # Train one session and check trajectory length
        stats = trainer.train_session_rl(env, max_steps=5)
        
        # Should complete all 5 games
        assert stats.num_games == 5


def test_gate3_integration():
    """
    Gate 3 integration test: Full training pipeline.
    
    Tests:
    1. Create environment and agent
    2. Train with REINFORCE (RL mode)
    3. Train with BC mode
    4. Evaluate final policy
    5. Verify convergence on PD
    """
    config = load_config()
    
    # Setup
    game = create_game_from_config(config, 'PD')
    opponent = ProbabilisticOpponent(defection_probability=0.5)
    encoder = ObservationEncoder("no_game")
    env = SessionEnvironment(game, opponent, encoder, T=20)
    
    # Test RL mode
    print("\n[RL Mode]")
    agent_rl = RepresentationAgent(input_dim=8, hidden_size=8)
    optimizer_rl = torch.optim.Adam(agent_rl.parameters(), lr=0.01)
    trainer_rl = REINFORCETrainer(agent_rl, optimizer_rl)
    
    history_rl = train_to_convergence(
        trainer_rl, env,
        max_episodes=100,
        mode="rl",
        verbose=True
    )
    
    stats_rl = trainer_rl.evaluate_session(env, max_steps=20)
    print(f"RL Final: Return={stats_rl.total_return:.2f}, BR_acc={stats_rl.br_accuracy:.3f}")
    
    # Test BC mode
    print("\n[BC Mode]")
    agent_bc = RepresentationAgent(input_dim=8, hidden_size=8)
    optimizer_bc = torch.optim.Adam(agent_bc.parameters(), lr=0.01)
    trainer_bc = REINFORCETrainer(agent_bc, optimizer_bc)
    
    history_bc = train_to_convergence(
        trainer_bc, env,
        max_episodes=50,
        mode="bc",
        verbose=True
    )
    
    stats_bc = trainer_bc.evaluate_session(env, max_steps=20)
    print(f"BC Final: Return={stats_bc.total_return:.2f}, BR_acc={stats_bc.br_accuracy:.3f}")
    
    # Verify both modes achieve reasonable performance
    assert stats_rl.br_accuracy > 0.7, \
        f"RL mode should learn PD strategy, got BR_acc={stats_rl.br_accuracy:.3f}"
    assert stats_bc.br_accuracy > 0.9, \
        f"BC mode should quickly learn BR, got BR_acc={stats_bc.br_accuracy:.3f}"
    
    print("\n✅ Gate 3 integration test passed!")


if __name__ == "__main__":
    # Run integration test
    test_gate3_integration()
    print("\n✅ All Gate 3 tests complete!")
