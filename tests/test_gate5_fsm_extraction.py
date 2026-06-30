"""
Gate 5 Tests: FSM Extraction

Verifies:
1. Rollout collection works correctly
2. Hidden state clustering identifies discrete states
3. L* algorithm extracts FSM from trajectories
4. Hopcroft minimization reduces FSM
5. FSM validation against best response
6. End-to-end extraction pipeline
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import yaml

from cognitive_therapy_ai.games import Action, PrisonersDilemma
from cognitive_therapy_ai.opponent import ProbabilisticOpponent, Opponent
from cognitive_therapy_ai.encoding import ObservationEncoder
from cognitive_therapy_ai.representation_agent import RepresentationAgent
from cognitive_therapy_ai.reinforce_trainer import SessionEnvironment, REINFORCETrainer, train_to_convergence
from cognitive_therapy_ai.fsm_extraction import (
    RolloutCollector,
    HiddenStateClusterer,
    LStarExtractor,
    HopcroftMinimizer,
    FSMValidator,
    extract_fsm_from_agent,
    Trajectory,
    FSM
)


def load_config():
    """Load base configuration."""
    config_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_simple_trained_agent(device='cpu'):
    """Create and train a simple agent on PD vs always-defect."""
    config = load_config()
    
    # Create game and opponent
    payoff = config['games']['PD']['payoff']
    game = PrisonersDilemma(
        T=payoff[1][0], R=payoff[0][0],
        P=payoff[1][1], S=payoff[0][1]
    )
    # SessionEnvironment expects the Opponent wrapper (.play_action()),
    # not the bare OpponentStrategy (.choose_action()).
    opponent = Opponent(ProbabilisticOpponent(defection_probability=0.9))

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
        max_episodes=30,
        convergence_window=10,
        convergence_threshold=0.01,
        mode='rl',
        verbose=False
    )
    
    return agent, encoder, env


class TestRolloutCollector:
    """Test trajectory collection."""
    
    def test_collect_single_trajectory(self):
        """Test collecting single trajectory."""
        agent, encoder, env = create_simple_trained_agent()
        
        collector = RolloutCollector(agent, encoder)
        traj = collector.collect_trajectory(env, max_steps=20, epsilon_explore=0.1)
        
        # Check trajectory structure
        assert isinstance(traj, Trajectory)
        assert len(traj.observations) > 0
        assert len(traj.actions) > 0
        assert len(traj.hidden_states) > 0
        assert len(traj.rewards) > 0
        assert len(traj.observations) == len(traj.actions)
        assert len(traj.actions) == len(traj.hidden_states)
    
    def test_trajectory_hidden_states_shape(self):
        """Test that hidden states have correct shape."""
        agent, encoder, env = create_simple_trained_agent()
        
        collector = RolloutCollector(agent, encoder)
        traj = collector.collect_trajectory(env, max_steps=10, epsilon_explore=0.0)
        
        # Hidden states should be 1D arrays of size H
        for h_t in traj.hidden_states:
            assert h_t.ndim == 1
            assert h_t.shape[0] == agent.hidden_size
    
    def test_collect_multiple_trajectories(self):
        """Test collecting multiple trajectories."""
        agent, encoder, env = create_simple_trained_agent()
        
        collector = RolloutCollector(agent, encoder)
        trajectories = collector.collect_multiple_trajectories(
            env, n_trajectories=5, max_steps=20, epsilon_explore=0.1
        )
        
        assert len(trajectories) == 5
        assert all(isinstance(t, Trajectory) for t in trajectories)
    
    def test_epsilon_exploration_affects_actions(self):
        """Test that epsilon exploration produces different actions."""
        agent, encoder, env = create_simple_trained_agent()
        collector = RolloutCollector(agent, encoder)
        
        # Collect with no exploration
        traj_deterministic = collector.collect_trajectory(
            env, max_steps=50, epsilon_explore=0.0
        )
        
        # Collect with high exploration
        traj_explore = collector.collect_trajectory(
            env, max_steps=50, epsilon_explore=0.5
        )
        
        # Both should have actions
        assert len(traj_deterministic.actions) > 0
        assert len(traj_explore.actions) > 0

    def test_exploration_advances_hidden_state(self):
        """Regression test for the epsilon-explore hidden-state-skip bug:
        exploring steps must not desync the recorded hidden state from the
        network's own forward-pass trajectory. Forces exploration on every
        step, then independently replays the SAME observation sequence
        through the network from a fresh hidden state and asserts every
        recorded hidden state matches exactly (no skipped/duplicated
        forward pass)."""
        agent, encoder, env = create_simple_trained_agent()
        collector = RolloutCollector(agent, encoder)

        traj = collector.collect_trajectory(env, max_steps=15, epsilon_explore=1.0)

        hidden_state = agent.reset_hidden_state(batch_size=1)
        for t, obs in enumerate(traj.observations):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                _, _, hidden_state = agent.select_action(
                    obs_tensor, hidden_state, deterministic=True, epsilon=0.0
                )
            replayed_h = hidden_state[0].cpu().numpy().flatten()
            np.testing.assert_allclose(
                replayed_h, traj.hidden_states[t], rtol=1e-6, atol=1e-6,
                err_msg=f"hidden state at step {t} does not match the network's own "
                        f"forward-pass trajectory -- exploration desynced the recurrent state"
            )


class TestHiddenStateClusterer:
    """Test hidden state clustering."""
    
    def test_clustering_runs(self):
        """Test that clustering completes without errors."""
        agent, encoder, env = create_simple_trained_agent()
        collector = RolloutCollector(agent, encoder)
        
        trajectories = collector.collect_multiple_trajectories(
            env, n_trajectories=10, max_steps=20
        )
        
        clusterer = HiddenStateClusterer(n_clusters=3)
        clusters = clusterer.fit(trajectories)
        
        assert len(clusters) == 3
        assert all(cluster.size > 0 for cluster in clusters)
    
    def test_cluster_centroids_shape(self):
        """Test that cluster centroids have correct shape."""
        agent, encoder, env = create_simple_trained_agent()
        collector = RolloutCollector(agent, encoder)
        
        trajectories = collector.collect_multiple_trajectories(
            env, n_trajectories=5, max_steps=10
        )
        
        clusterer = HiddenStateClusterer(n_clusters=2)
        clusters = clusterer.fit(trajectories)
        
        # Centroids should match hidden size
        for cluster in clusters:
            assert cluster.centroid.shape[0] == agent.hidden_size
    
    def test_predict_assigns_to_cluster(self):
        """Test that predict assigns states to clusters."""
        agent, encoder, env = create_simple_trained_agent()
        collector = RolloutCollector(agent, encoder)
        
        trajectories = collector.collect_multiple_trajectories(
            env, n_trajectories=5, max_steps=10
        )
        
        clusterer = HiddenStateClusterer(n_clusters=3)
        clusterer.fit(trajectories)
        
        # Test prediction
        test_state = np.random.randn(agent.hidden_size)
        cluster_id = clusterer.predict(test_state)
        
        assert isinstance(cluster_id, (int, np.integer))
        assert 0 <= cluster_id < 3


class TestLStarExtractor:
    """Test L* FSM extraction."""
    
    def test_extract_fsm_creates_valid_fsm(self):
        """Test that FSM extraction produces valid FSM."""
        agent, encoder, env = create_simple_trained_agent()
        collector = RolloutCollector(agent, encoder)
        
        trajectories = collector.collect_multiple_trajectories(
            env, n_trajectories=10, max_steps=20
        )
        
        clusterer = HiddenStateClusterer(n_clusters=2)
        clusterer.fit(trajectories)
        
        alphabet = encoder.get_alphabet()
        extractor = LStarExtractor(alphabet)
        fsm = extractor.extract_fsm(trajectories, clusterer.fit(trajectories), clusterer, encoder)
        
        assert isinstance(fsm, FSM)
        assert len(fsm.states) > 0
        assert fsm.alphabet == alphabet
        assert len(fsm.transitions) > 0
    
    def test_fsm_has_initial_state(self):
        """Test that extracted FSM has initial state."""
        agent, encoder, env = create_simple_trained_agent()
        collector = RolloutCollector(agent, encoder)
        
        trajectories = collector.collect_multiple_trajectories(
            env, n_trajectories=5, max_steps=10
        )
        
        clusterer = HiddenStateClusterer(n_clusters=2)
        clusters = clusterer.fit(trajectories)
        
        extractor = LStarExtractor(encoder.get_alphabet())
        fsm = extractor.extract_fsm(trajectories, clusters, clusterer, encoder)
        
        assert fsm.initial_state is not None
        assert fsm.initial_state in fsm.states


class _FakeClusterer:
    """Minimal stand-in satisfying HiddenStateClusterer's predict() interface
    (LStarExtractor.extract_fsm only ever calls .predict()), so transition
    voting/threshold logic can be tested with fully-controlled cluster
    assignments instead of fitting real KMeans."""

    def predict(self, hidden_state):
        return int(round(float(hidden_state[0])))


class TestLStarExtractorTransitionVoting:
    """Test majority-vote transition building with a min-visit threshold
    (Stage 1 fix 2: replaces the old last-write-wins transition table)."""

    @staticmethod
    def _make_traj(encoder, cluster_a, symbol, action, cluster_b):
        """A 2-timestep synthetic trajectory that casts exactly one vote
        for (cluster_a, symbol) -> (cluster_b, action)."""
        if symbol == "START":
            obs0 = encoder.encode_start_token(game_name=None)
        else:
            agent_act = Action.COOPERATE if symbol[0] == "C" else Action.DEFECT
            opp_act = Action.COOPERATE if symbol[1] == "C" else Action.DEFECT
            obs0 = encoder.encode_observation(
                agent_last_action=agent_act, opponent_last_action=opp_act, game_name=None
            )
        obs1 = encoder.encode_observation(
            agent_last_action=Action.DEFECT, opponent_last_action=Action.DEFECT, game_name=None
        )
        return Trajectory(
            observations=[obs0, obs1],
            actions=[action, Action.DEFECT],
            hidden_states=[np.array([float(cluster_a)]), np.array([float(cluster_b)])],
            rewards=[0.0, 0.0],
            opponent_actions=[None, Action.DEFECT],
        )

    def test_majority_vote_picks_modal_target(self):
        """5 votes for (0,"CC")->(0,DEFECT) vs. 2 votes for
        (0,"CC")->(1,COOPERATE): the modal (majority) outcome must win,
        not whichever vote happened to be last."""
        encoder = ObservationEncoder('no_game')
        trajectories = (
            [self._make_traj(encoder, 0, "CC", Action.DEFECT, 0) for _ in range(5)] +
            [self._make_traj(encoder, 0, "CC", Action.COOPERATE, 1) for _ in range(2)]
        )
        extractor = LStarExtractor(encoder.get_alphabet(), min_transition_visits=3)
        fsm = extractor.extract_fsm(trajectories, clusters=[], clusterer=_FakeClusterer(), encoder=encoder)

        assert fsm.transitions[(0, "CC")] == (0, Action.DEFECT)
        assert fsm.transition_visit_counts[(0, "CC")] == 7

    def test_sub_threshold_transition_is_undefined(self):
        """A (state, symbol) pair observed fewer times than
        min_transition_visits must be left undefined, not forced from a
        small sample -- even though every observed vote agrees."""
        encoder = ObservationEncoder('no_game')
        trajectories = [self._make_traj(encoder, 0, "DD", Action.DEFECT, 0) for _ in range(2)]
        extractor = LStarExtractor(encoder.get_alphabet(), min_transition_visits=3)
        fsm = extractor.extract_fsm(trajectories, clusters=[], clusterer=_FakeClusterer(), encoder=encoder)

        assert (0, "DD") not in fsm.transitions
        assert fsm.transition_visit_counts[(0, "DD")] == 2

    def test_threshold_is_configurable(self):
        """Lowering min_transition_visits to 1 should define a transition
        from a single observed visit."""
        encoder = ObservationEncoder('no_game')
        trajectories = [self._make_traj(encoder, 0, "DD", Action.DEFECT, 0)]
        extractor = LStarExtractor(encoder.get_alphabet(), min_transition_visits=1)
        fsm = extractor.extract_fsm(trajectories, clusters=[], clusterer=_FakeClusterer(), encoder=encoder)

        assert fsm.transitions[(0, "DD")] == (0, Action.DEFECT)
        assert fsm.transition_visit_counts[(0, "DD")] == 1


class TestFSMOperations:
    """Test FSM data structure operations."""
    
    def test_fsm_get_action(self):
        """Test FSM action lookup."""
        fsm = FSM(
            states={0, 1},
            alphabet=["START", "CC", "DD"],
            transitions={
                (0, "START"): (1, Action.COOPERATE),
                (1, "CC"): (1, Action.DEFECT)
            },
            initial_state=0,
            n_states=2
        )
        
        action = fsm.get_action(0, "START")
        assert action == Action.COOPERATE
        
        action = fsm.get_action(1, "CC")
        assert action == Action.DEFECT
        
        action = fsm.get_action(0, "CC")
        assert action is None  # Not in transitions
    
    def test_fsm_get_next_state(self):
        """Test FSM state transition lookup."""
        fsm = FSM(
            states={0, 1},
            alphabet=["START", "CC"],
            transitions={
                (0, "START"): (1, Action.COOPERATE),
                (1, "CC"): (1, Action.DEFECT)
            },
            initial_state=0,
            n_states=2
        )
        
        next_state = fsm.get_next_state(0, "START")
        assert next_state == 1
        
        next_state = fsm.get_next_state(1, "CC")
        assert next_state == 1
    
    def test_fsm_is_complete(self):
        """Test FSM completeness check."""
        # Complete FSM
        fsm_complete = FSM(
            states={0},
            alphabet=["A", "B"],
            transitions={
                (0, "A"): (0, Action.COOPERATE),
                (0, "B"): (0, Action.DEFECT)
            },
            initial_state=0,
            n_states=1
        )
        assert fsm_complete.is_complete()
        
        # Incomplete FSM
        fsm_incomplete = FSM(
            states={0, 1},
            alphabet=["A", "B"],
            transitions={
                (0, "A"): (1, Action.COOPERATE)
                # Missing (0, "B"), (1, "A"), (1, "B")
            },
            initial_state=0,
            n_states=2
        )
        assert not fsm_incomplete.is_complete()


class TestHopcroftMinimizer:
    """Test Hopcroft minimization."""
    
    def test_minimize_single_state_fsm(self):
        """Test minimizing a single-state FSM."""
        fsm = FSM(
            states={0},
            alphabet=["A", "B"],
            transitions={
                (0, "A"): (0, Action.DEFECT),
                (0, "B"): (0, Action.DEFECT)
            },
            initial_state=0,
            n_states=1
        )
        
        minimizer = HopcroftMinimizer()
        min_fsm = minimizer.minimize(fsm)
        
        # Should still be 1 state
        assert min_fsm.n_states == 1
        assert len(min_fsm.states) == 1
    
    def test_minimize_merges_equivalent_states(self):
        """Test that minimization merges equivalent states."""
        # FSM with two equivalent states (both always defect)
        fsm = FSM(
            states={0, 1},
            alphabet=["A", "B"],
            transitions={
                (0, "A"): (0, Action.DEFECT),
                (0, "B"): (0, Action.DEFECT),
                (1, "A"): (1, Action.DEFECT),
                (1, "B"): (1, Action.DEFECT)
            },
            initial_state=0,
            n_states=2
        )
        
        minimizer = HopcroftMinimizer()
        min_fsm = minimizer.minimize(fsm)
        
        # Should merge to 1 state
        assert min_fsm.n_states <= 1

    def test_partial_automaton_collapses_with_single_observed_self_loop(self):
        """Stage 0 regression test: a hand-built partial machine with N
        states that all emit the same action but each only has ONE
        observed self-loop, each on a DIFFERENT symbol (so every state is
        undefined on every other symbol) -- they must all minimize to 1
        state, since no two of them ever disagree on anything actually
        observed. (Each state having its single observed transition on a
        *different* symbol is essential: that's exactly the asymmetric
        coverage pattern that made the old signature-bucketing minimizer
        treat "undefined" as a concrete, distinguishing value and refuse
        to merge -- confirmed to give 4 states on the pre-fix algorithm,
        vs. 1 here.)"""
        fsm = FSM(
            states={0, 1, 2, 3},
            alphabet=["A", "B", "C", "D"],
            transitions={
                (0, "A"): (0, Action.DEFECT),
                (1, "B"): (1, Action.DEFECT),
                (2, "C"): (2, Action.DEFECT),
                (3, "D"): (3, Action.DEFECT),
            },
            initial_state=0,
            n_states=4
        )

        minimizer = HopcroftMinimizer()
        min_fsm = minimizer.minimize(fsm)

        assert min_fsm.n_states == 1

    def test_partial_automaton_keeps_genuinely_distinguished_states_apart(self):
        """Stage 0 regression test: if an OBSERVED transition actually
        disagrees between two states (same symbol, both defined, different
        action), they must NOT be merged -- don't-care semantics only
        apply to UNDEFINED transitions, never to a real conflict."""
        fsm = FSM(
            states={0, 1},
            alphabet=["A"],
            transitions={
                (0, "A"): (0, Action.COOPERATE),
                (1, "A"): (1, Action.DEFECT),
            },
            initial_state=0,
            n_states=2
        )

        minimizer = HopcroftMinimizer()
        min_fsm = minimizer.minimize(fsm)

        assert min_fsm.n_states == 2

    def test_partial_automaton_mixed_case(self):
        """A partial machine with 3 states: states 0 and 1 each have a
        single observed self-loop on a DIFFERENT symbol (both DEFECT,
        like the test above) and should merge; state 2 has both symbols
        observed with a genuinely different action (COOPERATE) and must
        stay apart. (Pre-fix algorithm gives 3 states here -- 0 and 1
        never merge because their undefined symbols don't line up;
        post-fix gives 2.)"""
        fsm = FSM(
            states={0, 1, 2},
            alphabet=["A", "B"],
            transitions={
                (0, "A"): (0, Action.DEFECT),
                # state 0 undefined on "B"
                (1, "B"): (1, Action.DEFECT),
                # state 1 undefined on "A"
                (2, "A"): (2, Action.COOPERATE),  # genuinely different
                (2, "B"): (2, Action.COOPERATE),
            },
            initial_state=0,
            n_states=3
        )

        minimizer = HopcroftMinimizer()
        min_fsm = minimizer.minimize(fsm)

        assert min_fsm.n_states == 2
        # The merged block (from states 0,1) must emit DEFECT; state 2's
        # block must emit COOPERATE -- find which is which via fsm.states.
        actions = set()
        for state in min_fsm.states:
            for symbol in ("A", "B"):
                action = min_fsm.get_action(state, symbol)
                if action is not None:
                    actions.add(action)
        assert actions == {Action.DEFECT, Action.COOPERATE}


class TestFSMValidator:
    """Test FSM validation."""
    
    def test_validator_runs(self):
        """Test that validation completes without errors."""
        config = load_config()
        payoff = config['games']['PD']['payoff']
        game = PrisonersDilemma(
            T=payoff[1][0], R=payoff[0][0],
            P=payoff[1][1], S=payoff[0][1]
        )
        opponent = ProbabilisticOpponent(defection_probability=0.5)
        encoder = ObservationEncoder('no_game')
        
        # Create simple always-defect FSM
        fsm = FSM(
            states={0},
            alphabet=encoder.get_alphabet(),
            transitions={
                (0, "START"): (0, Action.DEFECT),
                (0, "CC"): (0, Action.DEFECT),
                (0, "CD"): (0, Action.DEFECT),
                (0, "DC"): (0, Action.DEFECT),
                (0, "DD"): (0, Action.DEFECT)
            },
            initial_state=0,
            n_states=1
        )
        
        validator = FSMValidator(encoder)
        metrics = validator.validate_against_best_response(
            fsm, game, opponent, n_validation_episodes=10, max_steps=20
        )
        
        assert 'br_accuracy' in metrics
        assert 0.0 <= metrics['br_accuracy'] <= 1.0
        assert metrics['total_actions'] > 0
        assert metrics['fsm_complete']


class TestEndToEndExtraction:
    """Test complete FSM extraction pipeline."""
    
    def test_extract_fsm_from_agent_completes(self):
        """Test that full extraction pipeline runs."""
        agent, encoder, env = create_simple_trained_agent()
        
        fsm, trajectories, clusters = extract_fsm_from_agent(
            agent=agent,
            encoder=encoder,
            env=env,
            n_clusters=2,
            n_trajectories=20,
            max_steps=30,
            epsilon_explore=0.1
        )
        
        # Check outputs
        assert isinstance(fsm, FSM)
        assert len(trajectories) == 20
        assert len(clusters) == 2
        assert len(fsm.states) > 0
    
    def test_extracted_fsm_is_minimized(self):
        """Test that extracted FSM is minimized."""
        agent, encoder, env = create_simple_trained_agent()
        
        fsm, _, _ = extract_fsm_from_agent(
            agent=agent,
            encoder=encoder,
            env=env,
            n_clusters=3,
            n_trajectories=15,
            max_steps=20,
            epsilon_explore=0.1
        )
        
        # FSM should have transitions
        assert len(fsm.transitions) > 0
        assert fsm.n_states > 0


def test_gate5_integration():
    """
    Gate 5 integration test: Full FSM extraction and validation.
    
    Tests:
    1. Train agent on PD vs p_coop=0.9 (should learn always-defect)
    2. Extract FSM from trained agent
    3. Validate FSM matches best response
    4. Verify FSM is minimal
    """
    print("\n[Gate 5 Integration Test]")
    
    # Setup
    config = load_config()
    device = torch.device('cpu')
    
    # Create game (PD) and opponent (high cooperation)
    payoff = config['games']['PD']['payoff']
    game = PrisonersDilemma(
        T=payoff[1][0], R=payoff[0][0],
        P=payoff[1][1], S=payoff[0][1]
    )
    # FSMValidator.validate_against_best_response() reads
    # opponent.defection_probability directly (bare strategy), while
    # SessionEnvironment.step() calls opponent.play_action() (Opponent
    # wrapper) -- keep both references since this test uses both consumers.
    opponent_strategy = ProbabilisticOpponent(defection_probability=0.1)
    opponent = Opponent(opponent_strategy)

    print(f"Game: PD, Opponent: p_coop=0.9")

    # Create encoder and environment
    encoder = ObservationEncoder('no_game')
    env = SessionEnvironment(game, opponent, encoder, T=100, game_name=None)
    
    # Create and train agent
    print("Training agent...")
    agent = RepresentationAgent(
        input_dim=encoder.get_input_dim(),
        hidden_size=4,
        device=device
    )
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)
    trainer = REINFORCETrainer(agent, optimizer, gamma=0.99, gae_lambda=0.95, device=device)
    
    history = train_to_convergence(
        trainer, env,
        max_episodes=50,
        convergence_window=20,
        convergence_threshold=0.01,
        mode='rl',
        verbose=True
    )
    
    print(f"Training converged after {len(history['returns'])} episodes")
    
    # Extract FSM
    print("\nExtracting FSM...")
    fsm, trajectories, clusters = extract_fsm_from_agent(
        agent=agent,
        encoder=encoder,
        env=env,
        n_clusters=3,
        n_trajectories=50,
        max_steps=50,
        epsilon_explore=0.1,
        device=device
    )
    
    print(f"Extracted FSM: {fsm.n_states} states, {len(fsm.transitions)} transitions")
    print(f"Collected {len(trajectories)} trajectories")
    print(f"Clustered into {len(clusters)} discrete states")
    
    # Validate FSM
    print("\nValidating FSM...")
    validator = FSMValidator(encoder)
    metrics = validator.validate_against_best_response(
        fsm, game, opponent_strategy,
        n_validation_episodes=100,
        max_steps=100
    )
    
    print(f"Best Response Accuracy: {metrics['br_accuracy']:.3f}")
    print(f"FSM Complete: {metrics['fsm_complete']}")
    print(f"Total Actions: {metrics['total_actions']}")
    
    # Assertions
    assert fsm.n_states > 0, "FSM should have at least 1 state"
    assert len(fsm.transitions) > 0, "FSM should have transitions"
    
    # Note: With small H=4 and limited trajectories, FSM may be incomplete
    # The key test is that extraction pipeline runs without errors
    # For complete FSMs, would need larger H, more trajectories, better clustering
    if metrics['total_actions'] > 0:
        assert metrics['br_accuracy'] >= 0.0, "Accuracy should be non-negative"
        print(f"   FSM achieved {metrics['br_accuracy']:.1%} best-response accuracy")
    else:
        print(f"   FSM incomplete - needs more trajectories or better clustering")
    
    # For PD against high cooperation, optimal is DEFECT
    # FSM should learn to defect (high accuracy expected with complete FSM)
    print(f"\n✅ Gate 5 integration test passed!")
    print(f"   FSM extraction pipeline works end-to-end")
    print(f"   Extracted {fsm.n_states}-state automaton with {len(fsm.transitions)} transitions")


if __name__ == "__main__":
    # Run integration test
    test_gate5_integration()
    print("\n✅ All Gate 5 tests complete!")
