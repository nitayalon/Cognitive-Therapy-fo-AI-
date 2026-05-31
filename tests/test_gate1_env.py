"""
Tests for Gate 1: env (games, encoding, best response, opponents).

These tests must pass before proceeding to Gate 2.

Test coverage:
1. Payoff matrices match config and satisfy game-defining inequalities
2. encode_observation correctness: shapes, one-hot validity, start token
3. Memoryless opponent correctness and history-independence
4. analytic_best_response correctness on hand-checkable cases
5. PD defection dominance verification
"""

import pytest
import numpy as np
import yaml
import random
from pathlib import Path

# Import modules under test
from cognitive_therapy_ai.encoding import ObservationEncoder, Action, Outcome
from cognitive_therapy_ai.best_response import (
    analytic_best_response,
    analytic_best_response_value,
    verify_pd_dominance,
    verify_no_dominance,
    generate_best_response_table
)
from cognitive_therapy_ai.opponent import ProbabilisticOpponent


# ============================================================================
# Test 1: Payoff Matrix Inequality Validation
# ============================================================================

class TestPayoffMatrices:
    """Test that payoff matrices satisfy game-defining inequalities."""
    
    @pytest.fixture
    def config(self):
        """Load base config with payoff matrices."""
        config_path = Path(__file__).parent.parent / "config" / "base.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def test_pd_inequality(self, config):
        """Test Prisoner's Dilemma: T > R > P > S."""
        pd = config['games']['PD']
        payoff = np.array(pd['payoff'], dtype=np.float32)
        
        # Extract values: [[R, S], [T, P]]
        R = payoff[0, 0]
        S = payoff[0, 1]
        T = payoff[1, 0]
        P = payoff[1, 1]
        
        # Check T > R > P > S
        assert T > R, f"PD: T({T}) must be > R({R})"
        assert R > P, f"PD: R({R}) must be > P({P})"
        assert P > S, f"PD: P({P}) must be > S({S})"
        
        # Additional PD condition: 2R > T + S (prevents alternating defection)
        assert 2*R > T + S, f"PD: 2R({2*R}) must be > T+S({T+S})"
    
    def test_sh_inequality(self, config):
        """Test Stag Hunt: R > T >= P > S."""
        sh = config['games']['SH']
        payoff = np.array(sh['payoff'], dtype=np.float32)
        
        # Extract values: [[R, S], [T, P]]
        R = payoff[0, 0]
        S = payoff[0, 1]
        T = payoff[1, 0]
        P = payoff[1, 1]
        
        # Check R > T >= P > S
        assert R > T, f"SH: R({R}) must be > T({T})"
        assert T >= P, f"SH: T({T}) must be >= P({P})"
        assert P > S, f"SH: P({P}) must be > S({S})"
    
    def test_hd_inequality(self, config):
        """Test Hawk-Dove: T > R > S > P."""
        hd = config['games']['HD']
        payoff = np.array(hd['payoff'], dtype=np.float32)
        
        # Extract values: [[R, S], [T, P]]
        R = payoff[0, 0]
        S = payoff[0, 1]
        T = payoff[1, 0]
        P = payoff[1, 1]
        
        # Check T > R > S > P
        assert T > R, f"HD: T({T}) must be > R({R})"
        assert R > S, f"HD: R({R}) must be > S({S})"
        assert S > P, f"HD: S({S}) must be > P({P})"


# ============================================================================
# Test 2: Observation Encoding
# ============================================================================

class TestObservationEncoding:
    """Test encode_observation correctness."""
    
    def test_no_game_dimension(self):
        """Test no_game input dimension is 8."""
        encoder = ObservationEncoder(input_condition="no_game")
        assert encoder.get_input_dim() == 8
    
    def test_game_tag_dimension(self):
        """Test game_tag input dimension is 11 (8 + 3)."""
        encoder = ObservationEncoder(input_condition="game_tag")
        assert encoder.get_input_dim() == 11
    
    def test_start_token_no_game(self):
        """Test start token is all zeros for no_game."""
        encoder = ObservationEncoder(input_condition="no_game")
        start = encoder.encode_start_token()
        
        assert len(start) == 8
        assert np.all(start == 0)
    
    def test_start_token_game_tag(self):
        """Test start token has only game tag for game_tag condition."""
        encoder = ObservationEncoder(input_condition="game_tag")
        start = encoder.encode_start_token(game_name="PD")
        
        assert len(start) == 11
        # First 8 (history) should be zero
        assert np.all(start[:8] == 0)
        # Game tag should be one-hot
        assert np.sum(start[8:11]) == 1
        assert start[8] == 1  # PD is index 0
    
    def test_one_hot_structure_cc(self):
        """Test CC outcome encoding is valid one-hot."""
        encoder = ObservationEncoder(input_condition="no_game")
        encoding = encoder.encode_observation(
            agent_last_action=Action.COOPERATE,
            opponent_last_action=Action.COOPERATE
        )
        
        # Should be 8D
        assert len(encoding) == 8
        
        # Should have exactly 3 active bits
        assert np.sum(encoding) == 3
        
        # Agent cooperate: position 0
        assert encoding[0] == 1
        # Opponent cooperate: position 2
        assert encoding[2] == 1
        # Outcome CC: position 4
        assert encoding[4] == 1
        
        # Validate one-hot structure
        assert encoder.is_one_hot_valid(encoding)
    
    def test_one_hot_structure_dd(self):
        """Test DD outcome encoding is valid one-hot."""
        encoder = ObservationEncoder(input_condition="no_game")
        encoding = encoder.encode_observation(
            agent_last_action=Action.DEFECT,
            opponent_last_action=Action.DEFECT
        )
        
        # Agent defect: position 1
        assert encoding[1] == 1
        # Opponent defect: position 3
        assert encoding[3] == 1
        # Outcome DD: position 7
        assert encoding[7] == 1
        
        assert encoder.is_one_hot_valid(encoding)
    
    def test_one_hot_structure_cd(self):
        """Test CD outcome encoding."""
        encoder = ObservationEncoder(input_condition="no_game")
        encoding = encoder.encode_observation(
            agent_last_action=Action.COOPERATE,
            opponent_last_action=Action.DEFECT
        )
        
        # Agent cooperate: position 0
        assert encoding[0] == 1
        # Opponent defect: position 3
        assert encoding[3] == 1
        # Outcome CD: position 5
        assert encoding[5] == 1
        
        assert encoder.is_one_hot_valid(encoding)
    
    def test_one_hot_structure_dc(self):
        """Test DC outcome encoding."""
        encoder = ObservationEncoder(input_condition="no_game")
        encoding = encoder.encode_observation(
            agent_last_action=Action.DEFECT,
            opponent_last_action=Action.COOPERATE
        )
        
        # Agent defect: position 1
        assert encoding[1] == 1
        # Opponent cooperate: position 2
        assert encoding[2] == 1
        # Outcome DC: position 6
        assert encoding[6] == 1
        
        assert encoder.is_one_hot_valid(encoding)
    
    def test_game_tag_encoding(self):
        """Test game tag is correctly appended."""
        encoder = ObservationEncoder(input_condition="game_tag")
        
        # Test all three games
        for game_name, expected_idx in [("PD", 0), ("SH", 1), ("HD", 2)]:
            encoding = encoder.encode_observation(
                agent_last_action=Action.COOPERATE,
                opponent_last_action=Action.COOPERATE,
                game_name=game_name
            )
            
            # Check game tag (positions 8-10)
            game_tag = encoding[8:11]
            assert np.sum(game_tag) == 1
            assert game_tag[expected_idx] == 1
    
    def test_include_timestep_changes_dimension(self):
        """Test that include_timestep=True adds 1 dimension."""
        encoder = ObservationEncoder(input_condition="no_game", include_timestep=True)
        assert encoder.get_input_dim() == 9  # 8 + 1
        
        encoder_gt = ObservationEncoder(input_condition="game_tag", include_timestep=True)
        assert encoder_gt.get_input_dim() == 12  # 8 + 3 + 1
    
    def test_include_reward_changes_dimension(self):
        """Test that include_reward=True adds 2 dimensions."""
        encoder = ObservationEncoder(input_condition="no_game", include_reward=True)
        assert encoder.get_input_dim() == 10  # 8 + 2
    
    def test_alphabet_enumeration_no_game(self):
        """Test alphabet is correctly enumerated for no_game."""
        encoder = ObservationEncoder(input_condition="no_game")
        alphabet = encoder.get_alphabet()
        
        assert alphabet == ["START", "CC", "CD", "DC", "DD"]
    
    def test_alphabet_enumeration_game_tag(self):
        """Test alphabet is correctly enumerated for game_tag."""
        encoder = ObservationEncoder(input_condition="game_tag")
        alphabet = encoder.get_alphabet()
        
        # Should be 5 symbols × 3 games = 15
        assert len(alphabet) == 15
        
        # Check structure
        expected_start = ["PD_START", "PD_CC", "PD_CD", "PD_DC", "PD_DD"]
        assert alphabet[:5] == expected_start


# ============================================================================
# Test 3: Memoryless Opponent
# ============================================================================

class TestMemorylessOpponent:
    """Test that probabilistic opponent is truly memoryless."""
    
    def test_cooperation_rate_matches_p_coop(self):
        """Test empirical cooperation rate matches p_coop over many samples."""
        p_coop = 0.7
        opponent = ProbabilisticOpponent(defection_probability=1.0 - p_coop)
        
        # Sample many actions
        n_samples = 10000
        actions = []
        for _ in range(n_samples):
            action = opponent.choose_action(game_history=[], round_number=0)
            actions.append(action)
        
        # Count cooperations
        n_coop = sum(1 for a in actions if a == Action.COOPERATE)
        empirical_p_coop = n_coop / n_samples
        
        # Should be close to p_coop (within 3 std devs: ~0.015 for n=10000, p=0.7)
        assert abs(empirical_p_coop - p_coop) < 0.02, \
            f"Empirical p_coop ({empirical_p_coop:.3f}) != expected ({p_coop})"
    
    def test_history_independence(self):
        """Test that opponent action is independent of game history."""
        p_coop = 0.5
        opponent = ProbabilisticOpponent(defection_probability=0.5)
        
        # Create two different histories
        history1 = []  # Empty
        history2 = [  # Full of defections
            {'player_action': Action.DEFECT, 'opponent_action': Action.DEFECT}
            for _ in range(100)
        ]
        
        # Sample actions with both histories
        n_samples = 5000
        random.seed(42)
        actions1 = [opponent.choose_action(history1, i) for i in range(n_samples)]
        
        random.seed(42)  # Reset seed to get same random sequence
        opponent.reset()
        actions2 = [opponent.choose_action(history2, i) for i in range(n_samples)]
        
        # Actions should be identical (same random seed → same sequence)
        assert actions1 == actions2, "Opponent is not history-independent!"
    
    def test_always_defect(self):
        """Test always-defect opponent (p_coop=0.0)."""
        opponent = ProbabilisticOpponent(defection_probability=1.0)
        
        # All actions should be defect
        for _ in range(100):
            action = opponent.choose_action(game_history=[], round_number=0)
            assert action == Action.DEFECT
    
    def test_always_cooperate(self):
        """Test always-cooperate opponent (p_coop=1.0)."""
        opponent = ProbabilisticOpponent(defection_probability=0.0)
        
        # All actions should be cooperate
        for _ in range(100):
            action = opponent.choose_action(game_history=[], round_number=0)
            assert action == Action.COOPERATE


# ============================================================================
# Test 4: Analytic Best Response
# ============================================================================

class TestAnalyticBestResponse:
    """Test analytic best response calculations."""
    
    @pytest.fixture
    def config(self):
        """Load config for payoff matrices."""
        config_path = Path(__file__).parent.parent / "config" / "base.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def test_pd_always_defect(self, config):
        """Test PD: defection is best response for all p_coop."""
        payoff = np.array(config['games']['PD']['payoff'], dtype=np.float32)
        
        # Test at several p_coop values
        for p_coop in [0.0, 0.1, 0.5, 0.9, 1.0]:
            br = analytic_best_response(payoff, p_coop)
            assert br == Action.DEFECT, \
                f"PD: Best response should be DEFECT at p_coop={p_coop}, got {br}"
    
    def test_pd_dominance_verification(self, config):
        """Test that verify_pd_dominance correctly identifies PD."""
        payoff = np.array(config['games']['PD']['payoff'], dtype=np.float32)
        assert verify_pd_dominance(payoff)
    
    def test_sh_no_dominance(self, config):
        """Test SH: no dominant strategy (best response varies)."""
        payoff = np.array(config['games']['SH']['payoff'], dtype=np.float32)
        assert verify_no_dominance(payoff)
    
    def test_hd_no_dominance(self, config):
        """Test HD: no dominant strategy."""
        payoff = np.array(config['games']['HD']['payoff'], dtype=np.float32)
        assert verify_no_dominance(payoff)
    
    def test_best_response_value_matches_payoff(self):
        """Test that best response value equals expected payoff under BR."""
        # Simple test matrix
        payoff = np.array([[3, 0], [5, 1]], dtype=np.float32)  # PD
        p_coop = 0.5
        
        br = analytic_best_response(payoff, p_coop)
        value = analytic_best_response_value(payoff, p_coop)
        
        # Manually calculate expected value under BR
        if br == Action.COOPERATE:
            expected = p_coop * payoff[0, 0] + (1 - p_coop) * payoff[0, 1]
        else:
            expected = p_coop * payoff[1, 0] + (1 - p_coop) * payoff[1, 1]
        
        assert abs(value - expected) < 1e-6
    
    def test_hand_checkable_case_always_defect(self):
        """Test BR against always-defect opponent (p_coop=0.0)."""
        # In PD against always-defect, best response is defect
        payoff = np.array([[3, 0], [5, 1]], dtype=np.float32)  # [[R,S], [T,P]]
        br = analytic_best_response(payoff, p_coop=0.0)
        assert br == Action.DEFECT
        
        # Expected value = P = 1
        value = analytic_best_response_value(payoff, p_coop=0.0)
        assert abs(value - 1.0) < 1e-6
    
    def test_hand_checkable_case_always_cooperate(self):
        """Test BR against always-cooperate opponent (p_coop=1.0)."""
        # In PD against always-cooperate, best response is defect (exploit!)
        payoff = np.array([[3, 0], [5, 1]], dtype=np.float32)  # [[R,S], [T,P]]
        br = analytic_best_response(payoff, p_coop=1.0)
        assert br == Action.DEFECT
        
        # Expected value = T = 5
        value = analytic_best_response_value(payoff, p_coop=1.0)
        assert abs(value - 5.0) < 1e-6


# ============================================================================
# Integration Test
# ============================================================================

def test_gate1_integration():
    """
    Integration test: all Gate 1 components work together.
    
    This test simulates a minimal environment setup:
    1. Load config
    2. Create encoder
    3. Create opponent
    4. Compute best response
    5. Encode observations
    """
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "base.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create encoder
    encoder = ObservationEncoder(input_condition="no_game")
    
    # Create opponent
    opponent = ProbabilisticOpponent(defection_probability=0.3)  # p_coop = 0.7
    
    # Get PD payoff matrix
    payoff = np.array(config['games']['PD']['payoff'], dtype=np.float32)
    
    # Compute best response
    br = analytic_best_response(payoff, p_coop=0.7)
    assert br == Action.DEFECT  # PD: always defect
    
    # Encode start token
    obs_t0 = encoder.encode_start_token()
    assert len(obs_t0) == 8
    assert np.all(obs_t0 == 0)
    
    # Simulate one round
    agent_action = br
    opponent_action = opponent.choose_action([], 0)
    
    # Encode observation at t=1
    obs_t1 = encoder.encode_observation(
        agent_last_action=agent_action,
        opponent_last_action=opponent_action
    )
    assert encoder.is_one_hot_valid(obs_t1)
    
    print("✓ Gate 1 integration test passed!")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
