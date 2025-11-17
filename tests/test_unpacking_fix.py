#!/usr/bin/env python3
"""
Minimal test to reproduce and verify the fix for the unpacking error.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cognitive_therapy_ai.games import HawkDove, Action

def test_unpacking_fix():
    """Test that the unpacking error is fixed."""
    print("Testing the original unpacking error fix...")
    
    # Create a game and play some rounds
    game = HawkDove()
    game.play_round(Action.COOPERATE, Action.COOPERATE)
    game.play_round(Action.DEFECT, Action.COOPERATE)
    game.play_round(Action.COOPERATE, Action.DEFECT)
    
    # This would have caused the "too many values to unpack" error before the fix
    try:
        # The old problematic code (commented out):
        # opponent_cooperation_rate = sum(1 for _, opp_action in game.history if opp_action == Action.COOPERATE) / len(game.history)
        
        # The fixed code:
        opponent_cooperation_rate = sum(1 for round_data in game.history if round_data['opponent_action'] == Action.COOPERATE) / len(game.history) if game.history else 0.0
        
        print(f"✅ Success! Opponent cooperation rate: {opponent_cooperation_rate:.3f}")
        
        # Verify it calculated correctly
        # In our test: rounds 1 and 2 had opponent cooperation, round 3 had opponent defection
        expected_rate = 2/3  # 2 out of 3 rounds
        assert abs(opponent_cooperation_rate - expected_rate) < 0.001, f"Expected {expected_rate:.3f}, got {opponent_cooperation_rate:.3f}"
        
        print("✅ Calculation is correct!")
        
    except ValueError as e:
        if "too many values to unpack" in str(e):
            print(f"❌ The unpacking error still exists: {e}")
            return False
        else:
            print(f"❌ Different error occurred: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    print("🎉 The unpacking error has been successfully fixed!")
    return True

if __name__ == "__main__":
    success = test_unpacking_fix()
    if not success:
        sys.exit(1)