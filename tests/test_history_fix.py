#!/usr/bin/env python3
"""
Test script to verify the game history access fix.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cognitive_therapy_ai.games import HawkDove, Action

def test_game_history_access():
    """Test that game history can be accessed correctly."""
    print("Testing game history access...")
    
    # Create a game
    game = HawkDove()
    
    # Play a few rounds to populate history
    game.play_round(Action.COOPERATE, Action.COOPERATE)
    game.play_round(Action.DEFECT, Action.COOPERATE)
    game.play_round(Action.COOPERATE, Action.DEFECT)
    
    print(f"Game history length: {len(game.history)}")
    
    # Test the corrected opponent cooperation rate calculation
    opponent_cooperation_count = sum(1 for round_data in game.history if round_data['opponent_action'] == Action.COOPERATE)
    opponent_cooperation_rate = opponent_cooperation_count / len(game.history) if game.history else 0.0
    
    print(f"Opponent cooperation count: {opponent_cooperation_count}")
    print(f"Opponent cooperation rate: {opponent_cooperation_rate:.3f}")
    
    # Verify against expected values
    expected_count = 2  # First and second rounds had opponent cooperation
    expected_rate = 2/3  # 2 out of 3 rounds
    
    assert opponent_cooperation_count == expected_count, f"Expected {expected_count}, got {opponent_cooperation_count}"
    assert abs(opponent_cooperation_rate - expected_rate) < 0.001, f"Expected {expected_rate:.3f}, got {opponent_cooperation_rate:.3f}"
    
    print("✅ Game history access test passed!")
    
    # Test built-in method for comparison
    player_cooperation_rate = game.get_cooperation_rate()
    print(f"Player cooperation rate (built-in method): {player_cooperation_rate:.3f}")
    
    # Verify the built-in method works correctly
    expected_player_rate = 2/3  # Player cooperated in first and third rounds
    assert abs(player_cooperation_rate - expected_player_rate) < 0.001, f"Built-in method failed: expected {expected_player_rate:.3f}, got {player_cooperation_rate:.3f}"
    
    print("✅ Built-in cooperation rate method works correctly!")
    
    # Show the actual history structure for verification
    print("\nGame history structure:")
    for i, round_data in enumerate(game.history):
        print(f"  Round {i+1}: {round_data}")
    
    print("\n🎉 All game history access tests passed!")

if __name__ == "__main__":
    test_game_history_access()