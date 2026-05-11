#!/usr/bin/env python3
"""
Display the payoff matrices that agents see as input for each game.
This shows why agents behave differently across games - they receive different inputs!
"""

import numpy as np
import pandas as pd

# Game payoff matrices (from game implementations)
GAMES = {
    'Prisoner\'s Dilemma': {
        'abbrev': 'PD',
        'matrix': np.array([
            [3, 0],  # [R, S] - Agent cooperates
            [5, 1]   # [T, P] - Agent defects
        ]),
        'labels': ['R=3', 'S=0', 'T=5', 'P=1']
    },
    'Stag Hunt': {
        'abbrev': 'SH',
        'matrix': np.array([
            [4, 0],  # [R, S]
            [3, 2]   # [T, P]
        ]),
        'labels': ['R=4', 'S=0', 'T=3', 'P=2']
    },
    'Hawk-Dove': {
        'abbrev': 'HD',
        'matrix': np.array([
            [2, 1],  # [R, S]
            [3, 0]   # [T, P]
        ]),
        'labels': ['R=2', 'S=1', 'T=3', 'P=0']
    }
}

print("=" * 80)
print("PAYOFF MATRICES - AGENT STATE INPUTS")
print("=" * 80)
print()
print("The agent receives the payoff matrix as its input (first 4 elements of state).")
print("This is WHY agents behave differently across games - they see different numbers!")
print()

for game_name, info in GAMES.items():
    print(f"\n{game_name} ({info['abbrev']})")
    print("-" * 40)
    print("Matrix (Agent's perspective):")
    print("                Opp Cooperates  Opp Defects")
    print(f"Agent Cooperates:      {info['matrix'][0,0]}            {info['matrix'][0,1]}")
    print(f"Agent Defects:         {info['matrix'][1,0]}            {info['matrix'][1,1]}")
    print()
    print(f"State input: [{info['labels'][0]}, {info['labels'][1]}, {info['labels'][2]}, {info['labels'][3]}, round]")
    print(f"Flattened:   [{info['matrix'].flatten()[0]}, {info['matrix'].flatten()[1]}, {info['matrix'].flatten()[2]}, {info['matrix'].flatten()[3]}, round]")

print("\n" + "=" * 80)
print("WHY BEHAVIOR DIFFERS ACROSS TEST CONDITIONS")
print("=" * 80)

print("\n1. DIFFERENT GAMES (e.g., PD vs HD):")
print("   → Different payoff matrices → Different state inputs")
print("   → Network receives fundamentally different numbers")
print("   → Rational to behave differently!")
print()

print("2. SAME GAME, DIFFERENT OPPONENTS (e.g., PD with p=0.1 vs p=0.9):")
print("   → Same payoff matrix → Same state input at each timestep")
print("   → BUT: Different opponent behavior leads to:")
print("      - Different reward sequences")
print("      - Different LSTM hidden state evolution")
print("      - Network learns patterns through hidden state")
print("   → Agent adapts policy based on experienced outcomes!")
print()

print("3. KEY INSIGHT:")
print("   The LSTM hidden state acts as MEMORY of the interaction history.")
print("   Even with identical inputs, the hidden state encodes:")
print("   - Pattern of rewards received (high vs low)")
print("   - Inferred opponent strategy (cooperative vs exploitative)")
print("   - Learned adaptation (keep cooperating vs switch to defecting)")

print("\n" + "=" * 80)

# Show state vector comparison
print("\nEXAMPLE STATE VECTORS (Round 10):")
print("-" * 80)
for game_name, info in GAMES.items():
    state = list(info['matrix'].flatten()) + [10/100.0]
    print(f"{info['abbrev']:3s}: {state}")

print("\n→ Notice: Same game structure, completely different numerical values!")
print("→ The network has learned to interpret these values and choose actions.")
print()
