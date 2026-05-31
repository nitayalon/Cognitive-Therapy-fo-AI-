#!/usr/bin/env python
"""Quick test of imports."""
import sys
print("Python version:", sys.version)

print("Importing Action from cognitive_therapy_ai...")
from cognitive_therapy_ai import Action
print(f"✓ Action imported: {Action.COOPERATE}")

print("Importing ObservationEncoder...")
from cognitive_therapy_ai import ObservationEncoder
encoder = ObservationEncoder("no_game")
print(f"✓ ObservationEncoder created: input_dim={encoder.get_input_dim()}")

print("Importing best_response...")
from cognitive_therapy_ai import analytic_best_response
print(f"✓ analytic_best_response imported: {analytic_best_response}")

print("\n✅ All lightweight imports successful!")
