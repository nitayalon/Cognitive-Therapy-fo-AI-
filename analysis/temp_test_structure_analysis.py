# Quick analysis of which test conditions exist
import pandas as pd
import sys
sys.path.insert(0, r'c:\Users\User\OneDrive - huji.ac.il\מסמכים\Max_Planck\Cognitive-Therapy-fo-AI-')

# Show actual test pattern for Task 0
print("Task 0 (trained on prisoners-dilemma + very_low) tests on:")
print("  prisoners-dilemma: very_low (baseline), low, mid, high, very_high = 5 conditions")
print("  hawk-dove: very_low, low = 2 conditions")
print("  stag-hunt: very_low, low = 2 conditions")
print("  Total: 9 conditions out of 15 possible (3 games × 5 opponents)")
print()
print("This is the experimental design:")
print("  1. Baseline (same game + same opponents): 1 condition")
print("  2. Same game + 4 other opponent ranges: 4 conditions")
print("  3. Other games (2) tested at 2 specific opponent ranges: 4 conditions")
print()
print("The experiment does NOT test all 15 combinations!")
print("The missing cells (NaN) represent combinations that were NOT tested.")
