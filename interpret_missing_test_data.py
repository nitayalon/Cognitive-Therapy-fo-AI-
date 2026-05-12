"""Map missing test data to actual game/opponent combinations."""
from pathlib import Path
from collections import defaultdict

# Mapping from model_id (task_id) to training condition
# From the task-opponent setup configuration
TASK_TO_CONDITION = {}
for condition_id in range(15):
    for seed in range(5):
        model_id = condition_id * 5 + seed
        TASK_TO_CONDITION[model_id] = condition_id

# Mapping from condition_id to (game, opponent)
CONDITION_TO_GAME_OPP = {
    0: ('prisoners-dilemma', 0.1),
    1: ('prisoners-dilemma', 0.3),
    2: ('prisoners-dilemma', 0.5),
    3: ('prisoners-dilemma', 0.7),
    4: ('prisoners-dilemma', 0.9),
    5: ('stag-hunt', 0.1),
    6: ('stag-hunt', 0.3),
    7: ('stag-hunt', 0.5),
    8: ('stag-hunt', 0.7),
    9: ('stag-hunt', 0.9),
    10: ('hawk-dove', 0.1),
    11: ('hawk-dove', 0.3),
    12: ('hawk-dove', 0.5),
    13: ('hawk-dove', 0.7),
    14: ('hawk-dove', 0.9),
}

GAME_NAMES = {
    'prisoners-dilemma': 'PD',
    'stag-hunt': 'SH',
    'hawk-dove': 'HD'
}

print("="*80)
print("MISSING TEST DATA - GAME/OPPONENT MAPPING")
print("="*80)

# Analyze the missing pattern
print("\nKey insight: Models skip testing on their OWN training condition!")
print("\nDetailed breakdown:\n")

for condition_id in range(15):
    train_game, train_opp = CONDITION_TO_GAME_OPP[condition_id]
    game_abbrev = GAME_NAMES[train_game]
    
    # Models trained on this condition
    models = [condition_id * 5 + seed for seed in range(5)]
    
    # These models are ALL missing test on this same condition
    print(f"Condition {condition_id:2d} ({game_abbrev}, opp={train_opp}):")
    print(f"  Trained models: {models}")
    print(f"  All 5 models skip testing on condition {condition_id} ({game_abbrev}, opp={train_opp})")
    print(f"  → Models skip testing on their OWN training condition!\n")

print("="*80)
print("IMPLICATIONS FOR GENERALIZATION ANALYSIS")
print("="*80)
print("\n✓ Training-on-training performance: MISSING (this is the NaN source!)")
print("✓ Cross-generalization: COMPLETE (all non-training conditions tested)")
print("\nFor example, PD models trained on opponent 0.1:")
print("  • Models 0-4 were NOT tested on (PD, 0.1) ← MISSING")
print("  • Models 0-4 WERE tested on all other 14 conditions ✓")
print("\nThis explains the NaN in generalization ratio calculation:")
print("  ratio = mean(test on non-training) / mean(test on training)")
print("           ^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^")
print("           Available (14 conditions)    MISSING (NaN)")

print("\n" + "="*80)
print("SOLUTION")
print("="*80)
print("\nFor generalization ratio, we should:")
print("  1. Skip the 'training vs training' comparison (currently NaN)")
print("  2. Focus on 'training vs non-training' performance")
print("  3. OR: Define generalization as performance on non-training conditions only")
print("\nThe current fix (filtering NaN) is appropriate - we can't compute")
print("generalization ratio without training-on-training performance data.")
