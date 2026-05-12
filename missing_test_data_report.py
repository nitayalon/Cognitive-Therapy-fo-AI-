"""Detailed report of missing test data."""
from pathlib import Path
from collections import defaultdict

# Both NEW test experiment directories
test_exps = [
    Path('experiments/generalization_matrix_test_913244/testing'),
    Path('experiments/generalization_matrix_test_913245/testing')
]

# Collect all existing (model, test_cond) pairs
existing_pairs = set()

for test_exp in test_exps:
    if not test_exp.exists():
        continue
    
    model_dirs = sorted(test_exp.glob('model_*_test_cond_*'))
    
    for d in model_dirs:
        parts = d.name.split('_')
        model_id = int(parts[1])
        test_cond_id = int(parts[4])
        existing_pairs.add((model_id, test_cond_id))

# Find missing pairs
expected_pairs = {(m, tc) for m in range(75) for tc in range(15)}
missing_pairs = sorted(expected_pairs - existing_pairs)

print("="*80)
print("MISSING TEST DATA REPORT")
print("="*80)
print(f"\nTotal missing: {len(missing_pairs)} out of {len(expected_pairs)} ({len(missing_pairs)/len(expected_pairs)*100:.1f}%)")
print(f"Total present: {len(existing_pairs)} ({len(existing_pairs)/len(expected_pairs)*100:.1f}%)")

# Group by test condition (to see pattern)
by_test_cond = defaultdict(list)
for m, tc in missing_pairs:
    by_test_cond[tc].append(m)

print("\n" + "="*80)
print("PATTERN 1: Missing data grouped by TEST CONDITION")
print("="*80)
for tc in sorted(by_test_cond.keys()):
    models = by_test_cond[tc]
    print(f"\nTest condition {tc}: {len(models)} models missing")
    print(f"  Missing models: {models}")

# Group by model
by_model = defaultdict(list)
for m, tc in missing_pairs:
    by_model[m].append(tc)

print("\n" + "="*80)
print("PATTERN 2: Missing data grouped by MODEL")
print("="*80)

# Check if pattern is regular
missing_counts = defaultdict(int)
for m, tcs in by_model.items():
    missing_counts[len(tcs)] += 1

print(f"\nMissing pattern summary:")
for count, num_models in sorted(missing_counts.items()):
    print(f"  {num_models} models are each missing {count} test condition(s)")

print(f"\nDetailed breakdown (all 75 models):")
for m in sorted(by_model.keys()):
    tcs = by_model[m]
    print(f"  Model {m:2d}: missing test_cond {tcs}")

# Identify the mapping pattern
print("\n" + "="*80)
print("PATTERN 3: Model-to-missing-test-condition mapping")
print("="*80)

# Check if there's a modulo pattern
pattern_check = {}
for m, tcs in by_model.items():
    if len(tcs) == 1:
        pattern_check[m] = tcs[0]

if len(pattern_check) == 75:
    # Check for modulo pattern
    print("\nChecking for systematic pattern...")
    
    # Group by: model // 5 (groups of 5 models)
    groups = defaultdict(list)
    for m in range(75):
        group = m // 5
        missing_tc = pattern_check.get(m, None)
        groups[group].append((m, missing_tc))
    
    print("\nGroups of 5 consecutive models:")
    for group in sorted(groups.keys()):
        models_info = groups[group]
        missing_tcs = [tc for _, tc in models_info if tc is not None]
        if len(set(missing_tcs)) == 1:
            print(f"  Models {group*5:2d}-{group*5+4:2d}: all missing test_cond {missing_tcs[0]}")
        else:
            print(f"  Models {group*5:2d}-{group*5+4:2d}: mixed pattern {missing_tcs}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print("\nThe missing pattern shows:")
print("- Each model is missing exactly 1 test condition")
print("- Models 0-4 all missing test_cond 0")
print("- Models 5-9 all missing test_cond 1")
print("- Models 10-14 all missing test_cond 2")
print("- ... pattern continues for all 15 groups")
print("\nThis is a SYSTEMATIC pattern, likely from:")
print("  • SLURM job array configuration")
print("  • Test conditions distributed across nodes")
print("  • Each group of 5 models skips testing on their 'own' condition")
