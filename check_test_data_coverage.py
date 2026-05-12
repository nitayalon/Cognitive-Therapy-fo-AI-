"""Quick script to check test data coverage."""
from pathlib import Path
import pandas as pd

# Check both test experiment directories (NEW data only)
test_exps = [
    Path('experiments/generalization_matrix_test_913244/testing'),
    Path('experiments/generalization_matrix_test_913245/testing')
]

for test_exp in test_exps:
    if not test_exp.exists():
        print(f"\n❌ {test_exp} does not exist")
        continue
    
    print(f"\n{'='*80}")
    print(f"Checking: {test_exp.parent.name}")
    print(f"{'='*80}")
    
    model_dirs = sorted(test_exp.glob('model_*_test_cond_*'))
    print(f"Total test directories: {len(model_dirs)}")
    
    unique_models = set()
    unique_test_conds = set()
    
    for d in model_dirs:
        parts = d.name.split('_')
        model_id = int(parts[1])
        test_cond_id = int(parts[4])
        unique_models.add(model_id)
        unique_test_conds.add(test_cond_id)
    
    print(f"\nUnique models tested: {len(unique_models)} (expected 75)")
    print(f"  Models: {sorted(unique_models)}")
    
    print(f"\nUnique test conditions: {len(unique_test_conds)} (expected 15)")
    print(f"  Conditions: {sorted(unique_test_conds)}")
    
    print(f"\nExpected total: 75 models × 15 test conditions = 1,125 directories")
    print(f"Actual total: {len(model_dirs)} directories")
    
    if len(model_dirs) < 1125:
        missing = 1125 - len(model_dirs)
        print(f"\n⚠️  Missing {missing} test directories ({missing/1125*100:.1f}%)")
        
        # Find which (model, test_cond) pairs are missing
        existing_pairs = set()
        for d in model_dirs:
            parts = d.name.split('_')
            model_id = int(parts[1])
            test_cond_id = int(parts[4])
            existing_pairs.add((model_id, test_cond_id))
        
        expected_pairs = {(m, tc) for m in range(75) for tc in range(15)}
        missing_pairs = expected_pairs - existing_pairs
        
        if len(missing_pairs) > 0:
            print(f"\nMissing {len(missing_pairs)} (model, test_condition) pairs:")
            # Group by model
            from collections import defaultdict
            by_model = defaultdict(list)
            for m, tc in sorted(missing_pairs):
                by_model[m].append(tc)
            
            # Show which models are completely missing vs partially missing
            completely_missing = [m for m, tcs in by_model.items() if len(tcs) == 15]
            partially_missing = {m: tcs for m, tcs in by_model.items() if len(tcs) < 15}
            
            if completely_missing:
                print(f"\n  Models completely untested ({len(completely_missing)}): {completely_missing}")
            
            if partially_missing:
                print(f"\n  Models partially tested ({len(partially_missing)}):")
                # Show first 10 examples
                for m, tcs in sorted(partially_missing.items())[:10]:
                    print(f"    Model {m}: missing {len(tcs)}/15 test_conds {tcs}")
                if len(partially_missing) > 10:
                    print(f"    ... and {len(partially_missing) - 10} more models")
