"""
Inspect the structure of vanilla RL and proto-ToM pickle result files
to understand how to extract metrics.
"""

import pickle
from pathlib import Path
import pprint

# Paths
vanilla_task0 = Path("experiments/vanilla_rl_array_835338_20260223_065612/vanilla_matrix_task0/results/task_0_results.pkl")
proto_tom_task0 = Path("experiments/generalization_matrix_834222/generalization_matrix_task_0_20260211_053235/results")

print("="*80)
print("INSPECTING VANILLA RL TASK 0 RESULTS")
print("="*80)
print()

if vanilla_task0.exists():
    with open(vanilla_task0, 'rb') as f:
        vanilla_results = pickle.load(f)
    
    print(f"Type: {type(vanilla_results)}")
    print(f"\nTop-level keys:")
    if isinstance(vanilla_results, dict):
        pprint.pprint(list(vanilla_results.keys()), width=80)
        
        print("\n" + "-"*80)
        print("Detailed structure:")
        print("-"*80)
        for key in vanilla_results.keys():
            value = vanilla_results[key]
            print(f"\n{key}:")
            print(f"  Type: {type(value)}")
            if isinstance(value, dict):
                print(f"  Keys: {list(value.keys())[:10]}")  # First 10 keys
                # Show a sample of the data
                sample_key = list(value.keys())[0] if value else None
                if sample_key:
                    print(f"  Sample ({sample_key}):")
                    sample_value = value[sample_key]
                    if isinstance(sample_value, dict):
                        print(f"    {list(sample_value.keys())[:5]}")
                    else:
                        print(f"    Value: {str(sample_value)[:200]}")
            elif isinstance(value, list):
                print(f"  Length: {len(value)}")
                if value:
                    print(f"  First item type: {type(value[0])}")
                    print(f"  First item: {str(value[0])[:200]}")
            else:
                print(f"  Value: {str(value)[:200]}")
    else:
        print("\nNot a dictionary - showing full object:")
        pprint.pprint(vanilla_results, depth=3, width=80)
else:
    print("Vanilla task 0 results file not found")

print("\n")
print("="*80)
print("INSPECTING PROTO-TOM TASK 0 RESULTS")  
print("="*80)
print()

# Find proto-tom result file
if proto_tom_task0.exists():
    proto_files = list(proto_tom_task0.glob("*.pkl"))
    print(f"Found {len(proto_files)} pickle files:")
    for f in proto_files:
        print(f"  - {f.name}")
    
    if proto_files:
        proto_file = proto_files[0]
        print(f"\nInspecting: {proto_file.name}")
        print("-"*80)
        
        with open(proto_file, 'rb') as f:
            proto_results = pickle.load(f)
        
        print(f"Type: {type(proto_results)}")
        print(f"\nTop-level keys:")
        if isinstance(proto_results, dict):
            pprint.pprint(list(proto_results.keys()), width=80)
            
            print("\n" + "-"*80)
            print("Detailed structure:")
            print("-"*80)
            for key in list(proto_results.keys())[:10]:  # First 10 keys
                value = proto_results[key]
                print(f"\n{key}:")
                print(f"  Type: {type(value)}")
                if isinstance(value, dict):
                    print(f"  Keys: {list(value.keys())[:10]}")
                    sample_key = list(value.keys())[0] if value else None
                    if sample_key:
                        sample_value = value[sample_key]
                        print(f"  Sample value type: {type(sample_value)}")
                        if isinstance(sample_value, dict):
                            print(f"    Sample keys: {list(sample_value.keys())[:5]}")
                elif isinstance(value, list):
                    print(f"  Length: {len(value)}")
                    if value:
                        print(f"  First item: {str(value[0])[:200]}")
                else:
                    print(f"  Value: {str(value)[:200]}")
else:
    print("Proto-ToM results directory not found")

print("\n" + "="*80)
