import pickle
from pathlib import Path

# Load one task to check training results structure
task_file = Path(r'c:\Users\User\OneDrive - huji.ac.il\מסמכים\Max_Planck\Cognitive-Therapy-fo-AI-\experiments\generalization_matrix_834222\generalization_matrix_task_0_20260211_053235\results\task_0_results.pkl')

with open(task_file, 'rb') as f:
    data = pickle.load(f)

training_results = data.get('training_results', {})
print("Training results keys:", list(training_results.keys()))

epoch_results = training_results.get('epoch_results', {})
print("\nEpoch results:")
if isinstance(epoch_results, dict):
    print(f"  Type: dict")
    print(f"  Keys: {list(epoch_results.keys())[:5]}")
elif isinstance(epoch_results, list):
    print(f"  Type: list")
    print(f"  Length: {len(epoch_results)}")
    if len(epoch_results) > 0:
        print(f"  First entry type: {type(epoch_results[0])}")
        if isinstance(epoch_results[0], dict):
            print(f"  First entry keys: {list(epoch_results[0].keys())}")

final_metrics = training_results.get('final_metrics', {})
print(f"\nFinal metrics: {final_metrics}")
