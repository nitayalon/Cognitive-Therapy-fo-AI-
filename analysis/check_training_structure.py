import pickle
from pathlib import Path

# Load one task to check training results structure
task_file = Path(r'c:\Users\User\OneDrive - huji.ac.il\מסמכים\Max_Planck\Cognitive-Therapy-fo-AI-\experiments\generalization_matrix_834222\generalization_matrix_task_0_20260211_053235\results\task_0_results.pkl')

with open(task_file, 'rb') as f:
    data = pickle.load(f)

print("Top-level keys:")
print(list(data.keys()))

print("\nTraining results structure:")
training_results = data.get('training_results', {})
if isinstance(training_results, dict):
    print(f"  Keys: {list(training_results.keys())}")
    history = training_results.get('history', {})
    if isinstance(history, dict):
        print(f"  History keys: {list(history.keys())}")
        print(f"  History length: {len(history.get('total_loss', []))}")
else:
    print(f"  Type: {type(training_results)}")
