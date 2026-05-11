import pickle
from pathlib import Path

# Load one task to check format
task_file = Path(r'c:\Users\User\OneDrive - huji.ac.il\מסמכים\Max_Planck\Cognitive-Therapy-fo-AI-\experiments\generalization_matrix_834222\generalization_matrix_task_0_20260211_053235\results\task_0_results.pkl')

with open(task_file, 'rb') as f:
    data = pickle.load(f)

print("Training condition:")
print(data.get('training_condition', {}))
print("\nEvaluation result keys (first 15):")
eval_results = data.get('evaluation_results', {})
for i, key in enumerate(list(eval_results.keys())[:15]):
    print(f"  {i+1}. {key}")

print("\nSample evaluation result:")
first_key = list(eval_results.keys())[0]
print(f"Key: {first_key}")
print(f"Value keys: {eval_results[first_key].keys() if isinstance(eval_results[first_key], dict) else 'not a dict'}")
