import pickle
from pathlib import Path

# Load one task to check format
task_file = Path(r'c:\Users\User\OneDrive - huji.ac.il\מסמכים\Max_Planck\Cognitive-Therapy-fo-AI-\experiments\generalization_matrix_834222\generalization_matrix_task_0_20260211_053235\results\task_0_results.pkl')

with open(task_file, 'rb') as f:
    data = pickle.load(f)

print("Training condition:")
train_cond = data.get('training_condition', {})
print(f"  Game: {train_cond.get('game')}")
print(f"  Opponent range: {train_cond.get('opponent_range')}")
print(f"  Opponent probs: {train_cond.get('opponent_probs')}")

print("\nDetailed evaluation result structure:")
eval_results = data.get('evaluation_results', {})
for i, (key, value) in enumerate(list(eval_results.items())[:3]):
    print(f"\n{i+1}. Test condition key: '{key}'")
    if isinstance(value, dict):
        print(f"   Condition: {value.get('condition')}")
        print(f"   Game: {value.get('game')}")
        print(f"   Opponent range: {value.get('opponent_range')}")
        results = value.get('results', {})
        if isinstance(results, dict):
            print(f"   Results keys: {list(results.keys())}")
            print(f"   Cooperation rate: {results.get('cooperation_rate')}")
            print(f"   Avg reward: {results.get('avg_reward')}")
