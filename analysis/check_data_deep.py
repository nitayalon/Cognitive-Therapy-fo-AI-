import pickle
from pathlib import Path

# Load one task to check format
task_file = Path(r'c:\Users\User\OneDrive - huji.ac.il\מסמכים\Max_Planck\Cognitive-Therapy-fo-AI-\experiments\generalization_matrix_834222\generalization_matrix_task_0_20260211_053235\results\task_0_results.pkl')

with open(task_file, 'rb') as f:
    data = pickle.load(f)

eval_results = data.get('evaluation_results', {})

# Check baseline in detail
baseline = eval_results.get('baseline', {})
print("Baseline structure:")
print(f"  Condition: {baseline.get('condition')}")
print(f"  Game: {baseline.get('game')}")
print(f"  Opponent range: {baseline.get('opponent_range')}")

results = baseline.get('results', {})
print(f"\n  Results has {len(results)} opponent-specific results:")
for opp_key, opp_data in list(results.items())[:2]:
    print(f"\n  Opponent key: {opp_key}")
    if isinstance(opp_data, dict):
        print(f"    Keys: {list(opp_data.keys())}")
        print(f"    Cooperation rate: {opp_data.get('cooperation_rate')}")
        print(f"    Avg reward: {opp_data.get('avg_reward')}")
        print(f"    Opponent pred accuracy: {opp_data.get('opponent_prediction_accuracy')}")
