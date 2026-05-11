import pickle
from pathlib import Path
import json

exp_path = Path('..') / 'generalization_matrix_871188'
task_dir = sorted(exp_path.glob('generalization_matrix_task_10_*'))[0]
pkl_path = task_dir / 'results' / 'task_10_results.pkl'

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print('='*80)
print('EVALUATION_SUMMARIES STRUCTURE')
print('='*80)
print('Type:', type(data['evaluation_summaries']))
print('\nKeys:', list(data['evaluation_summaries'].keys()))

for key, value in data['evaluation_summaries'].items():
    print(f'\n{key}:')
    print(json.dumps(value, indent=2))
    break
