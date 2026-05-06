import pickle
from pathlib import Path

train_file = Path("experiments/generalization_matrix_train_888509/training/condition_0_seed_0/generalization_matrix_task_0_20260324_131903/results/training_task_0_results.pkl")

with open(train_file, 'rb') as f:
    data = pickle.load(f)

epochs = data.get('training_results', {}).get('epoch_results', [])
print(f'Number of epochs: {len(epochs)}')

if epochs:
    print('\nLast epoch keys:', list(epochs[-1].keys()))
    print('\nLast epoch sample:')
    for k, v in epochs[-1].items():
        if k != 'per_opponent':
            print(f'  {k}: {v}')
    
    # Check per_opponent data
    if 'per_opponent' in epochs[-1]:
        print('\nPer-opponent keys:', list(epochs[-1]['per_opponent'].keys())[:3])
        first_opp = list(epochs[-1]['per_opponent'].keys())[0]
        print(f'\nFirst opponent metrics: {epochs[-1]["per_opponent"][first_opp]}')
