import pandas as pd
import numpy as np

# Load gradient flow data
df = pd.read_csv('Results/task_opponent_918988_analysis/network_representation/gradient_flow.csv')

# Check HD-0.1
hd_01 = df[(df['game']=='hawk-dove') & (df['opponent']==0.1)]
print('HD-0.1 gradient norms:')
print(f'  Mean: {hd_01["grad_norm"].mean():.6e}')
print(f'  Max: {hd_01["grad_norm"].max():.6e}')
print(f'  Min: {hd_01["grad_norm"].min():.6e}')
print(f'  Num params: {len(hd_01)}')

# Check HD-0.9
hd_09 = df[(df['game']=='hawk-dove') & (df['opponent']==0.9)]
print('\nHD-0.9 gradient norms:')
print(f'  Mean: {hd_09["grad_norm"].mean():.6e}')
print(f'  Max: {hd_09["grad_norm"].max():.6e}')
print(f'  Min: {hd_09["grad_norm"].min():.6e}')
print(f'  Num params: {len(hd_09)}')

# For comparison, check other conditions
print('\nFor comparison:')
for game, opp in [('prisoners-dilemma', 0.5), ('hawk-dove', 0.5), ('stag-hunt', 0.5)]:
    data = df[(df['game']==game) & (df['opponent']==opp)]
    print(f'{game[:2].upper()}-{opp}: Mean={data["grad_norm"].mean():.6e}, Max={data["grad_norm"].max():.6e}')

# Load alignment matrix to check diagonal
mat = np.load('Results/task_opponent_918988_analysis/network_representation/gradient_alignment_matrix.npy')
games = ['PD', 'HD', 'SH']
opponents = [0.1, 0.3, 0.5, 0.7, 0.9]
labels = [f'{g}-{o}' for g in games for o in opponents]

print('\nDiagonal values (self-similarity):')
for i in range(15):
    print(f'{labels[i]}: {mat[i,i]:.6f}')
