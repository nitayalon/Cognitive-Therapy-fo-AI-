import pandas as pd

df = pd.read_csv('experiments/optimal_vs_actual_kld_analysis/optimal_vs_actual_kld.csv')

print('='*80)
print('OPTIMAL VS ACTUAL KLD STATISTICS')
print('='*80)
print(f'Total comparisons: {len(df)}')
print(f'Min KLD: {df["kld"].min():.6f}')
print(f'Max KLD: {df["kld"].max():.6f}')
print(f'Mean KLD: {df["kld"].mean():.6f}')
print(f'Median KLD: {df["kld"].median():.6f}')

diagonal = df[df['is_same_setup']]
print(f'\nDiagonal (same setup) - should be ~0:')
print(f'  Count: {len(diagonal)}')
print(f'  Mean: {diagonal["kld"].mean():.6f}')
print(f'  Max: {diagonal["kld"].max():.6f}')

off_diag = df[~df['is_same_setup']]
print(f'\nOff-diagonal (different setups):')
print(f'  Count: {len(off_diag)}')
print(f'  Mean: {off_diag["kld"].mean():.6f}')
print(f'  Max: {off_diag["kld"].max():.6f}')

print(f'\n' + '='*80)
print('TOP 10 LARGEST DIVERGENCES FROM OPTIMAL')
print('='*80)
top10 = df.nlargest(10, 'kld')
game_abbrev = {'prisoners-dilemma': 'PD', 'stag-hunt': 'SH', 'hawk-dove': 'HD'}
opp_abbrev = {'very_low': 'VL', 'low': 'L', 'mid': 'M', 'high': 'H', 'very_high': 'VH'}
for idx, row in top10.iterrows():
    train_label = f"{game_abbrev[row['train_game']]}-{opp_abbrev[row['train_opponent_range']]}"
    test_label = f"{game_abbrev[row['test_game']]}-{opp_abbrev[row['test_opponent_range']]}"
    print(f"Train: {train_label:8} | Test: {test_label:8} | "
          f"KLD: {row['kld']:.3f} | Optimal: {row['optimal_coop']:.3f} vs Actual: {row['actual_coop']:.3f}")

print(f'\n' + '='*80)
print('SUMMARY BY TRAINING CONDITION (Average KLD to other setups)')
print('='*80)
summary_data = []
for train_game in ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']:
    for train_opp in ['very_low', 'low', 'mid', 'high', 'very_high']:
        train_data = df[
            (df['train_game'] == train_game) &
            (df['train_opponent_range'] == train_opp) &
            (~df['is_same_setup'])
        ]
        if len(train_data) > 0:
            avg_kld = train_data['kld'].mean()
            train_label = f"{game_abbrev[train_game]}-{opp_abbrev[train_opp]}"
            print(f"{train_label:8} | Avg KLD to others: {avg_kld:.3f}")
