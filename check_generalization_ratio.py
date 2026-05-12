"""Check if generalization ratio has any valid (non-NaN) values."""
import pandas as pd
import numpy as np
from pathlib import Path

# Check task-opponent setup
csv_path = Path('Results/task_opponent_setup/unified_data/task_opponent_generalization_ratio_individual.csv')

if csv_path.exists():
    df = pd.read_csv(csv_path)
    
    print("="*80)
    print("TASK-OPPONENT GENERALIZATION RATIO CHECK")
    print("="*80)
    
    print(f"\nTotal agents: {len(df)}")
    print(f"Valid (non-NaN) ratios: {(~df['generalization_ratio'].isna()).sum()}")
    print(f"NaN ratios: {df['generalization_ratio'].isna().sum()}")
    
    print(f"\nFirst 10 agents:")
    print(df[['model_id', 'train_game', 'train_opponent', 'seed', 
              'mean_training_reward', 'mean_non_training_reward', 
              'generalization_ratio']].head(10).to_string())
    
    if (~df['generalization_ratio'].isna()).sum() > 0:
        print(f"\n✓ Some valid ratios found!")
        print(f"\nValid ratio examples:")
        valid = df[~df['generalization_ratio'].isna()]
        print(valid[['model_id', 'train_game', 'train_opponent', 
                    'mean_training_reward', 'generalization_ratio']].head())
    else:
        print(f"\n❌ ALL RATIOS ARE NaN!")
        print(f"\nConclusion: Generalization ratio metric is MEANINGLESS")
        print(f"because models were never tested on their training condition.")
else:
    print(f"File not found: {csv_path}")
