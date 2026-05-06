from pathlib import Path
import re

test_dir = Path(r'c:\Users\User\OneDrive - huji.ac.il\מסמכים\Max_Planck\Cognitive-Therapy-fo-AI-\experiments\generalization_matrix_test_893509\testing')
dirs = list(test_dir.glob('model_*_test_cond_*'))

model_ids = set()
for d in dirs:
    m = re.search(r'model_(\d+)_test_cond', d.name)
    if m:
        model_ids.add(int(m.group(1)))

print(f'Unique models: {sorted(model_ids)[:15]}...')
print(f'Total unique models: {len(model_ids)}')
print(f'Total directories: {len(dirs)}')
print(f'Expected: 75 models × 14 test conditions = 1050 directories')
