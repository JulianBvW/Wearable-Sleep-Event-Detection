from wearsed.training.metric import get_best_f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os

runs = [
    ('attention_none', [0, 1, 2, 3]),
    ('attention_gates', [0, 1, 2, 3]),
    ('attention_bottleneck', [0, 1])
]

best_f1s = {}

for run, folds in runs:
    for fold in folds:
        print(f'-> {run}, fold {fold}')
        idx = f'{run}_f{fold}'
        best_f1s[idx] = []
        best_f1 = 0
        for epoch in tqdm(range(50)):
            if os.path.isfile(f'wearsed/training/attention_unet/output/{run}/f-{fold}/test_preds_epoch_{epoch}.csv'):
                output = pd.read_csv(f'wearsed/training/attention_unet/output/{run}/f-{fold}/test_preds_epoch_{epoch}.csv')
                y_true, y_pred = output['targets'], output['predictions']
                best_f1, _, _ = get_best_f1_score(y_pred, y_true)
                best_f1s[idx].append(best_f1)
            else:
                best_f1s[idx].append(best_f1)


df = pd.DataFrame(best_f1s)
df.to_csv('Notebooks/45_results_with_folds_best_f1.csv', index=False)