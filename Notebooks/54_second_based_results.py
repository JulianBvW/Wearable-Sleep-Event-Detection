from wearsed.training.metric import get_best_f1_score
from tqdm import tqdm
import pandas as pd
import os

run, fold = 'attention_bottleneck', 1

best_f1s = {}

best_f1s = []
best_f1 = 0
for epoch in tqdm(range(50)):
    if os.path.isfile(f'wearsed/training/attention_unet/output/{run}/f-{fold}/test_preds_epoch_{epoch}.csv'):
        output = pd.read_csv(f'wearsed/training/attention_unet/output/{run}/f-{fold}/test_preds_epoch_{epoch}.csv')
        y_true, y_pred = output['targets'], output['predictions']
        best_f1, _, _ = get_best_f1_score(y_pred, y_true, event_based=False)
        best_f1s.append(best_f1)
    else:
        best_f1s.append(best_f1)


df = pd.DataFrame(best_f1s)
df.to_csv('Notebooks/54_second_based_results.csv', index=False)