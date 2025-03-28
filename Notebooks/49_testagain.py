from wearsed.training.metric import get_precision_recall
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

runs = [
    ('testagain_baseline_1', 44),
    ('testagain_baseline_2', 44),
    ('testagain_baseline_3', 44),
    ('testagain_plethpre_1', 44),
    ('testagain_plethpre_2', 44),
    ('testagain_plethpre_3', 44)
]

run_list = []
correctify_list = []
thr_list = []
prec_list = []
rec_list = []

for run, epoch in runs:
    output = pd.read_csv(f'wearsed/training/baseline_conv/output/{run}/test_preds_epoch_{epoch}.csv')
    y_true, y_pred = output['targets'], output['predictions']
    for correctify in [True, False]:
        print(f'### Run {run} {'with' if correctify else 'without'} correction...')
        for thr in tqdm([i / 20 for i in range(1, 20)]):
            precision, recall = get_precision_recall(y_pred, y_true, thr, correctify)
            run_list.append(run)
            correctify_list.append(correctify)
            thr_list.append(thr)
            prec_list.append(precision)
            rec_list.append(recall)

df = pd.DataFrame({
    'run': run_list,
    'correctify': correctify_list,
    'thr': thr_list,
    'precision': prec_list,
    'recall': rec_list
})
df['f1'] = (2 * df['precision'] * df['recall']) / (df['precision'] + df['recall'])
df.to_csv('Notebooks/49_testagain.csv', index=False)