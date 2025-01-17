from wearsed.training.metric import metric, correct
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

runs = [
    ('new_baseline_failsafe', 32),
    ('new_baseline_BN', 31),
    ('no_spo2', 31),
    ('pleth_pre_stat_vae', 49),
    ('pleth_pre_vae', 49),
    ('pleth_pre_stat', 49),
]

def get_precision_recall(y_true, y_pred, threshold, correctify):
    y_pred = (y_pred > threshold)*1
    TP, FP, FN = metric(y_pred, y_true, correctify=correctify)
    precision = TP / (TP + FP) if TP > 0 else 0
    recall = TP / (TP + FN) if TP > 0 else 0
    return precision, recall

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
        for thr in tqdm([i / 20 for i in range(21)]):
            precision, recall = get_precision_recall(y_true, y_pred, thr, correctify)
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
df.to_csv('Notebooks/33_prec_rec_curve_with_metric.csv', index=False)