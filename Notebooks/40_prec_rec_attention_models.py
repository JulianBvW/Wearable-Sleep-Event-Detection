from wearsed.training.metric import metric, correct
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

runs = [
    ('attention_none', 23),
    #('attention_gates', 25),
    #('attention_bottleneck', 25)
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
    output = pd.read_csv(f'wearsed/training/attention_unet/output/{run}/test_preds_epoch_{epoch}.csv')
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
df.to_csv('Notebooks/40_prec_rec_attention_models11.csv', index=False)