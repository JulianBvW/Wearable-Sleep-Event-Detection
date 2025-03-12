from wearsed.training.metric import metric, correct
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

runs = [
    ('attention_none', 22, [0, 1, 2, 3]),
    ('attention_gates', 22, [0, 1, 2, 3]),
    ('attention_bottleneck', 22, [0, 1]),
    # ('attention_gates_bottleneck', 25)
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

for run, epoch, folds in runs:
    outputs = []
    for fold in folds:
        output = pd.read_csv(f'wearsed/training/attention_unet/output/{run}/f-{fold}/test_preds_epoch_{epoch}.csv')
        outputs.append(output)
    output = pd.concat(outputs)
    output = output.drop(output[output['targets'] == -999].index)
    output = output.reset_index(drop=True)
    y_true, y_pred = output['targets'], output['predictions']
    for correctify in [True, False]:
        print(f'### Run {run} {'with' if correctify else 'without'} correction...')
        for thr in tqdm([i / 20 for i in range(1, 20)]):
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
df.to_csv('Notebooks/45_results_with_folds1.csv', index=False)