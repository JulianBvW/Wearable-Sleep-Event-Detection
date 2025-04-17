from wearsed.training.metric import get_precision_recall
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

runs = [
    ('final_default', 28, [0, 1, 2, 3]),
    ('final_no_spo2', 28, [0, 1, 2, 3]),
    ('attention_none', 49, [0, 1, 2, 3]),
    ('no_hypnogram', 39, [])
]

run_list = []
correctify_list = []
thr_list = []
prec_list = []
rec_list = []

for run, epoch, folds in runs:
    if len(folds) > 0:
        outputs = []
        for fold in folds:
            output = pd.read_csv(f'wearsed/training/attention_unet/output/{run}/f-{fold}/test_preds_epoch_{epoch}.csv')
            outputs.append(output)
        output = pd.concat(outputs)
        output = output.drop(output[output['targets'] == -999].index)
        output = output.reset_index(drop=True)
    else:
        output = pd.read_csv(f'wearsed/training/baseline_conv/output/{run}/test_preds_epoch_{epoch}.csv')
    y_true, y_pred = output['targets'], output['predictions']
    for correctify in [True, False]:
        print(f'### Run {run} {'with' if correctify else 'without'} correction...')
        for thr in tqdm([i / 20 for i in range(1, 20)]):
            precision, recall = get_precision_recall(y_pred, y_true, thr, correctify, correctify_size=3)
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
df.to_csv('Notebooks/62_plot_generation2.csv', index=False)