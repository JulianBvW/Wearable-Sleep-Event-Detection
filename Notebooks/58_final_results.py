from wearsed.training.metric import get_precision_recall
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

runs = [
    ('final_default', 28, [0, 1, 2, 3]),
    ('final_no_spo2', 28, [0, 1, 2, 3]),
    ('final_reduce_lr', 28, [0, 1, 2, 3]),
    ('final_plethpre', 28, [2, 3]),
    ('final_plethpre_se', 28, [0, 1, 2, 3]),
    ('mulstiscale_cnn', 24, [0, 1, 2, 3]),
    ('mulstiscale_cnn_se', 16, [0, 1, 2, 3])
]

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
df.to_csv('Notebooks/58_final_results.csv', index=False)