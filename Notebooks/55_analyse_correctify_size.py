import pandas as pd
from tqdm import tqdm
from wearsed.training.metric import get_precision_recall, calc_f1_score

run, folds, epoch = 'attention_gates_bottleneck', [0, 1, 2, 3], 29

def get_full_y(run, folds, epoch):
    outputs = []
    for fold in folds:
        output = pd.read_csv(f'wearsed/training/attention_unet/output/{run}/f-{fold}/test_preds_epoch_{epoch}.csv')
        outputs.append(output)
    output = pd.concat(outputs)
    output = output.drop(output[output['targets'] == -999].index)
    output = output.reset_index(drop=True)
    return output['targets'], output['predictions']

results_thr       = []
results_cor_size  = []
results_precision = []
results_recall    = []
results_f1        = []
y_true, y_pred = get_full_y(run, folds, epoch)

for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    print(f'-> {thr}')
    for correctify_size in tqdm([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
        precision, recall = get_precision_recall(y_pred, y_true, thr, True, correctify_size=correctify_size)
        f1 = calc_f1_score(precision, recall)
        results_thr.append(thr)
        results_cor_size.append(correctify_size)
        results_precision.append(precision)
        results_recall.append(recall)
        results_f1.append(f1)

pd.DataFrame({
    'thr': results_thr,
    'cor_size': results_cor_size,
    'precision': results_precision,
    'recall': results_recall,
    'f1': results_f1,
}).to_csv('Notebooks/55_analyse_correctify_size.csv', index=False)