'''
Load a generated k-Fold for the dataset using NSRR or SOMNOLYZER scorings
'''

import json
import numpy as np

def get_fold(fold_name, fold_nr, seed=42, path='wearsed/training/kfold/'):
    with open(f'{path}{fold_name}.txt', 'r') as f:
        json_dump = f.read()
    class_folds = json.loads(json_dump)
    class_folds = {int(k): v for k, v in class_folds.items()}

    train_set, test_set = [], []
    for class_id in range(4):
        for class_fold_nr in range(len(class_folds[class_id])):
            if class_fold_nr == fold_nr:
                test_set += class_folds[class_id][class_fold_nr]
            else:
                train_set += class_folds[class_id][class_fold_nr]

    np.random.seed(seed)
    np.random.shuffle(train_set)
    np.random.shuffle(test_set)

    return train_set, test_set