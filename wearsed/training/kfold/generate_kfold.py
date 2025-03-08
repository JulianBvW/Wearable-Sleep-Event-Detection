'''
Generate a k-Fold for the dataset using NSRR or SOMNOLYZER scorings
'''

from argparse import ArgumentParser
import pandas as pd
import numpy as np
import json

parser = ArgumentParser(description='Code for generating a k-Fold')
parser.add_argument('--seed', help='random seed', default=42, type=int)
parser.add_argument('--folds', help='number of folds', default=4, type=int)
parser.add_argument('--scorings-from', help='nsrr or somnolyzer', default='somnolyzer', type=str)
args = parser.parse_args()

mesa_ids = pd.read_csv(f'wearsed/dataset/data_ids/mesa_ids_{args.scorings_from}.csv')

np.random.seed(args.seed)

class_folds = {}
for class_id in range(4):
    class_values = mesa_ids[mesa_ids['ahi_severity_class'] == class_id]['id'].values
    np.random.shuffle(class_values)
    folds = []
    fold_size = len(class_values) // args.folds
    for fold_nr in range(args.folds):
        folds.append(sorted([int(x) for x in class_values[fold_nr*fold_size:(fold_nr+1)*fold_size]]))
    class_folds[class_id] = folds

with open(f'wearsed/training/kfold/fold-{args.folds}-{args.scorings_from}.txt', 'w') as f:
    f.write(json.dumps(class_folds))