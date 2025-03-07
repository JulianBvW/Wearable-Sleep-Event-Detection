'''
Generate a k-Fold for the dataset using NSRR or SOMNOLYZER scorings
'''

from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser(description='Code for generating a k-Fold')
parser.add_argument('--seed', help='random seed', default=42, type=int)
parser.add_argument('--folds', help='number of folds', default=4, type=int)
parser.add_argument('--scorings-from', help='nsrr or somnolyzer', default='somnolyzer', type=str)
args = parser.parse_args()

mesa_ids = pd.read_csv(f'wearsed/dataset/data_ids/mesa_ids_{args.scorings_from}.csv')

print(mesa_ids)