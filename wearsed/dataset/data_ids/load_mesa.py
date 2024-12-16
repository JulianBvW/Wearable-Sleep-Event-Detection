import os
import sys
import pandas as pd
from tqdm import tqdm

from wearsed.dataset.data_ids.load_scorings import load_scorings_nsrr, load_scorings_somnolyzer

if len(sys.argv) < 3:
    raise Exception('Usage: `load_mesa.py <MESA ROOT PATH> <SCORING [\'nsrr\', \'somnolyzer\']>`')
mesa_root = sys.argv[1]  # /vol/sleepstudy/datasets/mesa/
scoring_from = sys.argv[2]  # 'nsrr' or 'somnolyzer'
os.makedirs(mesa_root + 'scorings/', exist_ok=True)
os.makedirs(mesa_root + 'scorings/' + scoring_from, exist_ok=True)
os.makedirs(mesa_root + 'scorings/' + scoring_from + '/hypnogram', exist_ok=True)
os.makedirs(mesa_root + 'scorings/' + scoring_from + '/events', exist_ok=True)
os.makedirs(mesa_root + 'scorings/' + scoring_from + '/event_list', exist_ok=True)

def is_data_available(subject_id):
    path_psg     = mesa_root + f'polysomnography/edfs/mesa-sleep-{subject_id:04}.edf'
    path_annot   = mesa_root + f'polysomnography/annotations-events-nsrr/mesa-sleep-{subject_id:04}-nsrr.xml' if scoring_from == 'nsrr' else mesa_root + f'somnolyzer_scorings/mesa-sleep-{subject_id:04}.rml'

    available_psg    = os.path.isfile(path_psg)
    available_annot  = os.path.isfile(path_annot)
    return available_psg and available_annot, (available_psg, available_annot)


label_file = mesa_root + 'datasets/mesa-sleep-harmonized-dataset-0.7.0.csv'
ids = pd.read_csv(label_file)['mesaid']


print('### (1/2) Testing IDs')
failed = 0
failed_psg    = 0
failed_annot  = 0
available_ids = []
for id in tqdm(ids):
    available, (avail_psg, avail_annot) = is_data_available(id)
    if available:
        available_ids.append(id)
    else:
        failed += 1
        failed_psg    += int(not avail_psg)
        failed_annot  += int(not avail_annot)
available_ids = pd.Series(available_ids)
print(f'Failed: {failed}/{len(ids)}')
print(f'  Missing PSG:    {failed_psg}')
print(f'  Missing Annot:  {failed_annot}\n')


print('### (2/2) Reading Scorings')
load_scorings = load_scorings_nsrr if scoring_from == 'nsrr' else load_scorings_somnolyzer
for id in tqdm(available_ids.values):
    load_scorings(mesa_root, id)

available_ids.to_csv(f'wearsed/dataset/data_ids/mesa_ids_{scoring_from}.csv', header=False, index=False)

with open('wearsed/dataset/data_ids/mesa_root.txt', 'w') as f:
    f.write(mesa_root)
