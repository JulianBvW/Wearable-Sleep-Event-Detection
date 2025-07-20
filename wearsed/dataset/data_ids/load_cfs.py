import os
import sys
import pyedflib
import pandas as pd
from tqdm import tqdm
from wearsed.dataset.data_ids.load_scorings import load_scorings_somnolyzer

if len(sys.argv) < 2:
    raise Exception('Usage: `load_cfs.py <CFS ROOT PATH>`')
cfs_root = sys.argv[1]  # /vol/sleepstudy/datasets/cfs/

def is_data_available(subject_id):
    path_psg   = cfs_root + f'polysomnography/edfs/cfs-visit5-{subject_id}.edf'
    path_annot = cfs_root + f'polysomnography/annotations-events-nsrr/cfs-visit5-{subject_id}-nsrr.xml'
    path_scoring = cfs_root + f'somnolyzer_scorings/cfs-visit5-{subject_id}.rml'
    path_hypnogram = cfs_root + f'predicted_hypnogram/cfs-{subject_id}-1.csv'

    # Check if EDF and annotation files exist
    if not os.path.isfile(path_psg) or not os.path.isfile(path_annot):
        return False, False, os.path.isfile(path_scoring), os.path.isfile(path_hypnogram)

    # Check SpO2 and PPG signals existing
    edf_reader = pyedflib.EdfReader(path_psg)
    signal_labels = edf_reader.getSignalLabels()
    edf_reader.close()

    return True, 'PlethWV' in signal_labels and 'SpO2' in signal_labels, os.path.isfile(path_scoring), os.path.isfile(path_hypnogram)


label_file = cfs_root + 'datasets/cfs-visit5-harmonized-dataset-0.7.0.csv'
ids = pd.read_csv(label_file)['nsrrid']


print('### (1/2) Testing IDs')
failed_no_edf = 0
failed_no_ppg = 0
failed_no_scr = 0
failed_no_hyp = 0
available_ids = []
for id in tqdm(ids):
    file_available, ppg_available, scoring_available, hypnogram_available = is_data_available(id)
    if file_available and ppg_available and scoring_available and hypnogram_available:
        available_ids.append(id)
    else:
        failed_no_edf += not file_available
        failed_no_ppg += not ppg_available
        failed_no_scr += not scoring_available
        failed_no_hyp += not hypnogram_available
print(f'No EDF: {failed_no_edf}')
print(f'No PPG: {failed_no_ppg - failed_no_edf}')
print(f'No Scoring: {failed_no_scr}')
print(f'No Hypnogram: {failed_no_hyp}')
print(f'Available: {len(available_ids)}/{len(ids)}\n')


print('### (2/2) Reading Scorings')
for id in tqdm(available_ids):
    load_scorings_somnolyzer(cfs_root, id, file_prefix='cfs-visit5-')
print()


print('### Saving to disk... ', end='')
pd.Series(available_ids).to_csv(f'wearsed/dataset/data_ids/cfs_ids.csv', header=False, index=False)

with open('wearsed/dataset/data_ids/cfs_root.txt', 'w') as f:
    f.write(cfs_root)
print('Done!')
