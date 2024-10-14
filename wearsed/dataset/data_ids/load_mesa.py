import os
import sys
import pandas as pd
from tqdm import tqdm

if len(sys.argv) < 2:
    raise Exception('Usage: `load_mesa.py <MESA ROOT PATH>`')
mesa_root = sys.argv[1]  # /vol/sleepstudy/datasets/mesa/


def is_data_available(subject_id):
    path_psg     = mesa_root + f'polysomnography/edfs/mesa-sleep-{subject_id:04}.edf'
    path_rpoint  = mesa_root + f'polysomnography/annotations-rpoints/mesa-sleep-{subject_id:04}-rpoint.csv'
    path_annot   = mesa_root + f'polysomnography/annotations-events-nsrr/mesa-sleep-{subject_id:04}-nsrr.xml'

    available_psg    = os.path.isfile(path_psg)
    available_rpoint = os.path.isfile(path_rpoint)
    available_annot  = os.path.isfile(path_annot)
    return available_psg and available_rpoint and available_annot, (available_psg, available_rpoint, available_annot)


label_file = mesa_root + 'datasets/mesa-sleep-harmonized-dataset-0.7.0.csv'
ids = pd.read_csv(label_file)['mesaid']

failed = 0
failed_psg    = 0
failed_rpoint = 0
failed_annot  = 0
available_ids = []
for id in tqdm(ids):
    available, (avail_psg, avail_rpoint, avail_annot) = is_data_available(id)
    if available:
        available_ids.append(id)
    else:
        failed += 1
        failed_psg    += int(not avail_psg)
        failed_rpoint += int(not avail_rpoint)
        failed_annot  += int(not avail_annot)
available_ids = pd.Series(available_ids)
print(f'Failed: {failed}/{len(ids)}')
print(f'  Missing PSG:    {failed_psg}')
print(f'  Missing RPoint: {failed_rpoint}')
print(f'  Missing Annot:  {failed_annot}')

available_ids.to_csv('wearsed/dataset/data_ids/mesa_ids.csv')

with open('wearsed/dataset/data_ids/mesa_root.txt', 'w') as f:
    f.write(mesa_root)
