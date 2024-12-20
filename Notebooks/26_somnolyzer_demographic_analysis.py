from wearsed.dataset.WearSEDDataset import WearSEDDataset
from tqdm import tqdm
import pandas as pd

dataset = WearSEDDataset(scoring_from='somnolyzer', signals_to_read=[], return_recording=True)
subject_data = dataset.subject_infos.loc[dataset.mesa_ids][['nsrr_age', 'nsrr_bmi', 'nsrr_sex', 'nsrr_race', 'nsrr_current_smoker', 'nsrr_ever_smoker', 'nsrr_ahi_hp3r_aasm15', 'nsrr_phrnumar_f1', 'nsrr_tst_f1']]
subject_data.columns = ['nsrr_age', 'nsrr_bmi', 'nsrr_sex', 'nsrr_race', 'nsrr_current_smoker', 'nsrr_ever_smoker', 'nsrr_ahi', 'nsrr_ari', 'nsrr_tst']

ahis = []
aris = []
tsts = []  # TST in minutes

for recording_idx in tqdm(range(len(dataset))):
    recording = dataset[recording_idx]
    ahis.append(recording.get_ahi())
    aris.append(recording.get_ari())
    tsts.append(recording.total_sleep_time_in_sec / 60)

subject_data['self_ahi'] = ahis
subject_data['self_ari'] = aris
subject_data['self_tst'] = tsts

subject_data.to_csv('Notebooks/26_somnolyzer_demographic_data.csv')