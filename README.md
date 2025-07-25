# Wearable-Sleep-Event-Detection
**Sleep Apnea** is a breathing disorder affecting **roughly 10% of the adult population**, where breathing can stop periodically during the night, resulting in bad sleep and other side effects.
These resipiratory events are defined as complete (>90%, apnea) or partial (>30%, hypopnea) reductions in airflow during sleep.
Apnea can further be divided into central (the brain fails to send breathing signals to the muscles) or obstructive (blockage in airway canal) apnea.

The gold standard to detecting sleep events is a human-rated PSG (Polysomnography), measuring many signals including brain, eye, muscle, and heart activity, airflow, leg movements, blood oxygen levels, etc.
Due to the hard and costly setup, an estimated 80% of sleep apnea cases are unrecognized.

In this work we created a deep learning model, that can detect apnea events from easy-to-aquire sginals like SpO2 levels (blood oxygen saturation) and photoplethysmography (PPG, blood vessel volume). As the wearable measurement devices (e.g. finger-worn sensor, smart watch, smart ring) for these signals are very comfortable and cheap, our model could help with screening for the huge number of undiagnosed sleep apnea cases.

## Results

We achieved a peak event detection F1-score of 70% when using PPG and SpO2, and 61% with only PPG as input (useful with devices that cannot measure SpO2 reliably like smart watches).

When looking at AHI (events per hour of sleep) correlation, we achieved Spearman's rank correlation coefficient of 0.917 and ICC of 0.91 with little to no bias.

<img width="736" height="359" alt="image" src="https://github.com/user-attachments/assets/bc455c2f-af89-431e-8e14-fba3bca74c49" />

**TBA**: Evaluation results

## Datasets

We used the [MESA](https://sleepdata.org/datasets/mesa) dataset as train and test set, with just under 1900 overnight recordings.
For evaluation we used the [CFS](https://sleepdata.org/datasets/cfs) dataset with around 300 recordings.

## Architecture

The model is a modified U-Net with Attention mechanism baked into it.
As input, we used the raw PPG-signal, a PPG-predicted hypnogram (using the work from [Bakker et al.](https://pubmed.ncbi.nlm.nih.gov/33660612/)), and optionally SpO2.
The output is a 1D tensor that shows event probabilities at 1Hz.

<img width="550" height="616" alt="image" src="https://github.com/user-attachments/assets/c1176c16-6ffd-457d-962e-2d8af3d354f5" />

## Usage

### Installation

Create a conda environment:
```bash
conda create --name wearsed python=3.12
conda activate wearsed
```

Install packages and the the project:
```bash
pip install numpy pandas tqdm pyEDFlib lxml matplotlib torch torchvision torchaudio scikit-learn scikit-image ipykernel h5py positional-encodings
pip install -e .
```

### Training

To load the datasets, run `python wearsed/dataset/data_ids/load_<mesa or cfs>.py <path to mesa or cfs root>`.
Then, the dataset classes in `wearsed/dataset/` are able to load the recordings.

In `slurm/scripts/` you can find a selection of scripts for training the model and the code in `wearsed/training/` shows how to use the model.

### Evaluation

**TBA**: Add evaluation help

## MESA PSG Signals

Signal | Description | Frequency (Hz) | #Measurements in example recording
--- | --- | --: | --:
`EKG`       | Heart Activity |  256 | 11058944
`EOG-L`     | Movement of left Eye |  256 | 11058944
`EOG-R`     | Movement of right Eye |  256 | 11058944
`EMG`       | Muscle Activity |  256 | 11058944
`EEG1`      | Brain Activity |  256 | 11058944
`EEG2`      | Brain Activity |  256 | 11058944
`EEG3`      | Brain Activity |  256 | 11058944
`Pres`      | Airway Pressure |   32 |  1382368
`Flow`      | Airflow |   32 |  1382368
`Snore`     | Snoring Intensity |   32 |  1382368
`Thor`      | Thoracic (Chest) Movement |   32 |  1382368
`Abdo`      | Abdominal (Belly) Movement |   32 |  1382368
`Leg`       | Leg Movement |   32 |  1382368
`Therm`     | Airflow Temperature Changes |   32 |  1382368
`Pos`       | Body (Sleeping) Position |   32 |  1382368
`EKG_Off`   |  |    1 |    43199
`EOG-L_Off` |  |    1 |    43199
`EOG-R_Off` |  |    1 |    43199
`EMG_Off`   |  |    1 |    43199
`EEG1_Off`  |  |    1 |    43199
`EEG2_Off`  |  |    1 |    43199
`EEG3_Off`  |  |    1 |    43199
`Pleth`     | Blood Volume Changes |  256 | 11058944
`OxStatus`  | Oxygen Status |    1 |    43199
`SpO2`      | Blood Oxygen Saturation |    1 |    43199
`HR`        | Heartbeats per Minute |    1 |    43199
`DHR`       | Change in HR |  256 | 11058944
