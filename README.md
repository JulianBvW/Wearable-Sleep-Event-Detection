# Wearable-Sleep-Event-Detection
Detection of Sleep Events using Machine Learning Models on Wearable Sensor Modalities.

Bachelor Thesis for the University of Bielefeld.

**Description:** After a statistical analysis of the given data to find out relations between demographics data (age, sex, ...), sleep stages and forms of sleep-related events, the goal is to create a Machine Learning Model, for example based on Transformer architectures, that can detect sleep events, like Arousals, (obstructive vs. central) Apneas, or Hypopneas, from a minimal set of sensor modalities that can theoretically be acquired using simple, wearable hardware for home use. In contrast with most literature on the topic, where classification is performed based on epochs of a long duration (e.g. 1 minute), we will explore the use of higher output sampling (e.g. 2 Hz), allowing us to more accurately detect the start and end of each event interval. Performance will be evaluated against events scored based on PSG, both in terms of event detection (sensitivity, positive predictive value, etc) and in terms of agreement with aggregated metrics, such as AHI, arousal index, etc.

### Installation

Create a conda environment:
```bash
conda create --name wearsed python=3.12
conda activate wearsed
```

Install packages and the the project:
```bash
pip install numpy pandas tqdm pyEDFlib lxml matplotlib torch torchvision torchaudio scikit-learn ipykernel
pip install -e .
```

### PSG Signals

Signal | Description | Frequency (Hz) | #Measurements
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