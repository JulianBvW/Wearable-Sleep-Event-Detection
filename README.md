# Wearable-Sleep-Event-Detection
Sleep Apnea is a breathing disorder affecting roughly 10% of the adult population, where breathing can stop periodically during the night, resulting in bad sleep and other side effects.
These resipiratory events are defined as a complete (>90%, apnea) or partial (>30%, hypopnea) reductions in airflow while sleeping.
Apnea can further be divided into central (the brain fails to send breathing signals to the muscles) or obstructive (blockage in airway canal) apnea.
The gold standard to detecting sleep events is a human-rated PSG (Polysomnography), measuring many signals including brain, eye, muscle, and heart activity, airflow, leg movements, blood oxygen levels, etc.
Due to the hard and costly setup, an estimated 80% of cases are unrecognized.

This work tries to create a machine learning model, that can precisely detect and classify sleep-related events on wearable, easy-to-use, and comfortable measurement devices like a finger clip that can record SpO2 levels (blood oxygen saturation) and plethysmography (blood vessel volume).

---

## Task Description

**Detection of Sleep Events using Machine Learning Models on Wearable Sensor Modalities**

After a statistical analysis of the given data to find out relations between demographics data (age, sex, ...), sleep stages and forms of sleep-related events, the goal is to create a Machine Learning Model, for example based on Transformer architectures, that can detect sleep events, like Arousals, (obstructive vs. central) Apneas, or Hypopneas, from a minimal set of sensor modalities that can theoretically be acquired using simple, wearable hardware for home use.
In contrast with most literature on the topic, where classification is performed based on epochs of a long duration (e.g. 1 minute), we will explore the use of higher output sampling (e.g. 2 Hz), allowing us to more accurately detect the start and end of each event interval. Performance will be evaluated against events scored based on PSG, both in terms of event detection (sensitivity, positive predictive value, etc) and in terms of agreement with aggregated metrics, such as AHI, arousal index, etc.

---

## Roadmap

- [X] Request data access for [MESA](https://sleepdata.org/datasets/mesa) and [CFS](https://sleepdata.org/datasets/cfs) dataset
- [X] Create dataloader and do first tests with visuals
- [ ] Read some literature [6/17]
- [ ] Statistical analyses between events, arousals, demographic data, sleep stages, etc.
- [ ] Create simple model and training code (for example CNN) to get first results on binary classification (Event vs. No Event) at 1Hz
- [ ] Increase complexity for the label by distinguishing between more labels like apnea vs. hypopnea and obstructive vs. central or arousals
- [ ] Integrate new event scoring data to get central apnea labels (currently missing in the MESA scoring)
- [ ] Increase model complexity by using different architectures, like Transformers
- [ ] Analyse dependency on singals: Is PPG maybe enough and you can ignore SpO2? Is artigraphy data needed?
- [ ] Write the paper

---

## PSG Signals

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
