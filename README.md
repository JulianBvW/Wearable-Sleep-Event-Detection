# Wearable-Sleep-Event-Detection
Detection of Sleep Events using Machine Learning Models on Wearable Sensor Modalities.

Bachelor Thesis for the University of Bielefeld.

**Description:** After a statistical analysis of the given data to find out relations between demographics data (age, sex, ...), sleep stages and forms of sleep-related events, the goal is to create a Machine Learning Model, for example based on Transformer architectures, that can detect sleep events, like Arousals, (obstructive vs. central) Apneas, or Hypopneas, from a minimal set of sensor modalities that can theoretically be acquired using simple, wearable hardware for home use. In contrast with most literature on the topic, where classification is performed based on epochs of a long duration (e.g. 1 minute), we will explore the use of higher output sampling (e.g. 2 Hz), allowing us to more accurately detect the start and end of each event interval. Performance will be evaluated against events scored based on PSG, both in terms of event detection (sensitivity, positive predictive value, etc) and in terms of agreement with aggregated metrics, such as AHI, arousal index, etc.
