# Introduction to the contents of this repository

## 1. Top level

The main folder structure looks like this:
```
.
|- Notebooks/ # .ipynb Notebooks mainly for testing or visualizing data (some calculated .csv files also live here)
|- Paper/     # The Master Thesis .tex files
|- slurm/     # Scripts to run files on the CITEC GPU cluster
|- wearsed/   # The main source code of the project
|- setup.py   # A pip install script for this repository (see README.md #Usage)
```

## 2. Structure of the source code

#### Folder `dataset/`

Main classes are
- WearSEDDataset.py - for the 'main' dataset used for training: MESA
- CFSDataset.py - for the evaluation dataset: CFS

In `data_ids/` there are functions for correctly loading the data points with their ID and correlating them and giving them the AHI severity class for later fold splitting.

#### Folder `models/`

In here are three models for the apnea detection and one for preprocessing:
- `baseline_conv/` has the first model I created that is just a simple UNET
- `attention_unet/` is the main model that I used for the final results, which is a UNET with attention features
- `transformer_model/` is a failed experiment trying to get transformers involved, but results where very bad
- `vae/` sits the preprocessing model that transforms 1-dim, 265Hz PPG to 8-dim, 1Hz signal

While I could have put every 'Option' (SpO2 or not, PPG preprocessed or not, multi-class or not, ...) in one file and handle it with arguments, I thought it would be easier to create clones of the main file and name them with their option. So the Attention UNET without SpO2 is called `AttentionUNet_no_spo2.py` for example. But those are mainly the same model just with minor adjustments.

#### Folder `training/`

Here, everything training happens (on the MESA dataset).
As with the models, we have subfolders for every type of model. Next to those there are mainly utility `.py` files and two folders:
- `kfold/` is there to generate the fold
- `ahi_correction/` is another experiment where I tried to 'correct' the predicted AHI at the end by providing demographic data (like age, sex, etc.), but that did not work directly so I did other things

Again like with the models, have one file for every option, so while `train_attention_unet.py` is the main file, `train_attention_unet_no_spo2_info.py` is for example the run where I gave it no SpO2 and also saved the class (OSA vs CSA vs ...) and hypnogram data for further analysis.

In the training folder for every model there is an output folder with the prediction results, but those are in the .gitignore. So if they are needed, I could provide them.

#### Folder `inference/`

In here there are just simple scripts for running the trained model in inference mode. I think I used those for testing or visualizing results.

#### Folder `evaluation/`

After training on the MESA dataset (if you see somewhere `train_fully` that means the model is not trained on a k-fold, but on the whole MESA dataset), these scripts run the evaluation on the CFS dataset.

## 3. Notebooks

In here, all python notebooks live. I used them for testing and visualizing. They are ordered chronologically when I created them, so the first few notebooks are general data loaded tests or demographic analysis, while at the middle we have mainly results of the main training, and at the end we have visualizing notebooks for the theses or CFS evaluation results.

The interesting ones in there are probably all that have a `.csv` file for the same number as there I calculated results to visualize.

As with the models and training scripts we again have the same naming schema, so for example `59_final_AHI_results_25_cor3.ipynb` is showing the 'final' (meaning final before the thesis) training results at AHI level (differing from the `58_final_results.ipynb` which showed the event-level results) with a classification threshold of `.25` and with a correction value of `3` (the parameter that choose 'if two events are more less then that apart, we merge them').

Also interesting is the last one `73_signal_compare.ipynb`, which compares MESA and CFS signals showing that they are sadly pretty different sometimes, which probably explain the 10% worse performance with the CFS evaluation.