#!/bin/bash
#SBATCH --partition=study
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/%j.out

echo "preprocess vae job"
source ~/.bashrc
conda activate master
srun python wearsed/dataset/pleth_fe/vae_preprocess_pleth.py

echo "done"