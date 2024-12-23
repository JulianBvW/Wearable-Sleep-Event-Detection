#!/bin/bash
#SBATCH --partition=study
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --output=slurm/output/%j.out

echo "conv 5s vae model"
source ~/.bashrc
conda activate master
srun python wearsed/training/vae/train_vae_conv_5s.py

echo "done"
