#!/bin/bash
#SBATCH --partition=study
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00

latent_size=$1

source ~/.bashrc
conda activate master
srun python wearsed/training/vae/train_vae.py --latent-size $latent_size

echo "done"
