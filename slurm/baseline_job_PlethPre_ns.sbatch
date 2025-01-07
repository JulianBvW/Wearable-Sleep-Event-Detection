#!/bin/bash
#SBATCH --partition=study
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/%j.out

echo "pleth pre (vae latents)"
source ~/.bashrc
conda activate master
srun python wearsed/training/baseline_conv/train_baseline_conv_plethpre.py --use-pleth-vae-latents --out-dir pleth_pre_stat_vae_ns

echo "done"