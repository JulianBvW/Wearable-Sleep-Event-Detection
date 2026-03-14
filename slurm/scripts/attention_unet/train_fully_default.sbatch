#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/train_fully/default_%j.out

echo "train fully default"
echo ""
source ~/.bashrc
srun python wearsed/training/attention_unet/train_fully.py --out-dir fully_default --use-attention gates,bottleneck --fold-nr 0 --use-predicted-hypnogram --denoised-ppg lowpass

echo "done"