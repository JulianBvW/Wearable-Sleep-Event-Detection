#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/denoised/bandpass_%j.out

echo "attention none with bandpass denoising"
echo ""
source ~/.bashrc
srun python wearsed/training/attention_unet/train_attention_unet.py --out-dir denoising_bandpass --use-attention none --fold-nr 0 --denoised-ppg bandpass

echo "done"