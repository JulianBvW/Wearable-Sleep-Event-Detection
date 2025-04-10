#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/denoised/wavelet_dmey_%j.out

echo "attention none with wavelet_dmey denoising"
echo ""
source ~/.bashrc
srun python wearsed/training/attention_unet/train_attention_unet.py --out-dir denoising_wavelet_dmey --use-attention none --fold-nr 0 --denoised-ppg wavelet_dmey

echo "done"