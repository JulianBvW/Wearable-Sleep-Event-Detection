#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/denoised/wavelet_db4_%j.out

echo "attention none with wavelet_db4 denoising"
echo ""
source ~/.bashrc
srun python wearsed/training/attention_unet/train_attention_unet.py --out-dir denoising_wavelet_db4 --use-attention none --fold-nr 0 --denoised-ppg wavelet_db4

echo "done"