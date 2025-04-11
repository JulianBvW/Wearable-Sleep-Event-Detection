#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/final/mulstiscale_cnn_%j.out

fold_nr=$1

echo "final mulstiscale cnn"
echo ""
source ~/.bashrc
srun python wearsed/training/attention_unet/train_attention_unet_multiscale_cnn.py --out-dir mulstiscale_cnn --use-attention gates,bottleneck --fold-nr $fold_nr --use-predicted-hypnogram --denoised-ppg lowpass

echo "done"