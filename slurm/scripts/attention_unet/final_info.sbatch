#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/final/info_%j.out

fold_nr=$1

echo "final info"
echo ""
source ~/.bashrc
srun python wearsed/training/attention_unet/train_attention_unet_info.py --out-dir final_info --use-attention gates,bottleneck --fold-nr $fold_nr --use-predicted-hypnogram --denoised-ppg lowpass

echo "done"