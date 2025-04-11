#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/final/reduce_lr_%j.out

fold_nr=$1

echo "final reduce lr"
echo ""
source ~/.bashrc
srun python wearsed/training/attention_unet/train_attention_unet_reduce_lr.py --out-dir final_reduce_lr --use-attention gates,bottleneck --fold-nr $fold_nr --use-predicted-hypnogram --denoised-ppg lowpass

echo "done"