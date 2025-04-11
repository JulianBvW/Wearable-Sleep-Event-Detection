#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/final/plethpre_%j.out

fold_nr=$1

echo "final plethpre"
echo ""
source ~/.bashrc
srun python wearsed/training/attention_unet/train_attention_unet_plethpre.py --out-dir final_plethpre --use-attention gates,bottleneck --fold-nr $fold_nr --use-predicted-hypnogram

echo "done"