#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/cv/017_self_attention_bottleneck_%j.out

fold_nr=$1

echo "attention bottleneck"
echo ""
source ~/.bashrc
srun python wearsed/training/attention_unet/train_attention_unet.py --out-dir attention_bottleneck --use-attention bottleneck --fold-nr $fold_nr

echo "done"