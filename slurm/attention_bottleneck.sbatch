#!/bin/bash
#SBATCH --partition=study
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/017_self_attention_bottleneck_%j.out

echo "attention bottleneck"
echo ""
source ~/.bashrc
srun python wearsed/training/attention_unet/train_attention_unet.py --out-dir attention_bottleneck --use-attention bottleneck

echo "done"