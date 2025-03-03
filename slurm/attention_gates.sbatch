#!/bin/bash
#SBATCH --partition=study
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/016_attention_gates_%j.out

echo "attention gates"
echo ""
source ~/.bashrc
srun python wearsed/training/attention_unet/train_attention_unet.py --out-dir attention_gates --use-attention gates

echo "done"