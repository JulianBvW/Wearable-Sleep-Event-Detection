#!/bin/bash
#SBATCH --partition=study
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/015_attention_none_%j.out

echo "attention none"
echo ""
source ~/.bashrc
srun python wearsed/training/attention_unet/train_attention_unet.py --out-dir attention_none --use-attention none

echo "done"