#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm/output/evaluation/train_fully_default_%j.out

echo "train fully default evaluation"
echo ""
source ~/.bashrc
srun python wearsed/evaluation/eval_cfs.py

echo "done"