#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm/output/evaluation/pred_hypno_%j.out

echo "cfs eval with event scorings and predicted hypnogram"
echo ""
source ~/.bashrc
srun python wearsed/evaluation/eval_cfs.py

echo "done"