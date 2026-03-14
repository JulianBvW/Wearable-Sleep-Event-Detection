#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm/output/evaluation/train_fully_no_spo2_%j.out

echo "cfs eval without spo2 train fully"
echo ""
source ~/.bashrc
srun python wearsed/evaluation/eval_cfs_no_spo2.py

echo "done"