#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm/output/evaluation/first_%j.out

echo "first evaluation test for cfs dataset"
echo ""
source ~/.bashrc
srun python wearsed/evaluation/eval_cfs.py

echo "done"