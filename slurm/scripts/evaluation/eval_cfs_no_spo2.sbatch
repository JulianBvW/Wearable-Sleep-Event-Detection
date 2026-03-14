#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm/output/evaluation/no_spo2_f3_%j.out

echo "cfs eval without spo2 f3"
echo ""
source ~/.bashrc
srun python wearsed/evaluation/eval_cfs_no_spo2.py

echo "done"