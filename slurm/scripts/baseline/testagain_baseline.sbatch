#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/ta/baseline_%j.out

nn=$1

echo "baseline test again"
echo ""
source ~/.bashrc
srun python wearsed/training/baseline_conv/train_baseline_conv.py --out-dir testagain_baseline_$nn

echo "done"