#!/bin/bash
#SBATCH --partition=study
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=192:00:00
#SBATCH --output=slurm/output/%j.out

echo "first_multi_class (with more time)"
source ~/.bashrc
conda activate master
srun python wearsed/training/baseline_conv/train_baseline_conv_multi_class.py --out-dir first_multi_class

echo "done"