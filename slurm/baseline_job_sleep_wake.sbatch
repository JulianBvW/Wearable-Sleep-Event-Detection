#!/bin/bash
#SBATCH --partition=study
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=120:00:00
#SBATCH --output=slurm/output/%j.out

echo "sleep wake with new baseline"
source ~/.bashrc
conda activate master
srun python wearsed/training/baseline_conv/train_baseline_conv_sleep_wake.py --out-dir sleep_wake

echo "done"