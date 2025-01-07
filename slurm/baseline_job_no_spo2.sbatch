#!/bin/bash
#SBATCH --partition=study
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/%j.out

echo "no spo2 with new baseline"
source ~/.bashrc
conda activate master
srun python wearsed/training/baseline_conv/train_baseline_conv_no_spo2.py --out-dir no_spo2

echo "done"