#!/bin/bash
#SBATCH --partition=study
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=144:00:00
#SBATCH --output=slurm/output/%j.out

echo "no hypnogram with new baseline"
source ~/.bashrc
conda activate master
srun python wearsed/training/baseline_conv/train_baseline_conv_no_hypnogram.py --out-dir no_hypnogram

echo "done"