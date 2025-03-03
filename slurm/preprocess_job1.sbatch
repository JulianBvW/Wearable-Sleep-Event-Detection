#!/bin/bash
#SBATCH --partition=study
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/%j.out

echo "preprocess denoise job"
source ~/.bashrc
conda activate master
srun python wearsed/dataset/preprocessing/preprocess_ppg.py --start 1000 --end 1400

echo "done"