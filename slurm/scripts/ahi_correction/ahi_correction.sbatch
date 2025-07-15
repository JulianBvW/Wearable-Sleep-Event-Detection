#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=slurm/output/ahi_correction/%j.out

fold_nr=$1

echo "ahi correction"
echo ""
source ~/.bashrc
srun python wearsed/training/ahi_correction/train_ahi_correction.py --out-dir corr --fold-nr $fold_nr

echo "done"