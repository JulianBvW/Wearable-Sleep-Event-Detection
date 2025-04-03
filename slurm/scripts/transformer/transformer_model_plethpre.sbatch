#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/transformer/plethpre_%j.out

fold_nr=$1

echo "transformer model with pleth preprocessed"
echo ""
source ~/.bashrc
srun python wearsed/training/transformer_model/train_transformer_model_plethpre.py --out-dir plethpre --fold-nr $fold_nr --seq-length 600

echo "done"