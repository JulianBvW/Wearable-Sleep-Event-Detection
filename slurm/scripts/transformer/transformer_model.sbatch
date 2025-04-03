#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/transformer/tf-short-_%j.out

fold_nr=$1

echo "transformer model first test with 10min"
echo ""
source ~/.bashrc
srun python wearsed/training/transformer_model/train_transformer_model.py --out-dir first --fold-nr $fold_nr --seq-length 600

echo "done"