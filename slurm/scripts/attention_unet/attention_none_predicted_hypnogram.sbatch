#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/pred_hypno/pred_hypno_%j.out

fold_nr=$1

echo "attention none with predicted hypnogram"
echo ""
source ~/.bashrc
srun python wearsed/training/attention_unet/train_attention_unet.py --out-dir predicted_hypnogram --use-attention none --fold-nr $fold_nr --use-predicted-hypnogram

echo "done"