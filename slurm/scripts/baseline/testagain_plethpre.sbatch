#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/ta/plethpre_%j.out

nn=$1

echo "plethpre test again"
echo ""
source ~/.bashrc
srun python wearsed/training/baseline_conv/train_baseline_conv_plethpre.py --out-dir testagain_plethpre_$nn --use-pleth-statistical --use-pleth-vae-latents

echo "done"