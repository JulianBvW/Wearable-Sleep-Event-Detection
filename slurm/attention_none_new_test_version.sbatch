#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output/018_GPU_attention_none_new_test_version_%j.out

echo "GPU attention none new test version"
echo ""
source ~/.bashrc
srun python wearsed/training/attention_unet/train_attention_unet.py --out-dir attention_GPU_none_new_test_version --use-attention none

echo "done"