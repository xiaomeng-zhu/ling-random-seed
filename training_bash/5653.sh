#!/bin/bash

#SBATCH --mem=32G
#SBATCH --partition gpu
#SBATCH --gpus=h100:1
#SBATCH --job-name=rs5653
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL

module load miniconda
conda activate the
python train.py --output_dir model_5653 --random_seed 5653
        