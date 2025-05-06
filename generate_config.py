import numpy as np

np.random.seed(42)
seeds = np.random.choice(np.arange(1, 10000), size=20, replace=False)

for s in seeds:
    with open(f"training_bash/{s}.sh", "w") as f:
        f.write(f"""#!/bin/bash

#SBATCH --mem=32G
#SBATCH --partition gpu
#SBATCH --gpus=h100:1
#SBATCH --job-name=rs{s}
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL

module load miniconda
conda activate the
python train.py --output_dir model_{s} --random_seed {s}
        """)


with open(f"run_all.sh", "w") as f:
    for s in seeds:
        f.write(f"sbatch training_bash/{s}.sh"+"\n")
        