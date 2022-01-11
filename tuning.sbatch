#!/bin/bash
#SBATCH --job-name="Meet-U Parameter Tuning"
#SBATCH --partition=long
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=16
#SBATCH --account=form_2021_21

# Activate environment
# conda activate meetu

## Tune BananaTAD
srun sh tuning.sh
