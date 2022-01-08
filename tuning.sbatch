#!/bin/bash
#SBATCH --job-name="Meet-U Parameter Tuning"
#SBATCH --partition=long
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=16

## Tune BananaTAD
srun sh tuning.sh
