#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name="village_fit_finer_ntwks"
#SBATCH --partition=cs
#SBATCH --mail-user=nainab2@illinois.edu
#SBATCH --mail-type=ALL

module load gcc
module load python/3
export PYTHONPATH=/home/$USER/scratch/mypython3a:${PYTHONPATH}

python3 /home/$USER/scratch/IESimulation/village_fitting.py
