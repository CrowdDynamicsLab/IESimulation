#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --job-name="mele_fit7_a"
#SBATCH --partition=shared
#SBATCH --account=uic333
#SBATCH --mail-user=nainab2@illinois.edu
#SBATCH --mail-type=ALL

module load cpu/0.15.4
module load gcc/10.2.0
module load python/3.8.5
export PYTHONPATH=/home/$USER/scratch/mypython3:${PYTHONPATH}

python3 /home/$USER/scratch/IESimulation/mele_fitting.py
