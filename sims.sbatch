#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --job-name="max_30"
#SBATCH --partition=cs
#SBATCH --output=max_30.o%j
#SBATCH --mail-user=nainab2@illinois.edu
#SBATCH --mail-type=ALL

module load python/3
export PYTHONPATH=/home/$USER/scratch/mypython3b:${PYTHONPATH}
python3 /home/nainab2/scratch/IESimulation/cluster_sims.py
