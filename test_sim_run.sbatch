#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --job-name="st_comp_full"
#SBATCH --partition=cs
#SBATCH --mail-user=andy2@illinois.edu
#SBATCH --mail-type=ALL

module load python/3
export PYTHONPATH=/home/$USER/scratch/mypython3b:${PYTHONPATH}

gridvals=(0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1) 
nvals=(500 1000)
kvals=(10 150)

(for sc in ${gridvals[@]}; do (
    (for ho in ${gridvals[@]}; do (
        python3 /home/$USER/scratch/IESimulation/st_comparison.py 20 5 $sc $ho
    ) done)
) done)

