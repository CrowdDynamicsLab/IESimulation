#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --job-name="st budgetless"
#SBATCH --partition=cs
#SBATCH --output=max_30.o%j
#SBATCH --mail-user=andy2@illinois.edu
#SBATCH --mail-type=ALL

module load python/3
export PYTHONPATH=/home/$USER/scratch/mypython3b:${PYTHONPATH}

set sc_arr 0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1 
set ho_arr 0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1 

(for %%sc in (%sc_arr%) do (
    (for %%ho in (%ho_arr%) do (
        python3 /home/$USER/scratch/IESimulation/st_budgetless.py sc ho
    ))
))
