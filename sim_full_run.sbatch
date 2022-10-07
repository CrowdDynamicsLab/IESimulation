#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name="gen_comp_proposal"
#SBATCH --partition=shared
#SBATCH --account=uic333
#SBATCH --mail-user=andy2@illinois.edu
#SBATCH --mail-type=ALL

module load cpu/0.15.4
module load gcc/10.2.0
module load python/3.8.5
export PYTHONPATH=/home/$USER/scratch/mypython3:${PYTHONPATH}

gridvals=(0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1) 

(for sc in ${gridvals[@]}; do (
    (for ho in ${gridvals[@]}; do (
        python3 /home/$USER/scratch/IESimulation/st_comparison.py 150 10 $sc $ho
    ) done)
) done)

