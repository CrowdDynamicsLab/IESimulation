#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name="comp_proposal"
#SBATCH --partition=cs
#SBATCH --mail-user=andy2@illinois.edu
#SBATCH --output=VAR-%x.%j.out
#SBATCH --error=VAR-%x.%j.err

module load cpu/0.15.4
module load gcc/10.2.0
module load python/3.8.5
export PYTHONPATH=/home/$USER/scratch/mypython3:${PYTHONPATH}

gridvals=(0.375 0.875) 

(for sc in ${gridvals[@]}; do (
    (for ho in ${gridvals[@]}; do (
        python3 /home/$USER/scratch/IESimulation/st_var.py 150 10 $sc $ho
    ) done)
) done)

