#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --job-name="test_fixed"
#SBATCH --partition=shared
#SBATCH --account=uic333
#SBATCH --mail-user=andy2@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --output=TFIXED-%x.%j.out
#SBATCH --error=TFIXED-%x.%j.err

module load cpu/0.15.4
module load gcc/10.2.0
module load python/3.8.5
export PYTHONPATH=/home/$USER/scratch/mypython3:${PYTHONPATH}

gridvals=(0 1) 

(for sc in ${gridvals[@]}; do (
    (for ho in ${gridvals[@]}; do (
        python3 /home/$USER/scratch/IESimulation/st_fixed.py 50 5 $sc $ho
    ) done)
) done)

