#!/bin/bash

gridvals=(0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1) 
nvals=(500 1000)
kvals=(10 150)

(for sc in ${gridvals[@]}; do (
    (for ho in ${gridvals[@]}; do (
        sbatch sim_part.sbatch 100 8 $sc $ho
    ) done)
) done)

sbatch --dependency=singleton --job-name=st_comp end_dummy.sbatch

