#!/bin/bash

gridvals=(0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1) 

(for sc in ${gridvals[@]}; do (
    (for ho in ${gridvals[@]}; do (
        sbatch sim_part.sbatch $sc $ho
    ) done)
) done)

