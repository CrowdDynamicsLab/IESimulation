# test by generating two random graphs then compare them

import math
import random

import numpy as np

from wasserstein import emd

sim_dist = []
vill_dist = []

print('sim data')
with open('data/avg_sims.txt', 'r') as ast:
    total_freq = 0
    for l in ast:
        trimmed = l[1:-1]
        assort, ccoeff, freq = trimmed.split(',')
        assort = float(assort[1:])
        ccoeff = float(ccoeff[1:-1])
        freq = float(freq[1:-1])
        sim_dist.append( ((assort, ccoeff), freq) )

        total_freq += freq

    print('num pairs', len(sim_dist))
    print('total freq', total_freq)

print('vill data')
with open('data/vill.txt', 'r') as vt:
    total_freq = 0
    for l in vt:
        trimmed = l[1:-1]
        assort, ccoeff, freq = trimmed.split(',')
        assort = float(assort[1:])
        ccoeff = float(ccoeff[1:-1])
        freq = float(freq[1:-1])
        vill_dist.append( ((assort, ccoeff), freq) )

        total_freq += freq

    print('num pairs', len(vill_dist))
    print('total freq', total_freq)

print('EMD', emd(sim_dist, vill_dist))
