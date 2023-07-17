# test by generating two random graphs then compare them

import math
import random

import numpy as np

from wasserstein import emd

def gen_er_dists(size):
    # edge prob
    p = math.log(size) / size
    
    # assume types are half/half
    adj_mat = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            if random.random() <= p:
                adj_mat[i][j] = 1
                adj_mat[j][i] = 1

    # get the assort and cluster coeff
    opairs = []

    # useful constants
    # red in first floor(size / 2), blue in second half
    half = math.floor(size/2)
    max_edges = (size - 1) * (size - 2) / 2
    three_walks = (adj_mat @ adj_mat) @ adj_mat

    for i in range(size):
        deg = sum(adj_mat[i])

        assort = 0
        if i <= half:
            assort = sum(adj_mat[i][:half])
        else:
            assort = sum(adj_mat[i][half:])

        # not really realistic to our model but fine for the test
        if deg == 0:
            assort = 0
        else:
            assort = assort / deg

        # calculate cluster coeffs
        ccoeff = 0
        if deg <= 1:
            ccoeff = 0
        else:

            # triangles will be double counted
            ccoeff = (three_walks[i][i] / 2) / (deg * (deg - 1))

        opairs.append((assort, ccoeff))

    dist_map = { }
    for op in opairs:
        if op in dist_map:
            dist_map[op] += 1
        else:
            dist_map[op] = 1

    dist = []
    for op, freq in dist_map.items():
        dist.append( (op, freq / size) )

    return dist

dist1 = gen_er_dists(250)
dist2 = gen_er_dists(250)

print('EMD', emd(dist1, dist2))

