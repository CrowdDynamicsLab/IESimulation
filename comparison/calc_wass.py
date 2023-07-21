# calculate the wasserstein distance between simulation and observed distributions

import json

import numpy as np

from wasserstein import emd

# Put this file for the simulation output here
sim_file = 'data/1_10_0.0_0.0_losses.txt'

# enter this manually for now
kappa = 10

# local assortativity https://iopscience.iop.org/article/10.1209/0295-5075/89/49901/pdf

# naina code
f = open(sim_file,'r')
ntwk = f.readline()
attrs = f.readline()
ntwk = ntwk.replace('[', '')
ntwk = ntwk.replace(']', '')
ntwk = ntwk.replace("'", '')
num_nodes = int(np.sqrt((ntwk.count('0.0') + ntwk.count('1.0'))/5))
ntwk_arr = np.array([int(float(e)) for e in ntwk.split(',')])
arr1 = ntwk_arr[:num_nodes**2].reshape((num_nodes, num_nodes))
arr2 = ntwk_arr[num_nodes**2:2*num_nodes**2].reshape((num_nodes, num_nodes))
arr3 = ntwk_arr[2*num_nodes**2:3*num_nodes**2].reshape((num_nodes, num_nodes))
arr4 = ntwk_arr[3*num_nodes**2:4*num_nodes**2].reshape((num_nodes, num_nodes))
arr5 = ntwk_arr[4*num_nodes**2:5*num_nodes**2].reshape((num_nodes, num_nodes))

attrs = attrs.strip('{}\n')
pairs = attrs.split(',')
attr_dict = {int(key): int(value) for key, value in (pair.split(': ') for pair in pairs)}

def calc_dists(arr, attr_dict):

    opairs = []

    max_edges = (kappa - 1) * (kappa - 2) / 2
    three_walks = (arr @ arr) @ arr
    for v, vtype in attr_dict.items():
        deg = sum(arr[v])

        assort = 0
        for u, con in enumerate(arr[v]):
            if con and attr_dict[u] == attr_dict[v]:
                assort += 1
            elif con and attr_dict[u] != attr_dict[v]:
                assort -= 1

        if deg == 0:
            assort = 0
        else:
            assort = assort / deg

        ccoeff = 0
        if deg <= 1:
            ccoeff = 0
        else:
            ccoeff = (three_walks[v][v] / 2) / (deg * (deg - 1))

        opairs.append( ( (assort + 1) / 2, ccoeff) )

    dist_map = {}
    for op in opairs:
        if op in dist_map:
            dist_map[op] += 1
        else:
            dist_map[op] = 1

    dist = []
    for op, freq in dist_map.items():
        dist.append( (op, freq / len(attr_dict)) )

    return dist

def avg_dists(dists):
    dist_freq = {}
    for d in dists:
        for op, freq in d:
            if op in dist_freq:
                dist_freq[op] += freq
            else:
                dist_freq[op] = freq
    for op, freq in dist_freq.items():
        dist_freq[op] = freq / len(dists)

    return [ (op, freq) for op, freq in dist_freq.items() ]

#arr1_dist = calc_dists(arr1)
#all_dists = [ calc_dists(a) for a in [arr1, arr2, arr3, arr4, arr5] ]

#avg_dist = avg_dists(all_dists)

#print('EMD', emd(avg_dist, arr1_dist))
