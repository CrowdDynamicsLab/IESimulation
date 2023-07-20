from itertools import chain
import os
from os.path import exists
import numpy as np
import re
import json
import networkx as nx
from scipy.stats import ks_2samp as ks
import pandas as pd
from calc_wass import calc_dists, avg_dists
from wasserstein import emd
import matplotlib.pyplot as plt

vill_list_old = chain(range(12),range(13, 21),range(22, 77))

vill_list = [x+1 for x in vill_list_old]

#vill_list = [11]

sc_old = range(17)
ho_old = range(17)

sc_list = [x/16 for x in sc_old]
ho_list = [x/16 for x in ho_old]

k = 10

for vill_no in vill_list:

    # our observed network
    stata_household = pd.read_stata('../banerjee_data/datav4.0/Data/2. Demographics and Outcomes/household_characteristics.dta')
    file = '../banerjee_data/datav4.0/Data/1. Network Data/Adjacency Matrices/adj_' + 'allVillageRelationships' +'_HH_vilno_' + str(vill_no) + '.csv'
    vill_mat = (pd.read_csv(file, header=None)).to_numpy()
    stata_vill = stata_household.where(stata_household['village'] == vill_no).dropna(how = 'all')
    room_type = np.array(stata_vill['room_no']/np.sqrt((stata_vill['bed_no']+1))<=2)
    room_type_dict = {k: int(v)*2 -1 for k, v in enumerate(room_type)}

    best_sc = -1
    best_ho = -1
    best_emd = np.inf

    for sc in sc_list:
        for ho in ho_list:
            data_dir = '../finer_results_ntwk'
            filename = '{odir}/{vill_no}_{k}_{sc}_{ho}_losses.txt'.format(
                odir=data_dir, vill_no=str(vill_no), k = str(k), sc = str(sc), ho = str(ho))

            f = open(filename, "r")

            # nonsense code for reading the simulated networks and attrs
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

            sim_dists = [ calc_dists(a, attr_dict) for a in [arr1, arr2, arr3, arr4, arr5] ]
            avg_sim_dists = avg_dists(sim_dists)
            vill_dist = calc_dists(vill_mat, room_type_dict)


            # with open('avg_sims2.txt', 'w') as f:
            #     for line in avg_sim_dists:
            #         f.write(f"{line}\n")
            # f.close()
            #
            # with open('vill2.txt', 'w') as f:
            #     for line in vill_dist:
            #         f.write(f"{line}\n")
            # f.close()


            curr_emd = emd(avg_sim_dists, vill_dist)


            if curr_emd < best_emd:
                best_emd = curr_emd
                best_sc = sc
                best_ho = ho

    print('The best emd for village ', str(vill_no), ' is ', str(best_emd), ' for sc and ho ', str(best_sc), ', ', str(best_ho))
