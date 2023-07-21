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
from wass_example import gen_er_dists

vill_list_old = chain(range(12),range(13, 21),range(22, 77))

vill_list = [x+1 for x in vill_list_old]

for vill_no in vill_list:

    # our observed network
    stata_household = pd.read_stata('../banerjee_data/datav4.0/Data/2. Demographics and Outcomes/household_characteristics.dta')
    file = '../banerjee_data/datav4.0/Data/1. Network Data/Adjacency Matrices/adj_' + 'allVillageRelationships' +'_HH_vilno_' + str(vill_no) + '.csv'
    vill_mat = (pd.read_csv(file, header=None)).to_numpy()
    stata_vill = stata_household.where(stata_household['village'] == vill_no).dropna(how = 'all')
    room_type = np.array(stata_vill['room_no']/np.sqrt((stata_vill['bed_no']+1))<=2)
    room_type_dict = {k: int(v)*2 -1 for k, v in enumerate(room_type)}
    n = vill_mat.shape[0]

    vill_dist = calc_dists(vill_mat, room_type_dict)
    er_dist = gen_er_dists(1)

    err = emd(er_dist, vill_dist)

    print('Village ', str(vill_no), ': ', str(err))
