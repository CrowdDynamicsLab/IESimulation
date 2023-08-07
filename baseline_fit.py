import pandas as pd
import numpy as np
import networkx as nx
import model_comparison.christakis as chris
import model_comparison.mele as mele
from comparison.calc_wass import calc_dists, avg_dists
from comparison.wasserstein import emd
from itertools import product

vill_no = 10

stata_household = pd.read_stata('banerjee_data/datav4.0/Data/2. Demographics and Outcomes/household_characteristics.dta')
file = 'banerjee_data/datav4.0/Data/1. Network Data/Adjacency Matrices/adj_' + 'allVillageRelationships' +'_HH_vilno_' + str(vill_no) + '.csv'
vill_mat = (pd.read_csv(file, header=None)).to_numpy()
stata_vill = stata_household.where(stata_household['village'] == vill_no).dropna(how = 'all')
room_type = np.array(stata_vill['room_no']/np.sqrt((stata_vill['bed_no']+1))<=2)
room_type_dict = {k: int(v)*2 -1 for k, v in enumerate(room_type)}

vill_dist = calc_dists(vill_mat, room_type_dict)

X = np.concatenate((np.zeros(int(sum(room_type))), np.zeros(int(len(room_type)) - int(sum(room_type)))+1))
Eps = np.random.gumbel(size = (len(X), len(X)))

min_loss = np.inf
best_theta = None

for i in range(200):
    b0 = np.random.uniform(-0.45,-0.25)
    b1 = np.random.uniform(0.35, 0.55)
    om = np.random.uniform(0.2, 0.4)
    a0 = np.random.uniform(-0.2, 0)
    a1 = np.random.uniform(0, 0.004)
    a2 = np.random.uniform(2.68, 2.88)
    a3 = np.random.uniform(.19, .39)

    #best loss was  0.13501958722619897
    #for theta equals  (-0.36497042695419823, 0.4691771787764718, 0.3047811536839999, [-0.11666825556700311, 0.002019871018998638, 2.780372134048425, 0.288517123479205])


    theta = (b0, b1, om, [a0, a1, a2, a3])
    #print(theta)

    G_c = chris.run_sim(X, theta, Eps)
    type_dict = {k: v for k, v in enumerate(X)}

    chris_dist = calc_dists(G_c, type_dict)

    loss = emd(chris_dist, vill_dist)

    #print(loss)
    #print('\n')

    if loss < min_loss:
        min_loss = loss
        best_theta = theta

print('best christakis, vill 10 loss was ', min_loss)
print('for theta equals ', best_theta)
#print('\n')
#print(G_c)
#print('\n')
#print(type_dict)
