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
    b0 = np.random.uniform(-0.3,-0.1)
    b1 = np.random.uniform(0.1, 0.3)
    om = np.random.uniform(-0.1, 0.1)
    a0 = np.random.uniform(-0.1, 0.1)
    a1 = np.random.uniform(-.005, 0)
    a2 = np.random.uniform(2.7, 2.9)
    a3 = np.random.uniform(.7, .9)

    #best loss was 0.12838809232727344 for theta equals
    #(-0.21274401387580744, 0.19132708201097165, 0.02976315818095765, [-0.025178638000684483, -0.003974950214016679, 2.7680309300119696, 0.8479582084455608])


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
