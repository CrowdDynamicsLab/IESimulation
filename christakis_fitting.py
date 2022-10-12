import pandas as pd
import numpy as np
import networkx as nx

import model_comparison.christakis as chris

from itertools import product
import multiprocessing as mp

def fit_chris(params):

    b0 = params[0]
    b1 = params[1]
    om = params[2]
    a1 = params[3]
    a2 = params[4]
    a3 = params[5]
    a4 = params[6]

    theta = (b0, b1, om, [a1, a2, a3, a4])

    X = np.concatenate((np.zeros(int(sum(room_type))), np.zeros(int(len(room_type)) - int(sum(room_type)))+1))

    tri_cnt_arr = []
    assort_arr = []
    for i in range(5):
        Eps = np.random.gumbel(size = (len(X), len(X)))

        G_c = chris.run_sim(X, theta, Eps)

        G_cnx = nx.from_numpy_matrix(G_c)
        type_dict = {k: v for k, v in enumerate(X)}
        nx.set_node_attributes(G_cnx, type_dict, "type")

        curr_tri_cnt = sum((nx.triangles(G_cnx)).values())/3
        tri_cnt_arr.append(curr_tri_cnt)
        #print(curr_tri_cnt)
        curr_assort = nx.attribute_assortativity_coefficient(G_cnx, "type")
        assort_arr.append(curr_assort)
        #print(curr_assort)

    avg_tri_cnt = np.mean(tri_cnt_arr)
    avg_assort = np.mean(assort_arr)

    tri_loss = (data_tri_cnt1-avg_tri_cnt)/data_tri_cnt1
    assort_loss = (data_assort1 - avg_assort)/2

    loss = np.sqrt(tri_loss**2 + assort_loss**2)

    value = [str(loss), '\n']

    metrics = [str(avg_tri_cnt), '\n', str(avg_assort)]

    data_dir = 'christakis_results'

    filename = '{odir}/{vill_no}_{b0}_{b1}_{om}_{a1}_{a2}_{a3}_{a4}_losses.txt'.format(
        odir=data_dir, vill_no=str(vill_no), b0 =str(b0), b1 = str(b1), om = str(om), a1 = str(a1), a2 = str(a2), a3 = str(a3), a4 = str(a4))

    with open(filename, 'w') as f:
        f.writelines(value)
        f.writelines(metrics)
    f.close()

# we fit village 6 because it is of small size and our model fit it well
# also 7 because small and worse fit
# we only take edge set 1
vill_no = 7
money_hyp_files = ['borrowmoney', 'lendmoney', 'keroricecome', 'keroricego']

stata_household = pd.read_stata('banerjee_data/datav4.0/Data/2. Demographics and Outcomes/household_characteristics.dta')

# choose village and label with normalized room type
stata_vill = stata_household.where(stata_household['village'] == vill_no).dropna(how = 'all')
room_type = np.array(stata_vill['room_no']/np.sqrt((stata_vill['bed_no']+1))<=2)

old_file1 = 'banerjee_data/datav4.0/Data/1. Network Data/Adjacency Matrices/adj_' + money_hyp_files[0] +'_HH_vilno_' + str(vill_no) + '.csv'
ad_mat_old1 = (pd.read_csv(old_file1, header=None)).to_numpy()

for file in money_hyp_files[1:]:
    new_file1 = 'banerjee_data/datav4.0/Data/1. Network Data/Adjacency Matrices/adj_' + file +'_HH_vilno_' + str(vill_no) + '.csv'
    ad_mat_new1= (pd.read_csv(new_file1, header=None)).to_numpy()

    ad_mat_old1 = np.bitwise_or(ad_mat_old1, ad_mat_new1)

ad_mat_np1 = ad_mat_old1


matrix_key_filename = 'banerjee_data/datav4.0/Data/1. Network Data/Adjacency Matrix Keys/key_HH_vilno_' + str(vill_no) + '.csv'
ad_mat_key_vill = np.array(pd.read_csv(matrix_key_filename, header = None))

# confirm households in every village are actually labelled 1...h
assert(np.array_equal(ad_mat_key_vill.flatten(),stata_vill['HHnum_in_village'].values))

G_nx_data1 = nx.from_numpy_matrix(ad_mat_np1)

data_type_dict = {k: v for k, v in enumerate(room_type)}
nx.set_node_attributes(G_nx_data1, data_type_dict, "type")

# values to aim for
data_tri_cnt1 = sum((nx.triangles(G_nx_data1)).values())/3
data_assort1 = nx.attribute_assortativity_coefficient(G_nx_data1, "type")

# best loss was  0.01774770433614083
# for theta equals  (-1.1812111471290718, 0.1682709034133562, 0.08439179247757905, [-0.10016339241033151, -0.00259437014190718, 1.9263189421520241, 1.2667035584898267])

#best loss was  0.20807938075597165
#for theta equals  (-0.8225545718040959, 0.11217709835166051, 0.253450574997081, [-0.1908810055180419, 0.0010069741108848411, 3.4925360045274676, 1.734881759470818])

b0_list = [-.7, -.8, -.9]
b1_list = [0, .1, .2]
om_list = [.2, .3, .4]
a1_list = [-.3, -.2, -.1]
a2_list = [-.1, 0, .1]
a3_list = [3.4, 3.5, 3.6]
a4_list = [1.6, 1.7, 1.8]


paramlist = list(product(b0_list, b1_list, om_list, a1_list, a2_list, a3_list, a4_list))

if __name__ == '__main__':

    # create a process pool that uses all cpus
    with mp.Pool(processes = 32) as pool:
        # call the function for each item in parallel
        pool.map(fit_chris, paramlist)
