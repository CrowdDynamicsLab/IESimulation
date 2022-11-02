import pandas as pd
import numpy as np
import networkx as nx

import model_comparison.mele as mele

from itertools import product
import multiprocessing as mp

def fit_mele(params):

    alpha1 = params[0]
    alpha2 = params[1]
    alpha3 = params[2]
    alpha4 = params[3]
    beta1 = params[4]
    beta2 = params[5]
    gamma1 = params[6]
    gamma2 = params[7]
    gamma3 = params[8]

    alpha = [alpha1, alpha2, alpha3, alpha4]
    beta = [beta1, beta2]
    gamma = [gamma1, gamma2, gamma3]

    chars = np.concatenate((np.zeros(int(sum(room_type))), np.zeros(int(len(room_type)) - int(sum(room_type)))+1))

    tri_cnt_arr = []
    assort_arr = []
    edges_arr = []
    for i in range(5):

        G_m = mele.run_model(alpha, beta, gamma, chars)
        G_mnx = nx.from_numpy_matrix(G_m)
        type_dict2 = {k: v for k, v in enumerate(chars)}

        nx.set_node_attributes(G_mnx, type_dict2, "type")

        curr_tri_cnta = sum((nx.triangles(G_mnx)).values())/3
        tri_cnt_arr.append(str(curr_tri_cnta))
        tri_cnt_arr.append(', ')

        curr_edgesa = G_mnx.number_of_edges()
        edges_arr.append(str(curr_edgesa))
        edges_arr.append(', ')

        curr_assorta = nx.attribute_assortativity_coefficient(G_mnx, "type")
        assort_arr.append(str(curr_assorta))
        assort_arr.append(', ')

    #avg_tri_cnt = np.mean(tri_cnt_arr)
    #avg_assort = np.mean(assort_arr)

    #tri_loss = (data_tri_cnt1-avg_tri_cnt)/data_tri_cnt1
    #assort_loss = (data_assort1 - avg_assort)/2

    #loss = np.sqrt(tri_loss**2 + assort_loss**2)

    #value = [str(loss), '\n']

    #metrics = [str(avg_tri_cnt), '\n', str(avg_assort)]

    tri_cnt_arr.append(str('\n'))
    assort_arr.append(str('\n'))
    edges_arr.append(str('\n'))

    data_dir = 'mele_results'

    filename = '{odir}/{vill_no}_{a1}_{a2}_{a3}_{a4}_{b1}_{b2}_{g1}_{g2}_{g3}_losses.txt'.format(
        odir=data_dir, vill_no=str(vill_no), a1 =str(alpha1), a2 = str(alpha2), a3 = str(alpha3), a4 = str(alpha4), b1 = str(beta1), b2 = str(beta2), g1 = str(gamma1), g2 = str(gamma2), g3 = str(gamma3))

    with open(filename, 'w') as f:
        f.writelines(tri_cnt_arr)
        f.writelines(assort_arr)
        f.writelines(edges_arr)
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

alpha1_list = [1.3, 1.4]
alpha2_list = [.7, .8]
alpha3_list = [1.4,1.5]
alpha4_list = [2.5, 2.6]
beta1_list = [1.1,1.2]
beta2_list = [2.5,2.6]
gamma1_list = [.7,.8]
gamma2_list = [-.5, -.4]
gamma3_list = [.2, .3]

paramlist = list(product(alpha1_list, alpha2_list, alpha3_list, alpha4_list, beta1_list, beta2_list, gamma1_list, gamma2_list, gamma3_list))
if __name__ == '__main__':

    # create a process pool that uses all cpus
    with mp.Pool(processes = 32) as pool:
        # call the function for each item in parallel
        pool.map(fit_mele, paramlist)
