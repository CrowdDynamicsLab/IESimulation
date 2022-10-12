import pandas as pd
import numpy as np
import networkx as nx

import model_comparison.mele as mele

from itertools import product
import multiprocessing as mp

def fit_mele_rand(params):

    alpha1 = np.round(params[0],3)
    alpha2 = np.round(params[1],3)
    alpha3 = np.round(params[2],3)
    alpha4 = np.round(params[3],3)
    beta1 = np.round(params[4],3)
    beta2 = np.round(params[5],3)
    gamma1 = np.round(params[6],3)
    gamma2 = np.round(params[7],3)
    gamma2 = np.round(params[8],3)

    alpha = [alpha1, alpha2, alpha3, alpha4]
    beta = [beta1, beta2]
    gamma = [gamma1, gamma2, gamma3]

    chars = np.concatenate((np.zeros(int(sum(room_type))), np.zeros(int(len(room_type)) - int(sum(room_type)))+1))

    G_m = mele.run_model(alpha, beta, gamma, chars)
    G_mnx = nx.from_numpy_matrix(G_m)
    type_dict2 = {k: v for k, v in enumerate(chars)}

    nx.set_node_attributes(G_mnx, type_dict2, "type")

    curr_tri_cnta = sum((nx.triangles(G_mnx)).values())/3

    curr_assorta = nx.attribute_assortativity_coefficient(G_mnx, "type")

    tri_lossa = (data_tri_cnt1-curr_tri_cnta)/data_tri_cnt1
    assort_lossa = (data_assort1 - curr_assorta)/2

    lossa = np.sqrt(tri_lossa**2 + assort_lossa**2)

    value = [str(lossa), '\n']

    metrics = [str(curr_tri_cnta), '\n', str(curr_assorta)]

    data_dir = 'mele_random_results'

    filename = '{odir}/{vill_no}_{a1}_{a2}_{a3}_{a4}_{b1}_{b2}_{g1}_{g2}_{g3}_losses.txt'.format(
        odir=data_dir, vill_no=str(vill_no), a1 =str(alpha1), a2 = str(alpha2), a3 = str(alpha3), a4 = str(alpha4), b1 = str(beta1), b2 = str(beta2), g1 = str(gamma1), g2 = str(gamma2), g3 = str(gamma3))

    with open(filename, 'w') as f:
        f.writelines(value)
        f.writelines(metrics)
    f.close()

# we fit village 6 because it is of small size and our model fit it well
# also 7 because small and worse fit
# we only take edge set 1
vill_no = 6
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

paramlist = []

for i in range(100):
    alpha1 = np.random.uniform(-1, 1)
    alpha2 = np.random.uniform(-1, 1)
    alpha3 = np.random.uniform(-1, 1)
    alpha4 = np.random.uniform(-1, 1)
    beta1 = np.random.uniform(-1, 1)
    beta2 = np.random.uniform(-1, 1)
    gamma1 = np.random.uniform(-1, 1)
    gamma2 = np.random.uniform(-1, 1)
    gamma3 = np.random.uniform(-1, 1)

    paramlist.append((alpha1, alpha2, alpha3, alpha4, beta1, beta2, gamma1, gamma2, gamma3))

if __name__ == '__main__':

    # create a process pool that uses all cpus
    with mp.Pool(processes = 32) as pool:
        # call the function for each item in parallel
        pool.map(fit_mele_rand, paramlist)
