import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import community as community_louvain
from os.path import exists

import sim_lib.util as util
import sim_lib.attr_lib.util as alu
from sim_lib.attr_lib.formation import *
from math import isclose
from itertools import product
from itertools import combinations
from itertools import chain
import json
import sys
import os
import copy
import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")

_FIXED_ITERS = -1

satisfice = 1
num_iters = 500
min_iters = 10
st_count_track = 10
st_count_dev_tol = 0.01

def is_symmetric(A, tol=1e-8):
    return np.linalg.norm(A-A.T, np.Inf) < tol

########## functions ###########

def get_component_sizes(G):
    G_nx = alu.graph_to_nx(G)
    G_nx_comp_nodes = list(nx.connected_components(G_nx))
    G_nx_largest = G_nx.subgraph(max(G_nx_comp_nodes, key=len))
    G_nx_comps = [ G_nx.subgraph(G_nxc_nodes) for G_nxc_nodes in G_nx_comp_nodes ]
    component_sizes = [ len(G_nxc) for G_nxc in G_nx_comps ]
    return component_sizes

def get_summary_stats(G):
    num_components = len(get_component_sizes(G))

    avg_deg = np.mean([ v.degree for v in G.vertices ])
    avg_util = np.mean([v.data['struct_util'](v, G) + v.data['total_attr_util'](v,G) for v in G.vertices ])
    avg_cost = np.mean([alu.calc_cost(v, G) for v in G.vertices ])

    # Metric calc
    g_nx = alu.graph_to_nx(G)

    partition = {}
    partition = community_louvain.best_partition(g_nx)

    num_comm = max(partition.values())
    if g_nx.number_of_edges() > 0:
        modularity = community_louvain.modularity(partition, g_nx)
    else:
        modularity = -1
    num_comp = num_components
    comp_apl = np.mean([ nx.average_shortest_path_length(g_nx.subgraph(c))
        for c in nx.connected_components(g_nx) ])
    cluster_coeff = nx.average_clustering(g_nx)
    stable_triad_count = count_stable_triads(G)

    triangle_count = sum((nx.triangles(g_nx)).values())/3
    assortativity = nx.attribute_assortativity_coefficient(g_nx, "shape")
    edge_count = g_nx.number_of_edges()

    return {
        'degree' : avg_deg,
        'util' : avg_util,
        'cost' : avg_cost,
        'num_comm' : num_comm,
        'modularity' : modularity,
        'num_comp' : num_comp,
        'apl' : comp_apl,
        'cluster_coeff' : cluster_coeff,
        'stable_triad_count' : stable_triad_count,
        'triangle_count': triangle_count,
        'assortativity': assortativity,
        'edge_count': edge_count
    }

def add_sum_stat(st_dict, res):
    st_dict['degree_dist'].append(res['degree'])
    st_dict['util_dist'].append(res['util'])
    st_dict['cost_dist'].append(res['cost'])
    st_dict['num_comm'].append(res['num_comm'])
    st_dict['modularity'].append(res['modularity'])
    st_dict['num_comp'].append(res['num_comp'])
    st_dict['apl'].append(res['apl'])
    st_dict['cluster_coeff'].append(res['cluster_coeff'])
    st_dict['stable_triad_count'].append(res['stable_triad_count'])
    st_dict['triangle_count'].append(res['triangle_count'])
    st_dict['assortativity'].append(res['assortativity'])
    st_dict['edge_count'].append(res['edge_count'])


def count_stable_triads(G):
    num_stable_triad = 0
    num_em_ho_st = 0
    num_sc_ho_st = 0
    num_sc_he_st = 0
    for triad in combinations(G.vertices, 3):
        attr_funcs = [ t.data['attr'] for t in triad ]
        if len(set(attr_funcs)) != 1:
            continue

        struct_funcs = [ t.data['struct'] for t in triad ]
        if len(set(struct_funcs)) != 1:
            continue

        if triad[0].data['struct'] == 'em' and triad[0].data['attr'] == 'ho':

            # Homophily so all same type
            if len(set([ t.data['type_name'] for t in triad ])) != 1:
                continue

            # Triangle
            if G.are_neighbors(triad[0], triad[1]) and G.are_neighbors(triad[0], triad[2]) \
                    and G.are_neighbors(triad[1], triad[2]):
                num_em_ho_st += 1
        elif triad[0].data['struct'] == 'sc' and triad[0].data['attr'] == 'ho':

            # Homophily all same type
            if len(set([ t.data['type_name'] for t in triad ])) != 1:
                continue

            # Exactly two edges
            if sum([ G.are_neighbors(p[0], p[1]) for p in combinations(triad, 2) ]) == 2:
                num_sc_ho_st += 1
        elif triad[0].data['struct'] == 'sc' and triad[0].data['attr'] == 'he':

            # Heterophily so not all same type
            if len(set([ t.data['type_name'] for t in triad ])) == 1:
                continue

            # Exactly two edges
            edge_pairs = []
            for pair in combinations(triad, 2):
                if G.are_neighbors(pair[0], pair[1]):
                    edge_pairs.append(pair)

            if len(edge_pairs) != 2:
                continue
            if edge_pairs[0][0].data['type_name'] != edge_pairs[0][1].data['type_name'] and \
                    edge_pairs[1][0].data['type_name'] != edge_pairs[1][1].data['type_name']:
                num_sc_he_st += 1

    return num_em_ho_st + num_sc_ho_st + num_sc_he_st

def type_dict(context, shape, context_p, attr, struct, sc_likelihood, ho_likelihood):
    likelihood = context_p
    if struct == 'em':
        struct_func = alu.triangle_count
        likelihood = likelihood * (1 - sc_likelihood)
    else:
        struct_func = alu.num_disc_nbors
        likelihood = likelihood * sc_likelihood
    if attr == 'ho':
        attr_edge_func = alu.homophily
        likelihood = likelihood * ho_likelihood
    else:
        attr_edge_func = alu.heterophily
        likelihood = likelihood * (1 - ho_likelihood)

    #Base color is a rgb list
    base_dict = {'likelihood' : likelihood,
              'struct_util' : struct_func,
              'struct' : struct,
              'init_attrs' : context,
              'attr' : attr,
              'edge_attr_util' : attr_edge_func,
              'total_attr_util' : alu.agg_attr_util,
              'optimistic' : False,
              #'color' : 'rgb({rgb})'.format(rgb=', '.join([ str(c) for c in color ])),
              'shape' :  shape
              #'{shape}'.format(shape=', '.join([str(s) for s in shape]))
              }
    return base_dict

def run_sim(sc_likelihood, ho_likelihood, max_clique_size, ctxt_likelihood, _N, sim_iters, sub=False):
    ctxt_types = [-1, 1]
    #ctxt_base_colors = [[43, 98, 166], [161, 39, 45]]
    ctxt_base_shapes = [0 , 2]
    ctxt_p = [ctxt_likelihood, 1-ctxt_likelihood]
    struct_types = ['em', 'sc']
    attr_types = ['ho', 'he']
    type_itr = [ (ctxt, shape, ct_p, at, st) for (ctxt, shape, ct_p) in zip(ctxt_types, ctxt_base_shapes, ctxt_p)
                for (at, st) in [(a, s) for a in attr_types for s in struct_types] ]
    type_list = [type_dict(*t_args, sc_likelihood, ho_likelihood) for t_args \
                  in type_itr ]

    type_counts = [ int(np.floor(_N * tl['likelihood'])) for tl in type_list ]

    remaining_tc = _N - sum(type_counts)
    for i in range(int(remaining_tc)):
        type_counts[i] = type_counts[i] + 1
    assert sum(type_counts) == _N, 'Did that work?'

    tc_dict = { f'type{idx}' : tc for idx, tc in enumerate(type_counts) }
    vtx_types = { f'type{idx}' : tl for idx, tl in enumerate(type_list) }

    params = {
        'context_count' : 2, # Needed for simple utility
        'k' : 1, # Needed for simple attribute utility
        'edge_selection' : alu.seq_edge_sel_silent,
        'seed_type' : 'trivial', # Type of seed network
        'max_clique_size' : max_clique_size,
        'revelation_proposals' : alu.indep_revelation,
        'util_agg' : alu.linear_util_agg, # How to aggregate utility values
        'vtx_types' : vtx_types
    }

    vtx_types_list = np.array([ np.repeat(t, tc) for t, tc in tc_dict.items() ])
    vtx_types_list = np.hstack(vtx_types_list)
    #np.random.shuffle(vtx_types_list)
    params['type_assignment'] = { i : vtx_types_list[i] for i in range(_N) }

    type_assgn_copy = copy.deepcopy(params['type_assignment'])
    final_type_assignments = {}
    for i, ta in params['type_assignment'].items():
        final_type_assignments[i] = copy.deepcopy(vtx_types[ta])
        final_type_assignments[i].pop('struct_util', None)
        final_type_assignments[i].pop('edge_attr_util', None)
        final_type_assignments[i].pop('total_attr_util', None)

    assert math.isclose(sum([ t['likelihood'] for t in params['vtx_types'].values() ]), 1.0)

    summary_stats = {
        'degree_dist' : [],
        'util_dist' : [],
        'cost_dist' : [],
        'num_comm' : [],
        'modularity' : [],
        'num_comp' : [],
        'apl' : [],
        'cluster_coeff' : [],
        'stable_triad_count' : [],
        'exit_iter' : [num_iters] * sim_iters,
        'triangle_count': [],
        'assortativity':[],
        'edge_count':[]

    }

    summary_stats = {
        'standard' : copy.deepcopy(summary_stats)
    }

    final_networks = {
        'standard' : []
    }

    for si in range(sim_iters):

        # Create networks to be compared
        # Base case network
        G_std = attribute_network(_N, copy.deepcopy(params))

        st_counts = {
            'standard' : []
        }

        std_fin = False

        for it in range(num_iters):

            calc_edges(G_std)

            # If running fixed iters ignore stable triad checks
            if _FIXED_ITERS > 0 and it == _FIXED_ITERS - 1:
                break

            # Get all stable triad counts
            std_st_count = count_stable_triads(G_std)

            # If less than min number iterations has run, add and move on
            if len(st_counts['standard']) < st_count_track:
                st_counts['standard'].append(std_st_count)
                continue

            # Update all count arrays with current
            st_counts['standard'].pop(0)
            st_counts['standard'].append(std_st_count)

            # Check if base case has just terminated
            if np.std(st_counts['standard']) <= st_count_dev_tol and not std_fin:
                std_fin = True
                summary_stats['standard']['exit_iter'][si] = it
                add_sum_stat(summary_stats['standard'], get_summary_stats(G_std))
                final_networks['standard'].append(G_std.adj_matrix.tolist())

            if std_fin:
                break

    #print('k: ', max_clique_size, 'ho: ', ho_likelihood, 'sc: ', sc_likelihood)

    # Take mean of all summary stats
    for st, vs in summary_stats['standard'].items():
        summary_stats['standard'][st] = np.mean(vs)

    return summary_stats, final_networks, final_type_assignments

#k_loss_array1 = np.full((len(vill_list),len(max_clique_size_list)), np.inf)
#k_loss_array2 = np.full((len(vill_list),len(max_clique_size_list)), np.inf)
#k_loss_array3 = np.full((len(vill_list),len(max_clique_size_list)), np.inf)

def fit_village_data(params):

    vill_no = params[0]
    max_clique_size = params[1]
    sc_likelihood = params[2]
    ho_likelihood = params[3]

    sim_iters = 1

    stata_household = pd.read_stata('banerjee_data/datav4.0/Data/2. Demographics and Outcomes/household_characteristics.dta')


    # various edge sets
    money_hyp_files = ['borrowmoney', 'lendmoney', 'keroricecome', 'keroricego']
    trust_hyp_files = ['helpdecision', 'giveadvice', 'medic']
    trust_fact_files = ['visitcome','visitgo', 'templecompany']


    # choose village and label with normalized room type
    stata_vill = stata_household.where(stata_household['village'] == vill_no).dropna(how = 'all')
    room_type = np.array(stata_vill['room_no']/np.sqrt((stata_vill['bed_no']+1))<=2)

    # ad mat for money_hyp_files

    old_file1 = 'banerjee_data/datav4.0/Data/1. Network Data/Adjacency Matrices/adj_' + money_hyp_files[0] +'_HH_vilno_' + str(vill_no) + '.csv'
    ad_mat_old1 = (pd.read_csv(old_file1, header=None)).to_numpy()

    for file in money_hyp_files[1:]:
        new_file1 = 'banerjee_data/datav4.0/Data/1. Network Data/Adjacency Matrices/adj_' + file +'_HH_vilno_' + str(vill_no) + '.csv'
        ad_mat_new1= (pd.read_csv(new_file1, header=None)).to_numpy()

        ad_mat_old1 = np.bitwise_or(ad_mat_old1, ad_mat_new1)

    ad_mat_np1 = ad_mat_old1

    # ad mat for trust_hyp_files

    old_file2 = 'banerjee_data/datav4.0/Data/1. Network Data/Adjacency Matrices/adj_' + trust_hyp_files[0] +'_HH_vilno_' + str(vill_no) + '.csv'
    ad_mat_old2 = (pd.read_csv(old_file2, header=None)).to_numpy()

    for file in trust_hyp_files[1:]:
        new_file2 = 'banerjee_data/datav4.0/Data/1. Network Data/Adjacency Matrices/adj_' + file +'_HH_vilno_' + str(vill_no) + '.csv'
        ad_mat_new2= (pd.read_csv(new_file2, header=None)).to_numpy()

        ad_mat_old2 = np.bitwise_or(ad_mat_old2, ad_mat_new2)

    ad_mat_np2 = ad_mat_old2

    # ad mat for trust_fact_files

    old_file3 = 'banerjee_data/datav4.0/Data/1. Network Data/Adjacency Matrices/adj_' + trust_fact_files[0] +'_HH_vilno_' + str(vill_no) + '.csv'
    ad_mat_old3 = (pd.read_csv(old_file3, header=None)).to_numpy()

    for file in trust_fact_files[1:]:
        new_file3 = 'banerjee_data/datav4.0/Data/1. Network Data/Adjacency Matrices/adj_' + file +'_HH_vilno_' + str(vill_no) + '.csv'
        ad_mat_new3= (pd.read_csv(new_file3, header=None)).to_numpy()

        ad_mat_old3 = np.bitwise_or(ad_mat_old3, ad_mat_new3)

    ad_mat_np3 = ad_mat_old3

    matrix_key_filename = 'banerjee_data/datav4.0/Data/1. Network Data/Adjacency Matrix Keys/key_HH_vilno_' + str(vill_no) + '.csv'
    ad_mat_key_vill = np.array(pd.read_csv(matrix_key_filename, header = None))

    # confirm households in every village are actually labelled 1...h
    assert(np.array_equal(ad_mat_key_vill.flatten(),stata_vill['HHnum_in_village'].values))

    # confirm is symmetric
    assert(is_symmetric(ad_mat_np1))
    assert(is_symmetric(ad_mat_np2))
    assert(is_symmetric(ad_mat_np3))

    G_nx_data1 = nx.from_numpy_matrix(ad_mat_np1)
    G_nx_data2 = nx.from_numpy_matrix(ad_mat_np2)
    G_nx_data3 = nx.from_numpy_matrix(ad_mat_np3)

    data_type_dict = {k: v for k, v in enumerate(room_type)}
    nx.set_node_attributes(G_nx_data1, data_type_dict, "type")
    nx.set_node_attributes(G_nx_data2, data_type_dict, "type")
    nx.set_node_attributes(G_nx_data3, data_type_dict, "type")

    # values to aim for
    # data_tri_cnt1 = sum((nx.triangles(G_nx_data1)).values())/3
    # data_assort1 = nx.attribute_assortativity_coefficient(G_nx_data1, "type")
    #
    # data_tri_cnt2 = sum((nx.triangles(G_nx_data2)).values())/3
    # data_assort2 = nx.attribute_assortativity_coefficient(G_nx_data2, "type")
    #
    # data_tri_cnt3 = sum((nx.triangles(G_nx_data3)).values())/3
    # data_assort3 = nx.attribute_assortativity_coefficient(G_nx_data3, "type")

    ########## simulation ###########


    summ_stats, final_ntwks, final_types = run_sim(sc_likelihood, ho_likelihood, max_clique_size, ctxt_likelihood = sum(room_type)/len(room_type), _N = G_nx_data1.number_of_nodes(), sim_iters = sim_iters)

    #value = [str(loss1), '\n', str(loss2), '\n', str(loss3), '\n']

    metrics = [str(summ_stats['standard']['triangle_count']), '\n', str(summ_stats['standard']['assortativity']), '\n', str(summ_stats['standard']['edge_count'])]

    data_dir = 'finer_results'

    filename = '{odir}/{vill_no}_{k}_{sc}_{ho}_losses.txt'.format(
        odir=data_dir, vill_no=str(vill_no), k=str(max_clique_size), sc=str(sc_likelihood), ho=str(ho_likelihood))

    with open(filename, 'w') as f:
        #f.writelines(value)
        f.writelines(metrics)
    f.close()


vill_list_old = chain(range(12),range(13, 21),range(22, 77))
vill_list = [x+1 for x in vill_list_old]


max_clique_size_list = [10]
sc_likelihood_list_old = range(17)
ho_likelihood_list_old = range(17)

sc_likelihood_list = [x/16 for x in sc_likelihood_list_old]
ho_likelihood_list = [x/16 for x in ho_likelihood_list_old]

paramlist = list(product(vill_list, max_clique_size_list, sc_likelihood_list, ho_likelihood_list))

# to_run = []
#
# for i in paramlist:
#
#     vill_no = i[0] + 1
#     k = i[1]
#     sc = i[2]
#     ho = i[3]
#
#     new_i = list([vill_no, k, sc, ho])
#
#     data_dir = 'fine_results'
#
#     filename = '{odir}/{vill_no}_{k}_{sc}_{ho}_losses.txt'.format(
#         odir=data_dir, vill_no=str(vill_no), k = str(k), sc = str(sc), ho = str(ho))
#
#     if exists(filename):
#         continue
#     else:
#         to_run.append(new_i)

if __name__ == '__main__':

    # create a process pool that uses all cpus
    with mp.Pool(processes = 32) as pool:
        # call the function for each item in parallel
        pool.map(fit_village_data, paramlist)
