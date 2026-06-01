from collections import defaultdict
import math
from itertools import combinations, product
import json
import sys
import copy
import multiprocessing as mp

import networkx as nx
import community as community_louvain
from scipy.sparse import linalg as scp_sla
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sim_lib.util as util
import sim_lib.attr_lib.util as alu
from sim_lib.attr_lib.formation import *
import sim_lib.attr_lib.vis as vis

############### initializing params ###############

_N = 150
max_deg = 10
satisfice = 1
num_iters = 500
min_iters = 10
max_clique_size = 11
ctxt_likelihood = .5

sim_iters = 10
#sim_iters = 1

st_count_track = 10
st_count_dev_tol = 0.01

sc_vals = np.linspace(0, 1, 9)
ho_vals = np.linspace(0, 1, 9)
#sc_vals = [ 0, 1 ]
#ho_vals = [ 0, 1 ]

# Create types
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

################ graph functions ################

# size of components
def get_component_sizes(G):
    G_nx = alu.graph_to_nx(G)
    G_nx_comp_nodes = list(nx.connected_components(G_nx))
    G_nx_largest = G_nx.subgraph(max(G_nx_comp_nodes, key=len))
    G_nx_comps = [ G_nx.subgraph(G_nxc_nodes) for G_nxc_nodes in G_nx_comp_nodes ]
    component_sizes = [ len(G_nxc) for G_nxc in G_nx_comps ]
    return component_sizes

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


################ other functions ################

# constructing basic pdf from util list
def to_pdf(data):
    pdf = [0]*25
    counts = [0]*25
    for util in data:
        counts[int(util*10)] = counts[int(util*10)] + 1
    pdf = [(x / sum(counts)) + .001 for x in counts]
    return pdf

def gini_coefficient(x):
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("Input array is empty.")
    if np.any(x < 0):
        raise ValueError("Input array contains negative values.")
    if x.sum() == 0:
        return 0.0

    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
    return (2 * (index * x).sum()) / (n * x.sum()) - (n + 1) / n

def get_summary_stats(G):
    num_components = len(get_component_sizes(G))

    degrees = [ v.degree for v in G.vertices ]
    avg_deg = np.mean(degrees)
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
    cluster_coeff = nx.average_clustering(g_nx)
    stable_triad_count = count_stable_triads(G)

    # Degree based
    deg_deg_coeff = nx.degree_assortativity_coefficient(g_nx)
    deg_std = np.std(degrees)
    deg_gini = gini_coefficient(degrees)

    # Reachability
    apls = []
    diams = []
    for C in (g_nx.subgraph(c) for c in nx.connected_components(g_nx)):
        c_apl = nx.average_shortest_path_length(C)
        c_diam = nx.diameter(C)
        apls.append(c_apl)
        diams.append(c_diam)

    apl = np.mean(apls)
    diameter = np.mean(diams)

    # Connectedness
    g_nx_comp_nodes = list(nx.connected_components(g_nx))
    g_nx_largest = g_nx.subgraph(max(g_nx_comp_nodes, key=len))
    giant_prop = len(g_nx_largest) / len(g_nx)
    centrality = nx.betweenness_centrality(g_nx)
    cent_gini = gini_coefficient(list(centrality.values()))

    return {
        'degree' : avg_deg,
        'deg_deg' : deg_deg_coeff,
        'deg_std' : deg_std,
        'deg_gini' : deg_gini,
        'apl' : apl,
        'diameter' : diameter,
        'cluster_coeff' : cluster_coeff,
        'giant_prop' : giant_prop,
        'centrality_gini' : cent_gini,
        'util' : avg_util,
        'cost' : avg_cost,
        'stable_triad_count' : stable_triad_count,
        'num_comm' : num_comm,
        'num_comp' : num_comp,
        'modularity' : modularity,
    }

def add_sum_stat(st_dict, res):
    for mtr, val in res.items():
        st_dict[mtr].append(val)

################ run simulation ################

def run_sim(sc_likelihood, ho_likelihood, sim_iters, sub=False):
    ctxt_types = [-1, 1]
    #ctxt_base_colors = [[43, 98, 166], [161, 39, 45]]
    ctxt_base_shapes = [0 , 2]
    ctxt_p = [ctxt_likelihood, 1-ctxt_likelihood]
    struct_types = ['em', 'sc']
    attr_types = ['ho', 'he']
    type_itr = [ (ctxt, shape, ct_p, at, st) for (ctxt, shape, ct_p) in zip(ctxt_types, ctxt_base_shapes, ctxt_p)
                for (at, st) in [(a, s) for a in attr_types for s in struct_types] ]
    type_list = [ type_dict(*t_args, sc_likelihood, ho_likelihood) for t_args \
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
        'vtx_types' : vtx_types,
    }

    vtx_types_list = [ np.repeat(t, tc) for t, tc in tc_dict.items() ]
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
        'degree' : [],
        'deg_deg' : [],
        'deg_std' : [],
        'deg_gini' : [],
        'apl' : [],
        'diameter' : [],
        'cluster_coeff' : [],
        'giant_prop' : [],
        'centrality_gini' : [],
        'util' : [],
        'cost' : [],
        'stable_triad_count' : [],
        'num_comm' : [],
        'num_comp' : [],
        'modularity' : [],
        'exit_iter' : [num_iters] * sim_iters
    }

    for si in range(sim_iters):

        # Create networks to be compared
        G_std = attribute_network(_N, copy.deepcopy(params))

        st_counts = []

        std_fin = False

        for it in range(num_iters):
            
            # Calculate edges for networks
            G_std = calc_edges(G_std, k=2)

            # Get all stable triad counts
            std_st_count = count_stable_triads(G_std)

            # If less than min number iterations has run, add and move on
            if len(st_counts) < st_count_track:
                st_counts.append(std_st_count)  
                continue

            # Update all count arrays with current
            st_counts.pop(0)
            st_counts.append(std_st_count)

            # Check if base case has just terminated
            if np.std(st_counts) <= st_count_dev_tol and not std_fin:
                std_fin = True
                summary_stats['exit_iter'][si] = it
                add_sum_stat(summary_stats, get_summary_stats(G_std))

            if std_fin:
                break

    print('ho: ', ho_likelihood, 'sc: ', sc_likelihood)

    # Take mean of all summary stats
    for st, vs in summary_stats.items():
        summary_stats[st] = np.mean(vs)
    return summary_stats

################ run simulation with various params ################

if __name__ == "__main__":
    all_stats = { }
    for sc, ho in product(sc_vals, ho_vals):
        print('running', sc, ho)
        all_stats[ f'{sc} {ho}' ] = {}
        summary_stats = run_sim(sc, ho, sim_iters)
        all_stats[ f'{sc} {ho}' ] = summary_stats

    stat_outname = f'data/{_N}_{max_deg}_stats.json'
    with open(stat_outname, 'w+') as out:
        out.write(json.dumps(all_stats))

