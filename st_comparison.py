from collections import defaultdict
import math
from itertools import combinations
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

# Sim input params

n = int(sys.argv[1])
max_deg = int(sys.argv[2])
sc_likelihood = float(sys.argv[3])
ho_likelihood = float(sys.argv[4])

############### initializing params ###############

_N = n
satisfice = 1
num_iters = 500
min_iters = 10
max_clique_size = max_deg + 1
ctxt_likelihood = .5
sim_iters = 10
#sim_iters = 2
st_count_track = 10
st_count_dev_tol = 0.01

nonlocal_dists = list(range(3, 8))
# Make sure to add k = 150 here for large _N!
budgets = [math.ceil(math.log(_N)), math.ceil(_N / 2), math.ceil(3 * _N / 4), _N]

similarity_homophily, similarity_heterophily = alu.gen_similarity_funcs()
total_attr_util = alu.gen_attr_util_func(satisfice)

# Create types
def type_dict(context, shape, context_p, attr, struct):
    likelihood = context_p
    if struct == 'em':
        struct_func = alu.satisfice(satisfice)(alu.triangle_count)
        likelihood = likelihood * (1 - sc_likelihood)
    else:
        struct_func = alu.satisfice(satisfice)(alu.num_nbor_comp_nx)
        likelihood = likelihood * sc_likelihood
    if attr == 'ho':
        attr_edge_func = similarity_homophily
        likelihood = likelihood * ho_likelihood
    else:
        attr_edge_func = similarity_heterophily
        likelihood = likelihood * (1 - ho_likelihood)

    attr_total_func = total_attr_util

    #Base color is a rgb list
    base_dict = {'likelihood' : likelihood,
              'struct_util' : struct_func,
              'struct' : struct,
              'init_attrs' : context,
              'attr' : attr,
              'edge_attr_util' : attr_edge_func,
              'total_attr_util' : attr_total_func,
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
    modularity = community_louvain.modularity(partition, g_nx)
    num_comp = num_components
    comp_apl = np.mean([ nx.average_shortest_path_length(g_nx.subgraph(c))
        for c in nx.connected_components(g_nx) ])
    cluster_coeff = nx.average_clustering(g_nx)
    stable_triad_count = count_stable_triads(G)

    return {
        'degree' : avg_deg,
        'util' : avg_util,
        'cost' : avg_cost,
        'num_comm' : num_comm,
        'modularity' : modularity,
        'num_comp' : num_comp,
        'apl' : comp_apl,
        'cluster_coeff' : cluster_coeff,
        'stable_triad_count' : stable_triad_count
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

################ run simulation ################

def run_sim(sc_likelihood, ho_likeliood, sim_iters, sub=False):
    ctxt_types = [-1, 1]
    #ctxt_base_colors = [[43, 98, 166], [161, 39, 45]]
    ctxt_base_shapes = [0 , 2]
    ctxt_p = [ctxt_likelihood, 1-ctxt_likelihood]
    struct_types = ['em', 'sc']
    attr_types = ['ho', 'he']
    type_itr = [ (ctxt, shape, ct_p, at, st) for (ctxt, shape, ct_p) in zip(ctxt_types, ctxt_base_shapes, ctxt_p)
                for (at, st) in [(a, s) for a in attr_types for s in struct_types] ]
    type_list = [ type_dict(*t_args) for t_args \
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
    np.random.shuffle(vtx_types_list)
    params['type_assignment'] = { i : vtx_types_list[i] for i in range(_N) }

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
        'exit_iter' : [num_iters] * sim_iters
    }

    summary_stats = {
        'standard' : copy.deepcopy(summary_stats),
        'nonlocal' :
            { d : copy.deepcopy(summary_stats) for d in nonlocal_dists },
        'budget' :
            { k : copy.deepcopy(summary_stats) for k in budgets },
        'nonlocal_match' :
            { d : copy.deepcopy(summary_stats) for d in nonlocal_dists },
        'budget_match' : 
            { k : copy.deepcopy(summary_stats) for k in budgets },
    }

    final_networks = {
        'standard' : [],
        'nonlocal' :
            { d : [] for d in nonlocal_dists },
        'budget' :
            { k : [] for k in budgets },
        'nonlocal_match' :
            { d : [] for d in nonlocal_dists },
        'budget_match' :
            { k : [] for k in budgets }
    }
 
    for si in range(sim_iters):

        # Create networks to be compared
        # Base case network
        G_std = attribute_network(_N, copy.deepcopy(params))

        # Comparison networks
        G_bdgt = {}
        for k in budgets:
            bdgt_params = copy.deepcopy(params)
            bdgt_params['max_clique_size'] = k
            G_bdgt[k] = attribute_network(_N, bdgt_params)

        G_nl = { d : attribute_network(_N, copy.deepcopy(params))
            for d in nonlocal_dists }

        st_counts = {
            'standard' : [],
            'nonlocal' : 
                { d : [] for d in nonlocal_dists },
            'budget' : 
                { k : [] for k in budgets },
        }

        std_fin = False
        bdgt_fin = { k : False for k in budgets }
        nl_fin = { d : False for d in nonlocal_dists }
        
        for it in range(num_iters):
            
            # Calculate edges for networks

            # Attempt to parallelize
            procs = []
            if not std_fin:
                sf_proc =  mp.Process(target=calc_edges, args=(G_std,))
                procs.append(sf_proc)
                sf_procs.start()
            for k in budgets:
                if not bdgt_fin[k]:
                    bdgt_proc = mp.Process(target=calc_edges, args=(G_bdgt[k],))
                    procs.append(bdgt_proc)
                    bdgt_proc.start()
            for d in nonlocal_dists:
                if not nl_fin[d]:
                    nl_proc = mp.Process(target=calc_edges, args=(G_nl[d], d,))
                    procs.append(nl_proc)
                    nl_proc.start()

            for p in procs:
                p.join()

            # Get all stable triad counts
            std_st_count = count_stable_triads(G_std)
            bdgt_st_counts = {}
            for k in budgets:
                bdgt_st_counts[k] = count_stable_triads(G_bdgt[k])
            nl_st_counts = {}
            for d in nonlocal_dists:
                nl_st_counts[d] = count_stable_triads(G_nl[d])

            # If less than min number iterations has run, add and move on
            if len(st_counts['standard']) < st_count_track:
                st_counts['standard'].append(std_st_count)  
                for k in budgets:
                    st_counts['budget'][k].append(bdgt_st_counts[k])
                for d in nonlocal_dists:
                    st_counts['nonlocal'][d].append(nl_st_counts[d])
                continue

            # Update all count arrays with current
            st_counts['standard'].pop(0)
            st_counts['standard'].append(std_st_count)
            for k in budgets:
                st_counts['budget'][k].pop(0)
                st_counts['budget'][k].append(bdgt_st_counts[k])
            for k in nonlocal_dists:
                st_counts['nonlocal'][d].pop(0)
                st_counts['nonlocal'][d].append(nl_st_counts[d])

            # Check if base case has just terminated
            if np.std(st_counts['standard']) <= st_count_dev_tol and not std_fin:
                std_fin = True
                summary_stats['standard']['exit_iter'][si] = it
                add_sum_stat(summary_stats['standard'], get_summary_stats(G_std))
                final_networks['standard'].append(G_std.adj_matrix.tolist())

                for k in budgets:
                    summary_stats['budget_match'][k]['exit_iter'][si] = it
                    add_sum_stat(summary_stats['budget_match'][k],
                        get_summary_stats(G_bdgt[k]))
                    final_networks['budget_match'][k].append(G_bdgt[k].adj_matrix.tolist())

                for d in nonlocal_dists:
                    summary_stats['nonlocal_match'][d]['exit_iter'][si] = it
                    add_sum_stat(summary_stats['nonlocal_match'][d],
                        get_summary_stats(G_nl[d]))
                    final_networks['nonlocal_match'][d].append(G_nl[d].adj_matrix.tolist())

            for k in budgets:
                if np.std(st_counts['budget'][k]) <= st_count_dev_tol and not bdgt_fin[k]:
                    bdgt_fin[k] = True
                    summary_stats['budget'][k]['exit_iter'][si] = it
                    add_sum_stat(summary_stats['budget'][k], get_summary_stats(G_bdgt[k]))
                    final_networks['budget'][k].append(G_bdgt[k].adj_matrix.tolist())

            for d in nonlocal_dists:
                if np.std(st_counts['nonlocal'][d]) <= st_count_dev_tol and not nl_fin[d]:
                    nl_fin[d] = True
                    summary_stats['nonlocal'][d]['exit_iter'][si] = it
                    add_sum_stat(summary_stats['nonlocal'][d], get_summary_stats(G_nl[d]))
                    final_networks['nonlocal'][d].append(G_nl[d].adj_matrix.tolist())

            if std_fin and all(bdgt_fin.values()) and all(nl_fin.values()):
                break

    print('ho: ', ho_likelihood, 'sc: ', sc_likelihood)

    # Take mean of all summary stats
    for st, vs in summary_stats['standard'].items():
        summary_stats['standard'][st] = np.mean(vs)
    for k in budgets:
        for st in summary_stats['budget'][k].keys():
            summary_stats['budget'][k][st] = np.mean(summary_stats['budget'][k][st])
            summary_stats['budget_match'][k][st] = np.mean(
                summary_stats['budget_match'][k][st])
    for d in nonlocal_dists:
        for st in summary_stats['nonlocal'][d].keys():
            summary_stats['nonlocal'][d][st] = np.mean(summary_stats['nonlocal'][d][st])
            summary_stats['nonlocal_match'][d][st] = np.mean(
                summary_stats['nonlocal_match'][d][st])
    return summary_stats, final_networks

################ run simulation with various params ################

if __name__ == "__main__":
    summary_stats, final_networks = run_sim(sc_likelihood, ho_likelihood, sim_iters)

    stat_outname = 'data/comparison/{n}_{k}_{sc}_{ho}_stats.json'.format(
        n=str(n), k=str(max_deg), sc=str(sc_likelihood), ho=str(ho_likelihood))

    with open(stat_outname, 'w+') as out:
        out.write(json.dumps(summary_stats))

    ntwk_outname = 'data/comparison/{n}_{k}_{sc}_{ho}_networks.json'.format(
        n=str(n), k=str(max_deg), sc=str(sc_likelihood), ho=str(ho_likelihood))

    with open(ntwk_outname, 'w+') as out:
        out.write(json.dumps(final_networks))

