from collections import defaultdict
from math import isclose
from itertools import combinations
import json
import sys

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

n = float(sys.argv[1])
k = float(sys.argv[2])
sc_likelihood = float(sys.argv[3])
ho_likelihood = float(sys.argv[4])

############### initializing params ###############

_N = n
satisfice = 1
num_iters = 500
min_iters = 10
max_clique_size = k
ctxt_likelihood = .5
sc = [0, .125, .25, .375, .5, .625, .75, .875, 1]
ho = [0, .125, .25, .375, .5, .625, .75, .875, 1]
#sc = [0, 1]
#ho = [0, 1]
sim_iters = 10
#sim_iters = 1
st_count_track = 10
st_count_dev_tol = 0.01

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

def get_edge_types(G):
    edge_types = []
    G_nx = alu.graph_to_nx(G)
    for (u, v) in G_nx.edges():
        if u.data['struct'] == v.data['struct']:
            edge_types = edge_types + ['black']
        else:
            edge_types = edge_types + ['saddlebrown']
    return edge_types

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


################ plotting functions ################

def plot_heat_map(data, title, min, sc, ho):
    x_tick_labels = sc
    y_tick_labels = ho[::-1]
    ax = sns.heatmap(data, vmin = min,  cmap = 'YlOrBr', xticklabels = x_tick_labels, yticklabels = y_tick_labels, linewidth=0.5)
    plt.title(title)
    ax.set_xlabel('Homophily Prop')
    ax.set_ylabel('Social Capital Prop')
    title_save = 'figures/heatmaps_budgetless/' + title + '.png'
    #plt.show()
    plt.savefig(title_save, dpi = 300)
    plt.close('all')


################ other functions ################

# constructing basic pdf from util list
def to_pdf(data):
    pdf = [0]*25
    counts = [0]*25
    for util in data:
        counts[int(util*10)] = counts[int(util*10)] + 1
    pdf = [(x / sum(counts)) + .001 for x in counts]
    return pdf


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
    for i in range(remaining_tc):
        type_counts[i] = type_counts[i] + 1
    assert sum(type_counts) == _N, 'Did that work?'

    tc_dict = { f'type{idx}' : tc for idx, tc in enumerate(type_counts) }
    vtx_types = { f'type{idx}' : tl for idx, tl in enumerate(type_list) }

    params = {
        'context_count' : 2, # Needed for simple utility
        'k' : 1, # Needed for simple attribute utility
        'edge_selection' : alu.seq_projection_edge_edit,
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

    assert isclose(sum([ t['likelihood'] for t in params['vtx_types'].values() ]), 1.0)

    degree_dist = []
    util_dist = []
    cost_dist = []
    ind_cost_dist = []
    num_comm = []
    modularity = []
    num_comp = []
    apl = []
    cluster_coeff = []
    stable_triad_count = []

    exit_iter = [num_iters]*sim_iters
    
    for k in range(sim_iters):
        G = attribute_network(_N, params)
        G_nx = alu.graph_to_nx(G)
        st_count_hist = []
        for it in range(num_iters):
            G.sim_params['edge_selection'] = alu.seq_edge_sel_silent

            # Global now
            calc_edges_global(G)

            # Sim iter end, start stat collection
            attr_util_vals = [ u.data['total_attr_util'](u, G) for u in G.vertices ]
            struct_util_vals = [ u.data['struct_util'](u, G) for u in G.vertices ]
            costs = [ alu.calc_cost(u, G) for u in G.vertices ]
            values = [ attr_util_vals, struct_util_vals, costs ]

            ind_ob = lambda v : 1 if alu.remaining_budget(v, G) < 0 else 0
            sat_ob = lambda v : 1 if G.sim_params['util_agg'](
                v.data['total_attr_util'](v, G),
                v.data['struct_util'](v, G),
                alu.calc_cost(v, G),
                v, G
            ) == 2.0 else 0

            st_count = count_stable_triads(G)
            if len(st_count_hist) < st_count_track:
                st_count_hist.append(st_count)
            elif np.std(st_count_hist) <= st_count_dev_tol:
                exit_iter[k] = it
                break
            else:
                st_count_hist.pop(0)
                st_count_hist.append(st_count)

        #plot_mod_line(mod_iters, sc_likelihood, ho_likeliood)

        num_components = len(get_component_sizes(G))
        over_budget = sum([ind_ob(v) for v in G.vertices])
        num_sat = sum([sat_ob(v) for v in G.vertices])

        info_string = 'components: ' + str(num_components) + ', number over budget: ' + str(over_budget) + ', number satisfied: ' + str(num_sat)

        image_name = 'N' + str(_N) +  '_iter' + str(num_iters) + '_theta' + str(satisfice) + '_max' + str(max_clique_size) +\
                    '_ctx' + str(ctxt_likelihood) + '_sc' + str(sc_likelihood) + '_ho' + str(ho_likelihood)
        degree_dist = degree_dist + ([ v.degree for v in G.vertices ])
        util_dist = util_dist + ([v.data['struct_util'](v, G) + v.data['total_attr_util'](v,G) for v in G.vertices ])
        cost_dist = cost_dist + ([alu.calc_cost(v, G) for v in G.vertices ])

        # Metric calc
        g_nx = alu.graph_to_nx(G)

        partition = {}
        partition = community_louvain.best_partition(g_nx)

        num_comm.append(max(partition.values()))
        modularity.append(community_louvain.modularity(partition, g_nx))
        num_comp.append(num_components)
        comp_apl = np.mean([ nx.average_shortest_path_length(c) for c in nx.connected_components(G_nx) ])
        apl.append(comp_apl)
        cluster_coeff.append(nx.average_clustering(g_nx))
        stable_triad_count.append(count_stable_triads(G))

    print('ho: ', ho_likelihood, 'sc: ', sc_likelihood)

    summary_stats = [np.mean(degree_dist), np.mean(util_dist), np.mean(cost_dist),
                    np.mean(exit_iter), np.mean(num_comm), np.mean(modularity),
                    np.mean(num_comp), np.mean(apl), np.mean(cluster_coeff),
                    np.mean(stable_triad_count)]
    return summary_stats

################ run simulation with various params ################

summary_stats = run_sim(sc_likelihood, ho_likelihood, sim_iters)

sim_stats = {
    'sc' : sc_likelihood,
    'ho' : ho_likelihood,
    'degree' : summary_stats[0],
    'util' : summary_stats[1],
    'cost' : summary_stats[2],
    'iters' : summary_stats[3],
    'num_communities' : summary_stats[4],
    'modularity' : summary_stats[5],
    'num_components' : summary_stats[6],
    'apl' : summary_stats[7],
    'cluster_coeff' : summary_stats[8],
    'stable_triads' : summary_stats[9]
}

outname = 'data/global/{sc}_{ho}_simout.json'.format(
    sc=str(sc_likelihood), ho=str(ho_likelihood))

with open(outname, 'w+') as out:
    out.write(json.dumps(sim_stats))
