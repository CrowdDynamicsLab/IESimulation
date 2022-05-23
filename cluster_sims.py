from collections import defaultdict
from math import isclose

import networkx as nx
import community as community_louvain
from scipy.sparse import linalg as scp_sla
from scipy.special import rel_entr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sim_lib.util as util
import sim_lib.attr_lib.util as alu
from sim_lib.attr_lib.formation import *
import sim_lib.attr_lib.vis as vis
import sys

############### initializing params ###############

_N = 100
satisfice = 1
num_iters = 1000
min_iters = 10
max_clique_size = 10
ctxt_likelihood = .5
sc = [0, .125, .25, .375, .5, .625, .75, .875, 1]
ho = [0, .125, .25, .375, .5, .625, .75, .875, 1]
#sc = [0,1]
#ho = [0,1]
sim_iters = 1
kl_tolerance = .065

similarity_homophily, similarity_heterophily = alu.gen_similarity_funcs()
schelling_homophily, schelling_heterophily = alu.gen_schelling_seg_funcs(satisfice, 'sat_count')

# Create types
def type_dict(context, shape, context_p, attr, struct):
    likelihood = context_p
    if struct == 'em':
        struct_func = alu.satisfice(satisfice)(alu.triangle_count)
        likelihood = likelihood * (1 - sc_likelihood)
    else:
        struct_func = alu.satisfice(satisfice)(alu.degree_indep_size)
        likelihood = likelihood * sc_likelihood
    if attr == 'ho':
        attr_edge_func = similarity_homophily
        attr_total_func = schelling_homophily
        likelihood = likelihood * ho_likelihood
    else:
        attr_edge_func = similarity_heterophily
        attr_total_func = schelling_heterophily
        likelihood = likelihood * (1 - ho_likelihood)

    #Base color is a rgb list
    base_dict = {'likelihood' : likelihood,
              'struct_util' : struct_func,
              'init_attrs' : context,
              'edge_attr_util' : attr_edge_func,
              'total_attr_util' : attr_total_func,
              'optimistic' : False,
              #'color' : 'rgb({rgb})'.format(rgb=', '.join([ str(c) for c in color ])),
              'shape' :  shape
              #'{shape}'.format(shape=', '.join([str(s) for s in shape]))
              }

    return base_dict

################ logging/plotting ################

# size of components
def get_component_sizes(G):
    G_nx = alu.graph_to_nx(G)
    G_nx_comp_nodes = list(nx.connected_components(G_nx))
    G_nx_largest = G_nx.subgraph(max(G_nx_comp_nodes, key=len))
    G_nx_comps = [ G_nx.subgraph(G_nxc_nodes) for G_nxc_nodes in G_nx_comp_nodes ]
    component_sizes = [ len(G_nxc) for G_nxc in G_nx_comps ]
    return component_sizes

def plot_dist(G, degree_dist, util_dist, cost_dist, max_degree, title):

    G_nx = alu.graph_to_nx(G)
    max_degree = math.floor(1 / G.sim_params['direct_cost'])

    bins1 = np.linspace(0, 1.5*max_degree)
    bins2 = np.linspace(0, 2.5)
    bins3 = np.linspace(0, 1.5)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,8), sharey=True)

    ax1.set_title('Degree Distribution')
    ax1.set_ylabel('Num Nodes')
    ax1.hist(degree_dist, bins1, label='degree')

    ax2.set_title('Utility Distribution')
    ax2.hist(util_dist, bins2, label='utility')

    ax3.set_title('Cost Distribution')
    ax3.hist(cost_dist, bins3, label='cost')

    fig.suptitle(title)
    title_string = 'figures/distributions/' + title + '.png'
    fig.savefig(title_string, dpi = 300)
    plt.close('all')

def plot_heat_map(data, title, min, sc, ho):
    x_tick_labels = sc
    y_tick_labels = ho[::-1]
    ax = sns.heatmap(data, vmin = min,  cmap = 'YlOrBr', xticklabels = x_tick_labels, yticklabels = y_tick_labels, linewidth=0.5)
    plt.title(title)
    ax.set_xlabel('Homophily Prop')
    ax.set_ylabel('Social Capital Prop')
    title_save = 'figures/heatmaps/' + title + '.png'
    #plt.show()
    plt.savefig(title_save, dpi = 300)
    plt.close('all')

# constructing basic pdf from util list
def to_pdf(data):
    pdf = [0]*25
    counts = [0]*25
    for util in data:
        counts[int(util*10)] = counts[int(util*10)] + 1
    # adding 1 so none of the probs = 0
    pdf = [(x / sum(counts)) + .001 for x in counts]
    return pdf

################ run simulation ################

def run_sim(sc_likelihood, ho_likeliood, sim_iters):
    ctxt_types = [alu.init_cont_homophily, alu.init_cont_heterophily]
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
    exit_iter = [num_iters]*sim_iters
    kl_divergence = np.inf
    for k in range(sim_iters):
        prev_iter_util_dist = [0]*_N
        curr_iter_util_dist = [0]*_N
        G = attribute_network(_N, params)
        G_nx = alu.graph_to_nx(G)
        max_degree = math.floor(1 / G.sim_params['direct_cost'])
        for it in range(num_iters):
            G.sim_params['edge_selection'] = alu.seq_edge_sel_silent
            calc_edges(G)

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
            prev_iter_util_dist = curr_iter_util_dist
            curr_iter_util_dist = ([v.data['struct_util'](v, G) + v.data['total_attr_util'](v,G) for v in G.vertices ])
            prev_util_pdf = to_pdf(prev_iter_util_dist)
            curr_util_pdf = to_pdf(curr_iter_util_dist)
            kl_divergence = sum(rel_entr(prev_util_pdf, curr_util_pdf))
            #print(kl_divergence)
            if (kl_divergence <= kl_tolerance):
                #print('kl divergence small at iter ', it)
                if it <= min_iters:
                    continue
                exit_iter[k] = it
                break

        num_components = len(get_component_sizes(G))
        over_budget = sum([ind_ob(v) for v in G.vertices])
        num_sat = sum([sat_ob(v) for v in G.vertices])

        info_string = 'components: ' + str(num_components) + ', number over budget: ' + str(over_budget) + ', number satisfied: ' + str(num_sat)

        image_name = 'N' + str(_N) +  '_iter' + str(num_iters) + '_theta' + str(satisfice) + '_max' + str(max_clique_size) +\
                    '_ctx' + str(ctxt_likelihood) + '_sc' + str(sc_likelihood) + '_ho' + str(ho_likelihood)
        degree_dist = degree_dist + ([ v.degree for v in G.vertices ])
        util_dist = util_dist + ([v.data['struct_util'](v, G) + v.data['total_attr_util'](v,G) for v in G.vertices ])
        cost_dist = cost_dist + ([alu.calc_cost(v, G) for v in G.vertices ])

    ind_cost_dist = np.array(cost_dist) - np.array(degree_dist)*G.sim_params['direct_cost']
    print('ho: ', ho_likelihood, 'sc: ', sc_likelihood, 'exited in ', np.round(np.mean(exit_iter),2), ', std: ', np.round(np.std(exit_iter),2))
    #print(exit_iter)
    partition = {}
    partition = community_louvain.best_partition(alu.graph_to_nx(G))
    #print(partition)
    vis.graph_vis(G, image_name, info_string, partition)
    #vis.draw_graph(G, partition, image_name)

    plot_dist(G, degree_dist, util_dist, cost_dist, max_degree, image_name)
    summary_stats = [np.mean(degree_dist), np.mean(util_dist), np.mean(cost_dist), np.mean(exit_iter), np.mean(ind_cost_dist)]
    return summary_stats

################ run simulation with various params ################

degree_array = np.zeros((len(sc), len(ho)))
util_array = np.zeros((len(sc), len(ho)))
cost_array = np.zeros((len(sc), len(ho)))
iter_array = np.zeros((len(sc), len(ho)))
ind_cost_array = np.zeros((len(sc), len(ho)))

for i in sc:
    sc_likelihood = float(i)
    for j in ho:
        ho_likelihood = float(j)
        summary_stats = run_sim(sc_likelihood, ho_likelihood, sim_iters)
        degree_array[int((1-sc_likelihood)/float(sc[1])), int(ho_likelihood/float(ho[1]))] = summary_stats[0]
        util_array[int((1-sc_likelihood)/float(sc[1])), int(ho_likelihood/float(ho[1]))] = summary_stats[1]
        cost_array[int((1-sc_likelihood)/float(sc[1])), int(ho_likelihood/float(ho[1]))] = summary_stats[2]
        iter_array[int((1-sc_likelihood)/float(sc[1])), int(ho_likelihood/float(ho[1]))] = summary_stats[3]
        ind_cost_array[int((1-sc_likelihood)/float(sc[1])), int(ho_likelihood/float(ho[1]))] = summary_stats[4]
plot_heat_map(degree_array, 'Avg Degree', 0, sc, ho)
plot_heat_map(util_array, 'Avg Utility', 0, sc, ho)
plot_heat_map(cost_array, 'Avg Cost', 0, sc, ho)
plot_heat_map(iter_array, 'Avg Iterations', 0, sc, ho)
plot_heat_map(ind_cost_array, 'Avg Ind Cost', 0, sc, ho)
