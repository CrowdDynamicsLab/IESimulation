"""
Util functions used in attribute search network
"""

import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm

import sim_lib.graph_networkx as gnx

#########################
# Probability functions #
#########################

def exp_surprise(u, v, G, context_dist):
    total_surprise = 0
    for ctx in G.data[u]:
        if ctx in G.data[v]:
            total_surprise += - np.log2(context_dist[ctx])
    return 1 - 2 ** (-2 * total_surprise)

def simple_sigmoid(u, v, G, context_dist):
    # Simple sigmoid, |X| / (1 + |X|) where X is intersecting
    # contexts. Does not account for rarity of context
    match_count = 0
    for ctx in G.data[u]:
        if ctx in G.data[v]:
            match_count += 1
    return match_count / (1 + match_count)

##################
# Cost functions #
##################

def calc_cost(u, direct_cost, indirect_cost, G):
    u_subgraph_degree = 0
    nbor_set = set(u.nbors)
    for v in nbor_set:
        u_subgraph_degree += len(nbor_set.intersection(set(v.nbors)))
    assert u_subgraph_degree % 2 == 0, 'sum of degrees must be even'
    edge_count = u_subgraph_degree / 2
    total_direct_cost = len(nbor_set) * direct_cost
    total_indirect_cost = (edge_count - len(nbor_set)) * indirect_cost

    return total_direct_cost + total_indirect_cost

#########################
# Measurement functions #
#########################
def indirect_distance(u, v, G):
    G.remove_edge(u, v)
    G_nx = gnx.graph_to_nx(G)
    indirect_dist = 0
    try:
        indirect_dist = nx.shortest_path_length(G_nx, source=u, target=v)
    except nx.NetworkXNoPath:
        indirect_dist = -1
    G.add_edge(u, v, 1)
    return indirect_dist

###########################
# Attribute distributions #
###########################
def pareto_dist(num_attrs):
    dist = np.random.pareto(2, num_attrs)
    return dist / sum(dist)

def uniform_dist(num_attrs):
    return [ 1 / num_attrs for _ in range(num_attrs) ]

###########
# Metrics #
###########
def modularity(G_nx):
    # Gets modularity based on greedily optimized maximum
    partitions = nx_comm.greedy_modularity_communities(G_nx)
    return nx_comm.modularity(G_nx, partitions)
