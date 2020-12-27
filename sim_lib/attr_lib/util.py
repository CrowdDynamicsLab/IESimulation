"""
Util functions used in attribute search network
"""

import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm

import sim_lib.graph_networkx as gnx

##########################
# Edge utility functions #
##########################

def exp_surprise(u, v, G):
    total_surprise = 0
    matches = []
    for ctx in G.data[u]:
        if ctx in G.data[v]:
            total_surprise += np.log2(u.data[ctx] * v.data[ctx])
            matches.append(ctx)
    return 1 - 2 ** (total_surprise)

def simple_sigmoid(u, v, G, scale=1.0):

    # Simple sigmoid, |X| / (1 + |X|) where X is intersecting
    # contexts. Does not account for rarity of context
    match_count = 0
    for ctx, attrs in G.data[u].items():
        if ctx in G.data[v] and attrs in G.data[v][ctx]:
            match_count += 1
    return match_count / (scale + match_count)

##############################
# Edge probability functions #
##############################

def logistic(u, util, scale=1.0):
    total_util = u.data + util
    return (2 / (1 + np.exp(-1 * scale * total_util))) - 1

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

def discrete_pareto_val():

    # From Buddana Kozubowski discrete Pareto
    alpha = 2 # "shape"
    sigma = 1 # "size"
    std_exp_val  = np.random.exponential(scale=1.0)
    gamma_val = np.random.gamma(alpha, scale=sigma)
    return np.ceil(std_exp_val / gamma_val)

#NOTE: Both pareto dists have roughly 20% of attributes taking 50% of prob mass
def pareto_dist(num_attrs):

    # From Buddana Kozubowski discrete Pareto
    alpha = 2 # "shape"
    sigma = 1 # "size"
    std_exp_dist  = np.random.exponential(scale=1.0, size=num_attrs)
    gamma_dist = np.random.gamma(alpha, scale=sigma, size=num_attrs)
    pareto_dist = np.ceil(std_exp_dist / gamma_dist)
    return pareto_dist / sum(pareto_dist)

def quasi_pareto_dist(num_attrs):
    dist = np.random.pareto(2, num_attrs)
    return dist / sum(dist)

def split_dist(num_attrs, attr_split=0.2, mass_split=0.8):
    # Splits mass_split of the probability amongst attr_split of the attributes
    # evenly. Distributes the remainder evenly.

    heavy_attrs = int(np.ceil(num_attrs * attr_split))
    light_attrs = num_attrs - heavy_attrs
    heavy_probs = [ mass_split / heavy_attrs] * heavy_attrs
    light_probs = [ (1 - mass_split) / light_attrs] * light_attrs
    return heavy_probs + light_probs

def log_dist(num_attrs):
    # Attempts to follow 80/20 rules by log distribution
    hattrs = int(np.ceil(num_attrs * 0.2))
    lattrs = num_attrs - hattrs
    hlogs = np.logspace(0, 0.8, num=hattrs + 1, base=2.0)
    llogs = np.logspace(0.8, 1.0, endpoint=True, num=lattrs + 1, base=2.0)
    hprobs = [ hlogs[i + 1] - hlogs[i] for i in range(hattrs) ]
    lprobs = [ llogs[i + 1] - llogs[i] for i in range(lattrs) ]

    assert sum(lprobs) + sum(hprobs) == 1.0, 'log_dist must have probabilities summing to 1.0'
    return hprobs + lprobs

def gen_peak_dist(num_attrs, peak, peak_prob=0.99):
    def peak_dist(num_attrs):
        probs = [ (1 - peak_prob) / (num_attrs - 1) for _ in range(num_attrs) ]
        probs[peak] = peak_prob
        return probs
    return peak_dist

def uniform_dist(num_attrs):
    return [ 1 / num_attrs for _ in range(num_attrs) ]

###########
# Metrics #
###########
def modularity(G_nx):
    # Gets modularity based on greedily optimized maximum
    partitions = nx_comm.greedy_modularity_communities(G_nx)
    return nx_comm.modularity(G_nx, partitions)
