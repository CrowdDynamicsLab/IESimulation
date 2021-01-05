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
        if ctx in G.data[v]:
            match_count += len(attrs.intersection(G.data[v][ctx]))
    return match_count / (scale + match_count)

def discrete_pareto_pdf(k, alpha=2, sigma=1):
    
    # PDF of discrete Pareto distribution (Pareto II)
    k_1 = ( 1 / ( 1 + k / sigma) ) ** alpha
    k_2 = ( 1 / ( 1 + ( (k + 1) / sigma ) ) ) ** alpha
    return k_1 - k_2

def total_likelihood(u, v, G):
    # Gives utility as the summed likelihood of shared attributes
    # In discrete choice terms it is 1_u^T theta 1_v where theta is the
    # diagonal matrix with normalized attribute likelihoods

    sum_likelihood = 0
    for ctx, u_attrs in G.data[u].items():
        if ctx in G.data[v]:
            for attr in u_attrs.intersection(G.data[v][ctx]):
                sum_likelihood += discrete_pareto_pdf(attr)
    return sum_likelihood / G.sim_params['k']

##############################
# Edge probability functions #
##############################

def marginal_logistic(u, util, scale=15.0):
    log_func = lambda x : (2 / (1 + np.exp(-1 * scale * x))) - 1
    return log_func(u.data + util) - log_func(u.data)

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
    total_indirect_cost = edge_count * indirect_cost

    return total_direct_cost + total_indirect_cost

def remaining_budget(u, G, dunbar=150):
    u_cost = calc_cost(u, G.sim_params['direct_cost'],
            G.sim_params['indirect_cost'], G)
    budget = dunbar * G.sim_params['direct_cost']
    return budget - u_cost

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

def discrete_pareto_val(alpha=2, sigma=1):

    # From Buddana Kozubowski discrete Pareto
    # alpha "shape" sigma "size"
    std_exp_val  = np.random.exponential(scale=1.0)
    gamma_val = np.random.gamma(alpha, scale=sigma)
    return np.ceil(std_exp_val / gamma_val)

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
