from collections.abc import Iterable
import copy
import math
import time

import networkx as nx
import numpy as np

import sim_lib.graph as graph
import sim_lib.attr_lib.util as attr_lib_util

# Edge selection
def calc_utils(G):

    # Calculates attribute utility over each edge (homophily or heterophily)
    util_mat = np.zeros((G.num_people, G.num_people))
    for i, u in enumerate(G.vertices):
        for v in G.vertices[i + 1:]:
            util_mat[u.vnum][v.vnum] = u.data['edge_attr_util'](u, v, G)
            util_mat[v.vnum][u.vnum] = v.data['edge_attr_util'](v, u, G)
    G.potential_utils = util_mat
    return G.potential_utils

def calc_edges(G, k=2):
   
    # Get distance k agents for proposals
    adj_mat = G.adj_matrix
    dk_mat = np.linalg.matrix_power(adj_mat, k)
    nbor_mask = -1 * (adj_mat - 1)
    np.fill_diagonal(nbor_mask, 0)
    edge_proposals = nbor_mask * dk_mat
    edge_proposals[edge_proposals > 0] = 1

    # Add revalation and check budget
    revelations = G.sim_params['revelation_proposals'](G)

    # Only propose to vertices with non-negative expected utility
    all_costs = attr_lib_util.calc_all_costs(G)
    edge_prop_dict = {}
    
    for v in G.vertices:
        v_attr_util, v_struct_util = v.utility_values(G)
        v_cost = all_costs[v.vnum]
        v_agg_util = G.sim_params['util_agg'](v_attr_util, v_struct_util, v_cost, v, G)

        # Skip satiated
        if v_agg_util >= 2.0 or v_cost >= 1.0:
            edge_prop_dict[v] = None
            continue

        # Only propose to max value candidate
        #NOTE: max_val = 0 implies non-optimism
        max_val = 0
        max_cand = None
        candidates = [ G.vertices[i] for i in np.nonzero(edge_proposals[v.vnum])[0]]
        candidates.append(G.vertices[revelations[v.vnum]])
        for u in candidates:
            if G.are_neighbors(v, u):
                continue
            G.add_edge(v, u)
            pattr, pstruct = v.utility_values(G)
            pcost = attr_lib_util.calc_cost(v, G)
            pagg_util = G.sim_params['util_agg'](pattr, pstruct, pcost, v, G)

            # Optimism from >= as opposed to >
            util_del = pagg_util - v_agg_util
            if util_del > 0 and pagg_util > max_val:
                max_val = pagg_util
                max_cand = u
            G.remove_edge(v, u)
        edge_prop_dict[v] = max_cand

    # Returns metadata
    metadata = G.sim_params['edge_selection'](G, edge_prop_dict)
    return G

def initialize_vertex(G, vtx=None):
    # If no vertex is passed as arg, creates a vertex. Otherwise uses given.
    if vtx == None:
        vtx = graph.Vertex(G.num_people)

    vtx_type_dists = { t : td['likelihood'] for t, td in G.sim_params['vtx_types'].items() }

    chosen_type = None
    if 'type_assignment' in G.sim_params:
        if vtx in G.sim_params['type_assignment']:
            chosen_type = G.sim_params['type_assignment'][vtx]
        elif vtx.vnum in G.sim_params['type_assignment']:
            chosen_type = G.sim_params['type_assignment'][vtx.vnum]
    else:
        # coin flip type selection
        vtx_types = list(vtx_type_dists.keys())
        vtx_type_likelihoods = [ vtx_type_dists[vt] for vt in vtx_types ]
        chosen_type = np.random.choice(vtx_types, p=vtx_type_likelihoods)

    vtx.data = copy.copy(G.sim_params['vtx_types'][chosen_type])
    vtx.data['type_name'] = chosen_type
    vtx.data.pop('likelihood')
    vtx.attr_type = vtx.data['init_attrs']

    #NOTE: Keep old code for reference of multi-context pareto attr init
#    contexts = np.random.choice(list(range(G.sim_params['context_count'])),
#            replace=False, size=G.sim_params['k'])
#    G.data[vtx] = { context : { vtx.data['attr_func']() } for context in contexts }

    return vtx

# Graph creation
def attribute_network(n, params):
    # If clique is true, initialize network as a clique

    G = graph.Graph()
    G.data = {}
    G.sim_params = params

    vtx_set = []

    max_clique_degree = G.sim_params['max_clique_size'] - 1
    max_indirect_edges = max_clique_degree * (max_clique_degree - 1) / 2
    cost_polynomial = [ max_indirect_edges, max_clique_degree, -1 ]
    cost_roots = np.roots(cost_polynomial)

    G.sim_params['direct_cost'] = max(cost_roots)
    G.sim_params['indirect_cost'] = G.sim_params['direct_cost'] ** 2
    
    # Ignore indirect cost
    #G.sim_params['max_degree'] = math.floor(1 / G.sim_params['direct_cost'])
    G.sim_params['max_degree'] = max_clique_degree

    for i in range(n):
        vtx = graph.Vertex(i)
        vtx = initialize_vertex(G, vtx)
        vtx_set.append(vtx)

    G.vertices = vtx_set

    # Calculate edge utils
    calc_utils(G)

    if params['seed_type'] == 'clique':

        # This may give a network that starts as over budget
        for v_idx in range(n):
            for u_idx in range(v_idx + 1, n):
                G.add_edge(G.vertices[v_idx], G.vertices[u_idx])
    elif params['seed_type'] == 'trivial':
        pass
    elif params['seed_type'] == 'erdos_renyi':
        edge_prob = ((1 + (2 ** -10)) * math.log(n)) / n
        #edge_prob = 1 / n
        for v_idx in range(n):
            for u_idx in range(v_idx + 1, n):
                if np.random.random() <= edge_prob:
                    G.add_edge(G.vertices[v_idx], G.vertices[u_idx])
    elif params['seed_type'] == 'grid':
        n_sqrt = math.floor(math.sqrt(n))
        assert n_sqrt ** 2 == n, 'Have not implemented non-square n for grid seed'
        grid_l = n_sqrt
        grid_w = n_sqrt
        coord = lambda i : (i % grid_w, i // grid_l) # gives (x, y) coordinate
        row_idx = lambda x, y : x + (y * grid_l)
        in_bound = lambda x, y : x < grid_w and y < grid_l
        for v_idx in range(n):
            v_x, v_y = coord(v_idx)
            right_vtx = (v_x + 1, v_y)
            down_vtx = (v_x, v_y + 1)
            if in_bound(*right_vtx):
                G.add_edge(G.vertices[v_idx], G.vertices[row_idx(*right_vtx)])
            if in_bound(*down_vtx):
                G.add_edge(G.vertices[v_idx], G.vertices[row_idx(*down_vtx)])
        for v in G.vertices:
            assert v.degree > 1, 'How can degree be < 2'

    G.init_adj_matrix()
    return G

