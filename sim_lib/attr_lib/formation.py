from collections.abc import Iterable
import copy

import networkx as nx
import numpy as np

import sim_lib.graph as graph
import sim_lib.attr_lib.util as attr_util

# Edge selection
def calc_utils(G):
    util_mat = np.zeros((G.num_people, G.num_people))
    for i, u in enumerate(G.vertices):
        for v in G.vertices[i + 1:]:
            util_mat[u.vnum][v.vnum] = G.sim_params['edge_util_func'](u, v, G)
            util_mat[v.vnum][u.vnum] = util_mat[u.vnum][v.vnum]
    G.potential_utils = util_mat
    return G.potential_utils

def calc_edges(G, walk_proposals=False, dunbar=150):
    
    if not walk_proposals:
        edge_proposals = { v : [ u for u in G.vertices if u != v ] \
                for v in G.vertices if attr_util.remaining_budget(v, G) > 0 }
        G.sim_params['edge_selection'](G, edge_proposals)
        return

    edge_proposals = {}

    for u in G.vertices:
        edge_proposals[u] = []

        # Only need to check visited edges for proposal
        for v in u.data['visited']:
            if G.are_neighbors(u, v) or u == v:
                continue
           
            edge_util = G.potential_utils[u.vnum][v.vnum]
            if G.sim_params['edge_proposal'](u, edge_util) >= np.random.random():
                edge_proposals[u].append(v)

    G.sim_params['edge_selection'](G, edge_proposals)

def initialize_vertex(G, vtx=None):
    # If no vertex is passed as arg, creates a vertex. Otherwise uses given.
    if vtx == None:
        vtx = graph.Vertex(G.num_people)

    vtx_type_dists = { t : td['likelihood'] for t, td in G.sim_params['vtx_types'].items() }
    vtx_types = list(vtx_type_dists.keys())
    vtx_type_likelihoods = [ vtx_type_dists[vt] for vt in vtx_types ]
    chosen_type = np.random.choice(vtx_types, p=vtx_type_likelihoods)
    vtx.data = copy.copy(G.sim_params['vtx_types'][chosen_type])
    vtx.data['type_name'] = chosen_type
    vtx.data.pop('likelihood')

    vtx.data['visited'] = set()

    contexts = np.random.choice(list(range(G.sim_params['context_count'])),
            replace=False, size=G.sim_params['k'])
    G.data[vtx] = { context : { G.sim_params['attr_func']() } for context in contexts }

    return vtx

# Graph creation
def attribute_network(n, params):
    # If clique is true, initialize network as a clique

    G = graph.Graph()
    G.data = {}
    G.sim_params = params

    vtx_set = []

    max_degree = G.sim_params['max_clique_size'] - 1
    max_indirect_edges = max_degree * (max_degree - 1) / 2
    cost_polynomial = [ max_indirect_edges, max_degree, -1 ]
    cost_roots = np.roots(cost_polynomial)

    G.sim_params['direct_cost'] = max(cost_roots)
    G.sim_params['indirect_cost'] = G.sim_params['direct_cost'] ** 2

    for i in range(n):
        vtx = graph.Vertex(i)
        vtx = initialize_vertex(G, vtx)
        vtx_set.append(vtx)

    G.vertices = vtx_set

    if params['seed_type'] == 'clique':

        # This may give a network that starts as over budget
        calc_utils(G)
        for v_idx in range(n):
            for u_idx in range(v_idx + 1, n):
                G.add_edge(G.vertices[v_idx], G.vertices[u_idx])
        return G
    elif params['seed_type'] == 'trivial':
        calc_utils(G)
        return G

    # Calculate edge utils
    calc_utils(G)

    # Set initial edges
    calc_edges(G)

    return G

# For adding to graph
def add_attr_graph_vtx(G, vtx=None):
    vtx = initialize_vertex(G, vtx)

    # Select initial neighbor candidate
    likelihoods = [ G.sim_params['edge_util_func'](vtx, u, G) for u in G.vertices ]
    scaled_likelihoods = []
    total_likelihood = sum(likelihoods)
    if total_likelihood > 0:
        scaled_likelihoods = [ lk / sum(likelihoods) for lk in likelihoods ]
    else:
        scaled_likelihoods = [ 1 / G.num_people for _ in range(G.num_people) ]
    candidate = np.random.choice(G.vertices, p=scaled_likelihoods)
    single_random_walk(G, vtx, candidate)

    G.vertices.append(vtx)
    calc_utils(G)
    return vtx

def simul_random_walk(G):
    # Take a random walk

    walk_lengths = { v : attr_util.random_walk_length(v, G) for v in G.vertices }
    pos_tokens = { v : v for v in G.vertices }

    # Reset visited vertices
    for v in G.vertices:
        v.data['visited'] = set()

    max_iters = max(walk_lengths.values())
    for _ in range(max_iters):
        if len(walk_lengths) == 0:
            break
        pop_list = []
        context_updates = {}
        for v in walk_lengths:
            if walk_lengths[v] == 0 or v.degree == 0:
                pop_list.append(v)
                continue
            cur_vtx = pos_tokens[v]
            edge_utils = [ e.util for e in cur_vtx.edges.values() ]
            next_vtx = np.random.choice(list(cur_vtx.edges.keys()),
                    p=[ eu / sum(edge_utils) for eu in edge_utils ])
            pos_tokens[v] = next_vtx
            if cur_vtx == v or cur_vtx in v.data['visited']:
                continue
            v.data['visited'].add(cur_vtx)
            context_updates[v] = G.sim_params['attr_copy'](v, next_vtx, G)
        for v in pop_list:
            walk_lengths.pop(v)
        for v, ctxts in context_updates.items():
            G.data[v] = ctxts

def single_random_walk(G, v, start=None):

    # Random walk of a single vertex
    v.data['visited'] = set()
    if start is not None:
        cur_vtx = start
        v.data['visited'].add(cur_vtx)
        G.data[v] = G.sim_params['attr_copy'](v, cur_vtx, G)
    else:
        cur_vtx = v

    if cur_vtx.degree == 0:
        return

    for i in range(attr_util.random_walk_length(v, G)):
        edge_utils = [ e.util for e in cur_vtx.edges.values() ]
        next_vtx = np.random.choice(list(cur_vtx.edges.keys()),
                p=[ eu / sum(edge_utils) for eu in edge_utils ])
        if cur_vtx in v.data['visited'] or cur_vtx == v or (i == 0 and start is not None):
            cur_vtx = next_vtx
        else:
            v.data['visited'].add(cur_vtx)
            G.data[v] = G.sim_params['attr_copy'](v, cur_vtx, G)
            cur_vtx = next_vtx

def seq_random_walk(G):

    # Reset visited vertices
    for v in G.vertices:
        v.data['visited'] = set()

    for v in np.random.permutation(G.vertices):
        single_random_walk(G, v)
