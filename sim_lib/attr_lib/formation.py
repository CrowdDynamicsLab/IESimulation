from collections.abc import Iterable
import copy

import networkx as nx
import numpy as np

import sim_lib.graph as graph
import sim_lib.graph_create as gc
import sim_lib.graph_networkx as gnx
import sim_lib.util as util
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

def calc_edges(G, dunbar=150):
    
    edge_proposals = {}

    # NOTE: May eventually constrain to "known" vertices
    for u in G.vertices:
        edge_proposals[u] = []
        for v in G.vertices:
            if G.are_neighbors(u, v) or u == v:
                continue
            
            edge_util = G.potential_utils[u.vnum][v.vnum]
            if G.sim_params['edge_proposal'](u, edge_util) >= np.random.random():
                edge_proposals[u].append(v)

    G.sim_params['edge_selection'](G, edge_proposals)

# Graph creation
def attribute_network(n, params):
    vtx_set = []

    vtx_type_dists = { t : td['likelihood'] for t, td in params['vtx_types'].items() }
    vtx_types = list(vtx_type_dists.keys())
    vtx_type_likelihoods = [ vtx_type_dists[vt] for vt in vtx_types ]
    for i in range(n):
        vtx = graph.Vertex(i)
        chosen_type = np.random.choice(vtx_types, p=vtx_type_likelihoods)
        vtx.data = copy.copy(params['vtx_types'][chosen_type])
        vtx.data['type_name'] = chosen_type
        vtx.data.pop('likelihood')
        vtx_set.append(vtx)

    G = graph.Graph()
    G.vertices = vtx_set
    
    G.data = {}
    G.sim_params = params
    
    # Initialize contexts
    for vtx in G.vertices:
        contexts = np.random.choice(list(range(params['context_count'])),
                replace=False, size=params['k'])
        G.data[vtx] = { context : { params['attr_func']() } for context in contexts }
    
    # Calculate edge utils
    calc_utils(G)

    # Set initial edges
    calc_edges(G)

    return G

# For adding to graph
def add_attr_graph_vtx(G, v):
    contexts = np.random.choice(list(range(G.sim_params['context_count'])),
            replace=False, size=G.sim_params['k'])
    G.data[v] = { context : { G.sim_params['attr_func']() } for context in contexts }

    G.vertices.append(v)
    return v

def simul_random_walk(G):
    # Take a random walk

    walk_lengths = { v : attr_util.random_walk_length(v, G) for v in G.vertices }
    pos_tokens = { v : v for v in G.vertices }

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
            if cur_vtx == v:
                continue
            context_updates[v] = G.sim_params['attr_copy'](v, next_vtx, G)
        for v in pop_list:
            walk_lengths.pop(v)
        for v, ctxts in context_updates.items():
            G.data[v] = ctxts

def seq_random_walk(G):
    for v in np.random.permutation(G.vertices):
        cur_vtx = v
        for _ in range(attr_util.random_walk_length(v, G)):
            edge_utils = [ e.util for e in cur_vtx.edges.values() ]
            next_vtx = np.random.choice(list(cur_vtx.edges.keys()),
                    p=[ eu / sum(edge_utils) for eu in edge_utils ])
            cur_vtx = next_vtx
            if cur_vtx == v:
                continue
            G.data[v] = G.sim_params['attr_copy'](v, cur_vtx, G)
            
