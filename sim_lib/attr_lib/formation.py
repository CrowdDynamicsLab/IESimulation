import networkx as nx
import numpy as np
from collections.abc import Iterable

import sim_lib.graph as graph
import sim_lib.graph_create as gc
import sim_lib.graph_networkx as gnx
import sim_lib.util as util

import sim_lib.attr_lib.util as attr_util

# Edge selection
def has_edge(u, v, G):
    edge_util = G.potential_utils[u.vnum][v.vnum]
    u_edge_prob = G.sim_params['edge_prob_func'](u, edge_util)
    v_edge_prob = G.sim_params['edge_prob_func'](v, edge_util)
    return np.random.random() <= u_edge_prob * v_edge_prob

def add_edge(u, v, G):
    assert (v in u.edges) == (u in v.edges), 'connection must be symmetric'
    if not G.are_neighbors(u, v):
        edge_util = G.potential_utils[u.vnum][v.vnum]
        G.add_edge(u, v, edge_util)
        u.data += edge_util
        v.data += edge_util

def remove_edge(u, v):
    edge_util = u.edges[v].util

    u.edges[v].data = None
    u.edges.pop(v)
    u.data -= edge_util

    v.edges[u].data = None
    v.edges.pop(u)
    v.data -= edge_util
    
def calc_utils(G):
    util_mat = np.zeros((G.num_people, G.num_people))
    for i, u in enumerate(G.vertices):
        for v in G.vertices[i + 1:]:
            util_mat[u.vnum][v.vnum] = G.sim_params['edge_util_func'](u, v, G)
            util_mat[v.vnum][u.vnum] = util_mat[u.vnum][v.vnum]
    G.potential_utils = util_mat
    return util_mat

def budget_resolution(v, G):

    # When v goes over budget due to some edge not incident to v
    # v must drop an edge

    inv_util_set = [ 1 / G.potential_utils[v.vnum][u.vnum] for u in v.nbors ]
    dropped_nbor = np.random.choice(v.nbors,
            p= [ iut / sum(inv_util_set) for iut in inv_util_set ])
    remove_edge(v, dropped_nbor)

def calc_edges(G, dunbar=150):
    edge_candidates = []
    
    # NOTE: May eventually constrain to "known" vertices
    for uidx in range(G.num_people - 1):
        u = G.vertices[uidx]
        
        for vidx in range(uidx + 1, G.num_people):
            v = G.vertices[vidx]
            
            potential_edge = has_edge(u, v, G)
            if potential_edge:
                edge_candidates.append((u, v))
            elif G.are_neighbors(u, v) and not potential_edge:
                remove_edge(u, v)
                
    # Add valid edges in random order until vertex runs out of budget
    np.random.shuffle(edge_candidates)
    for ec_u, ec_v in edge_candidates:
        add_edge(ec_u, ec_v, G)
        for v in G.vertices:
            itr_count = 0
            while attr_util.remaining_budget(v, G, dunbar) < 0:
                budget_resolution(v, G)
                itr_count += 1
                assert itr_count <= len(G.vertices), \
                        'Budget resolution occured more than number of vertices'

# Graph creation
def attribute_network(n, params):
    vtx_set = []

    for i in range(n):
        vtx = graph.Vertex(i)
        vtx.data = 0
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

def attr_copy(u, v, G):
    # Copy an attribute from v to u
    v_contexts = list(G.data[v].keys())
    v_context_sizes = [ len(G.data[v][vctxt]) for vctxt in v_contexts ]
    v_context = np.random.choice(v_contexts,
            p=[ csize / sum(v_context_sizes) for csize in v_context_sizes ])
    v_attr = np.random.choice(list(G.data[v][v_context]))

    if v_context in G.data[u]:
        G.data[u][v_context].add(v_attr)
    else: # Case where context may be switched
        G.data[u][v_context] = { v_attr }
        u_contexts = list(G.data[u].keys())
        u_context_sizes = [ len(G.data[u][uctxt]) for uctxt in u_contexts ]
        context_count = G.sim_params['k']
        u_context_set = np.random.choice(u_contexts,
                size=context_count, replace=False,
                p=[ csize / sum(u_context_sizes) for csize in u_context_sizes ])
        u_context_map = { ctxt : G.data[u][ctxt] for ctxt in u_context_set }
        G.data[u] = u_context_map

def random_walk(u, G, k):
    # Take a random walk starting at u on G of length k

    if u.degree == 0:
        return

    starting_vtx = u
    visited = []
    cur_vtx = starting_vtx
    for _ in range(k):
        edge_utils = [ e.util for e in cur_vtx.edges.values() ]
        next_vtx = np.random.choice(list(cur_vtx.edges.keys()),
                p=[ eu / sum(edge_utils) for eu in edge_utils ])
        cur_vtx = next_vtx
        if cur_vtx == starting_vtx:
            continue
        attr_copy(starting_vtx, cur_vtx, G)
