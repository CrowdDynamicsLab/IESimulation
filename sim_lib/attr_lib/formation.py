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
    edge_util = G.sim_params['edge_util_func'](u, v, G)
    u_edge_prob = G.sim_params['edge_prob_func'](u, edge_util)
    v_edge_prob = G.sim_params['edge_prob_func'](v, edge_util)
    return np.random.random() <= u_edge_prob * v_edge_prob

def add_edge(u, v, G):
    assert (v in u.edges) == (u in v.edges), 'connection must be symmetric'
    if not G.are_neighbors(u, v):
        G.add_edge(u, v, 1)
        edge_util = G.sim_params['edge_util_func'](u, v, G)
        u.data += edge_util
        v.data += edge_util

def remove_edge(u, v, G):
    edge_util = G.sim_params['edge_util_func'](u, v, G)

    u.edges[v].data = None
    u.edges.pop(v)
    u.data -= edge_util

    v.edges[u].data = None
    v.edges.pop(u)
    v.data -= edge_util
        
def calc_edges(G, dunbar=150):
    edge_candidates = []
    
    # NOTE: May eventually constrain to "known" vertices
    for uidx in range(G.num_people - 1):
        u = G.vertices[uidx]
        
        for vidx in range(uidx + 1, G.num_people):
            v = G.vertices[vidx]
            
            if has_edge(u, v, G):
                edge_candidates.append((u, v))
            elif G.are_neighbors(u, v) and not has_edge(u, v, G):
                remove_edge(u, v, G)
                
    # Add valid edges in random order until vertex runs out of budget
    np.random.shuffle(edge_candidates)
    for ec_u, ec_v in edge_candidates:
        add_edge(ec_u, ec_v, G)
        u_cost = attr_util.calc_cost(ec_u,
                G.sim_params['direct_cost'], G.sim_params['indirect_cost'], G)
        v_cost = attr_util.calc_cost(ec_v,
                G.sim_params['direct_cost'], G.sim_params['indirect_cost'], G)
        budget = dunbar * G.sim_params['direct_cost']
        if u_cost > budget or v_cost > budget:
            remove_edge(u, v, G)

# Graph creation
def attribute_network(n, params):
    vtx_set = []

    for i in range(n):
        vtx = graph.Vertex(0, 0, {0 : 0}, i)
        vtx.data = 0
        vtx_set.append(vtx)

    G = graph.Graph()
    G.vertices = vtx_set
    
    G.data = {}
    G.sim_params = params
    
    # Initialize contexts
    for vtx in G.vertices:
        contexts = np.random.choice(list(range(params['context_count'])), size=params['k'])
        G.data[vtx] = { context : [ params['attr_func']() ] for context in contexts }
    
    # Set initial edges
    calc_edges(G)
    
    return G

# For adding to graph
def add_attr_graph_vtx(G, v):
    contexts = np.random.choice(list(range(G.sim_params['context_count'])), size=G.sim_params['k'])
    G.data[v] = { context : [ G.sim_params['attr_func']() ] for context in contexts }

    G.vertices.append(v)
    return v

