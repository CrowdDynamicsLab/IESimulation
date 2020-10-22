import networkx as nx
import numpy as np

import sim_lib.graph as graph
import sim_lib.graph_create as gc
import sim_lib.graph_networkx as gnx
import sim_lib.util as util

import sim_lib.attr_lib.util as attr_util

def pick_attr(vtx, G, params, init=False):
    if init:
        return np.random.choice(list(range(params['attr_value_count'])),
                                params['attr_count'], replace=False,
                                p=params['attr_prob'])
    
    cur_contexts = G.data[vtx]
    for attr_idx in range(len(cur_contexts)):        
        if np.random.random() <= params['context_switch_prob']:
            switched = False
            while not switched:
                new_context = np.random.choice(list(range(params['attr_value_count'])),
                                 1, p=params['attr_prob'])
                if new_context not in cur_contexts:
                    G.data[vtx][attr_idx] = new_context
                    switched = True
                    
    return G.data[vtx]

# Edge selection
def has_edge(u, v, G, params):
    edge_prob = params['prob_func'](u, v, G, params['attr_prob'])
    return np.random.random() <= edge_prob

def add_edge(u, v, G):
    assert (v in u.edges) == (u in v.edges), 'connection must be symmetric'
    if not G.are_neighbors(u, v):
        G.add_edge(u, v, 1)
        
def calc_edges(G, params, dunbar=150):
    edge_candidates = []
    
    for uidx in range(G.num_people - 1):
        u = G.vertices[uidx]
        
        for vidx in range(uidx + 1, G.num_people):
            v = G.vertices[vidx]
            
            if has_edge(u, v, G, params):
                edge_candidates.append((u, v))
            elif G.are_neighbors(u, v) and not has_edge(u, v, G, params):
                G.remove_edge(u, v)
                
    np.random.shuffle(edge_candidates)
    for ec_u, ec_v in edge_candidates:
        add_edge(ec_u, ec_v, G)
        u_cost = attr_util.calc_cost(ec_u, params['direct_cost'], params['indirect_cost'], G)
        v_cost = attr_util.calc_cost(ec_v, params['direct_cost'], params['indirect_cost'], G)
        budget = dunbar * params['direct_cost']
        if u_cost > budget or v_cost > budget:
            G.remove_edge(u, v)

# Graph creation
def attribute_block_graph(n, params):
    vtx_set = []

    for i in range(n):
        vtx = graph.Vertex(0, 0, {0 : 0}, i)
        vtx.data = 0
        vtx_set.append(vtx)

    G = graph.Graph()
    G.vertices = vtx_set
    
    G.data = {}
    
    # Initialize contexts
    for vtx in G.vertices:
        G.data[vtx] = pick_attr(vtx, G, params, init=True)
    
    # Set initial edges
    calc_edges(G, params)
    
    return G

# For adding to graph
def add_attr_graph_vtx(G, v, params):
    v.data = 0
    G.data[v] = pick_attr(v, G, params, init=True)

    G.vertices.append(v)
    return v

def update_attr_dim(G, params):
    
    # Pick new values
    for v in G.vertices:
        v.data += 1
        G.data[v] = pick_attr(v, G, params)

