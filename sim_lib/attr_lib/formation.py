from collections.abc import Iterable
import copy

import networkx as nx
import numpy as np
import math

import sim_lib.graph as graph
import sim_lib.attr_lib.util as attr_lib_util

# Edge selection
def calc_utils(G):
    util_mat = np.zeros((G.num_people, G.num_people))
    for i, u in enumerate(G.vertices):
        for v in G.vertices[i + 1:]:
            util_mat[u.vnum][v.vnum] = u.data['edge_attr_util'](u, v, G)
            util_mat[v.vnum][u.vnum] = v.data['edge_attr_util'](v, u, G)
    G.potential_utils = util_mat
    return G.potential_utils

def calc_edges(G, walk_proposals='fof_pe', dunbar=150):
   
    edge_proposals = {}

    def get_fof(v):
        d2_vertices = set()
        for u in v.nbors:
            d2_vertices = d2_vertices.union(u.nbor_set)
        get_vnum = lambda v : v.vnum
        d2_vertices.remove(v)
        return sorted(list(d2_vertices), key=get_vnum)

    # walk_proposals to override 
    if walk_proposals == 'global':
        edge_proposals = { v : [ u for u in G.vertices if u != v ] \
                for v in G.vertices if attr_lib_util.remaining_budget(v, G) > 0 }
    elif walk_proposals == 'fof':
        for v in G.vertices:
            if v.degree == 0 or attr_lib_util.remaining_budget(v, G) <= 0:
                continue
            edge_proposals[v] = get_fof(v) 
    elif walk_proposals == 'fof_pe':
        for v in G.vertices:
            if v.degree == 0 or attr_lib_util.remaining_budget(v, G) <= 0:
                continue
            v_fof = get_fof(v)
            v_fof_pe = []
            v_attr_util = v.data['total_attr_util'](v, G)
            for u in v_fof:
                G.add_edge(v, u)
                if v.data['total_attr_util'](v, G) - v_attr_util > 0:
                    v_fof_pe.append(u)
                G.remove_edge(v, u)
            edge_proposals[v] = v_fof_pe
    else:
        for u in G.vertices:
            edge_proposals[u] = []

            # Only need to check visited edges for proposal
            for v in u.data['visited']:
                if G.are_neighbors(u, v) or u == v:
                    continue
               
                edge_proposals[u].append(v)

    G.sim_params['edge_selection'](G, edge_proposals)

def initialize_vertex(G, vtx=None):
    # If no vertex is passed as arg, creates a vertex. Otherwise uses given.
    if vtx == None:
        vtx = graph.Vertex(G.num_people)

    vtx.init_attr_obs(G)
    vtx_type_dists = { t : td['likelihood'] for t, td in G.sim_params['vtx_types'].items() }
    vtx_types = list(vtx_type_dists.keys())
    vtx_type_likelihoods = [ vtx_type_dists[vt] for vt in vtx_types ]
    chosen_type = np.random.choice(vtx_types, p=vtx_type_likelihoods)
    vtx.data = copy.copy(G.sim_params['vtx_types'][chosen_type])
    vtx.data['type_name'] = chosen_type
    vtx.data.pop('likelihood')

    vtx.data['visited'] = set()

    G.data[vtx] = vtx.data['init_attrs'](vtx, G)

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
    G.sim_params['max_degree'] = math.floor(1 / G.sim_params['direct_cost'])

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
        #edge_prob = ((1 + (2 ** -10)) * math.log(n)) / n
        edge_prob = 1 / n
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
    else:
        calc_edges(G)

    return G

# For adding to graph
def add_attr_graph_vtx(G, vtx=None, walk=False):
    vtx = initialize_vertex(G, vtx)

    # Select initial neighbor candidate
    likelihoods = [ vtx.data['edge_attr_util'](vtx, u, G) for u in G.vertices ]
    scaled_likelihoods = []
    total_likelihood = sum(likelihoods)
    if total_likelihood > 0:
        scaled_likelihoods = [ lk / sum(likelihoods) for lk in likelihoods ]
    else:
        scaled_likelihoods = [ 1 / G.num_people for _ in range(G.num_people) ]
    candidate = np.random.choice(G.vertices, p=scaled_likelihoods)

    if walk:
        single_random_walk(G, vtx, candidate)

    G.vertices.append(vtx)
    calc_utils(G)

    return vtx

def simul_random_walk(G):
    # Take a random walk

    walk_lengths = { v : attr_lib_util.random_walk_length(v, G) for v in G.vertices }
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

    for i in range(attr_lib_util.random_walk_length(v, G)):
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

def seq_global_walk(G, constrain_walk=False):

    # Have everyone "walk" the entire graph in sequential order
    for v in G.vertices:
        v.data['visited'] = set()
        can_add = attr_lib_util.remaining_budget(v, G) < G.sim_params['direct_cost']
        if constrain_walk and can_add:
            continue

        for u in G.vertices:
            if u == v:
                continue

            v.data['visited'].add(u)
            #G.data[v] = G.sim_params['attr_copy'](v, u, G)
            v.update_attr_obs(G, u)

