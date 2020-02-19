from collections import defaultdict

import numpy as np

from graph import Graph, Vertex, Edge
from util import gen_const_ratings, ring_slice, sample_powerlaw, is_connected

def force_connected(graph_func):
    """
    Forces a graph creation func to return a connected graph
    """

    def con_graph_func(*args):
        G = graph_func(*args)
        while not is_connected(G):
            G = graph_func(*args)
        return G
    return con_graph_func

def ring_lattice(k, n, itrate, time_alloc):
    """
    Generates a simple k-regular graph with n vertices
    If n <= k, n will be rounded up to k + 1
    Creates sqrt(n) providers
    Randomly assigns ratings for each member of network to each provider
    Randomly assigns a initial provider to each patient

    k: Regularity of graph
    n: Min number of vertices
    itrate: Initial transmission rate
    time_alloc: Initial time allocation

    Note: This is not guaranteed to generate a simple connected graph
    """
    kreg = Graph()
    
    #Min number vertices required
    if n <= k:
        n = k + 1

    #Must have even number of vertices by handshake lemma
    if n % 2 == 1:
        n += 1

    #Initialize providers
    num_prov = int(n ** 0.5)
    
    #Increasing number of providers increases number of non-convergence cases
    #but does not create case where converges in >3 iterations
    #num_prov = n
    provs = list(range(num_prov))

    #Assume rankings are objective and global
    global_rank = gen_const_ratings(provs)

    #Create vertex set of graph
    vtx_idx = 0
    def gen_vertex():
        nonlocal vtx_idx
        vtx = Vertex(time_alloc, np.random.choice(provs), global_rank, vnum=vtx_idx)
        vtx_idx += 1
        return vtx

    vertices = [ gen_vertex() for i in range(n) ]
    kreg.vertices = vertices

    #Generate cycle
    step_size = -1

    #Handle special case, k == 1
    if k == 1:
        for i in range(0, n, 2):
            lvert = kreg.vertices[i]
            rvert = kreg.vertices[(i + 1) % n]
            edgeL = Edge(itrate)
            edgeR = Edge(itrate)
            lvert.edges[rvert](edgeL)
            rvert.edges[lvert](edgeR)
            return kreg

    m = k // 2
    for i in range(n):
        cvert = kreg.vertices[i]
        prev_start = (i - 1)
        prev_stop = (prev_start - m)
        for pi in range(prev_start, prev_stop, -1):
            cvert.edges[kreg.vertices[pi % n]] = Edge(itrate)
        next_start = (i + 1)
        next_stop = (next_start + m)
        for ni in range(next_start, next_stop):
            cvert.edges[kreg.vertices[ni %n ]] = Edge(itrate)
        if k % 2 == 1:
            op_idx = (i + (n // 2)) % n
            cvert.edges[kreg.vertices[op_idx]] = Edge(itrate)

    return kreg

def watts_strogatz(n, k, b, close_trate, far_trate, time_alloc):
    """
    Creates a Watts-Strogatz model graph
    n: Number of vertices
    k: Average degree
    b: Rewiring factor beta
    close_trate: Transmission rate of close (initial) homophily edges
    far_trate: Transmission rate of further (rewired) heterophily edges
    time_alloc: Initial time allocation
    """

    init_g = ring_lattice(k, n, close_trate, time_alloc)
    min_vtx_num = min([ vtx.vnum for vtx in init_g.vertices ])
    def non_nbor(u, v):
        return (u not in v.edges and v not in u.edges and u != v)
    for vtx in init_g.vertices:

        #Possibly rewire rightmost k/2 edges
        start_idx = (vtx.vnum - min_vtx_num + 1) % n
        end_idx = (vtx.vnum - min_vtx_num + 1 + (k // 2)) % n
        for nbor in ring_slice(init_g.vertices, start_idx, end_idx):

            #With probability b rewire the edge
            if np.random.rand() <= b:
                non_nbors = [ v for v in init_g.vertices if non_nbor(vtx, v) ]
                new_edge_idx = np.random.randint(len(non_nbors))
                new_nbor = non_nbors[new_edge_idx]

                #Rewire edges
                vtx.edges.pop(nbor)
                nbor.edges.pop(vtx)
                vtx.edges[new_nbor] = Edge(far_trate)
                new_nbor.edges[vtx] = Edge(far_trate)
    return init_g

def erdos_renyi(n, ep, p, time_alloc):
    """
    Creates an erdos-renyi graph
    n: Number of edges
    ep: Probability of an edge forming between any pair of vertices
    p: Transmission probability
    time_alloc: Time allocated to each vertex
    """

    G = ring_lattice(0, n, p, time_alloc)

    for vtx in G.vertices:
        for pnbor in G.vertices:
            if vtx == pnbor or (pnbor in vtx.edges):
                continue
            if np.random.rand() < ep:
                vtx.edges[pnbor] = Edge(p)
                pnbor.edges[vtx] = Edge(p)

    return G

def configuration_model(n, degree_seq, p, time_alloc):
    """
    Creates a coniguration model graph by the following parameters
    n: number of vertices
    degree_seq: degree of each vertex
    p: probability of transmission
    time_alloc: time allocated to each vertex

    NOTE: Uses repeated configuration model as described by
    https://arxiv.org/pdf/1509.06985.pdf
    """
    assert len(degree_seq) == n, 'Degree sequence must be of same length as number vertices'
    assert sum(degree_seq) % 2 == 0, 'Degree sequence must have even sum'

    G = ring_lattice(0, n, p, time_alloc)

    # For randomly matching, edge_stubs is flattened edge_acc
    vtx_deg = zip(G.vertices, degree_seq)
    edge_acc = [ [vtx] * deg for vtx, deg in vtx_deg ]
    edge_stubs_orig = [ vtx for vtx_list in edge_acc for vtx in vtx_list ]

    G_simple = False

    def next_vtx_start(vtx_list, cur_v):
        for idx, vtx in enumerate(vtx_list):
            if vtx.vnum != cur_v.vnum:
                return idx
        return -1

    #Repeatedly find matchings until G is a simple graph
    edge_matches = defaultdict(set)
    while not G_simple:

        edge_stubs = edge_stubs_orig.copy()
        edge_matches = defaultdict(set)

        loop_found = False
        while edge_stubs:
            cur_stub = edge_stubs.pop(0)

            #Finds index of next vertex that is not cur_stub
            match_start_idx = next_vtx_start(edge_stubs, cur_stub)

            #Should only occur if only cur_stub left (maybe common in powerlaw degree dist)
            if match_start_idx == -1:
                loop_found = True
                break

            match_idx = np.random.randint(low=match_start_idx, high=len(edge_stubs))
            matched = edge_stubs.pop(match_idx)
            edge_matches[cur_stub].add(matched)

        #Only check for multi-edges if no loops
        multiedge_found = False
        if not loop_found:
            for vtx, deg in vtx_deg:
                if len(edge_matches[cur_stub]) != deg:
                    multiedge_found = True
                    break

        if not (loop_found or multiedge_found):
            G_simple = True

    for vtx, edge_set in edge_matches.items():
        for nbor in edge_set:

            #Add edge
            vtx.edges[nbor] = Edge(p)
            nbor.edges[vtx] = Edge(p)

    return G

def reduce_providers_simplest(G):
    """
    Reduces a graph G to the simplest case where exactly one vertex
    has the optimal providers and all else have the worst
    """
    max_prov = max([ vtx.provider for vtx in G.vertices ])
    for vtx in G.vertices:
        vtx.provider = 0
    G.vertices[0].provider = max_prov

def powerlaw_dist_time(G, plaw_exp, plaw_coeff=1):
    """
    Redistribute time per person based on specified powerlaw
    """
    tot_time = sum([ vtx.time for vtx in G.vertices ])
    plaw_times = sample_powerlaw(len(G.vertices), tot_time, plaw_exp, plaw_coeff, True)
    for idx, vtx in enumerate(G.vertices):
        vtx.time = plaw_times[idx]

