from collections import defaultdict

import numpy as np

from sim_lib.graph import Graph, Vertex, Edge
from sim_lib.util import gen_const_ratings, ring_slice, sample_powerlaw, is_connected

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

def stochastic_block_model(ncomm, comm_size, in_prob, out_prob, p, r):
    """
    ncomm is the number of communities
    comm_size is the size of each community
    in_prob the probability of an edge existing in each community
    out_prob the probability of an edge existing going out of a community
    p probability of transmission
    r time allocated per person

    NOTE: Only one vertex is given the optimal provider
    """
    stoch_block = Graph()

    n = ncomm * comm_size

    num_prov = int(n ** 0.5)
    provs = list(range(num_prov))
    global_rank = gen_const_ratings(provs)

    opt_prov_vtx = np.random.randint(0, n + 1)
    max_prov = max(global_rank, key=global_rank.get)
    min_prov = min(global_rank, key=global_rank.get)

    # Create vertex set of graph
    vtx_idx = 0
    def gen_vertex():
        nonlocal vtx_idx
        prov = min_prov
        if vtx_idx == opt_prov_vtx:
            prov = max_prov
        vtx = Vertex(r, prov, global_rank, vnum=vtx_idx)
        vtx_idx += 1
        return vtx

    vertices = [ gen_vertex() for i in range(n) ]
    stoch_block.vertices = vertices

    for vtx in stoch_block.vertices:

        # Iterate over possible neighbors of vidx
        for pnbor in stoch_block.vertices:
            if vtx == pnbor:
                continue

            # If pnbor in same community
            if (vtx.vnum // comm_size) == (pnbor.vnum // comm_size):
                if np.random.random() <= in_prob:
                    vtx.edges[pnbor] = Edge(p)
                    pnbor.edges[vtx] = Edge(p)
            else:
                if np.random.random() <= out_prob:
                    vtx.edges[pnbor] = Edge(p)
                    pnbor.edges[vtx] = Edge(p)

    return stoch_block
   
def kleinberg_grid(n, m, r, p, k=1, q=2):
    """
    Creates a (n x m) Kleinberg grid
    Each vertex has k random edges
    q is the clustering coefficient determining the edge probability
    r is the amount of resources allocated to each vertex
    p is the probability of information transmission
    """

    # Create graph
    k_grid = Graph()

    num_prov = int(n ** 0.5)
    provs = list(range(num_prov))
    global_rank = gen_const_ratings(provs)

    # Generate vertices
    vtx_idx = 0
    def gen_vertex():
        nonlocal vtx_idx
        prov = np.random.choice(provs)
        vtx = Vertex(r, prov, global_rank, vnum=vtx_idx)
        vtx_idx += 1
        return vtx

    vertices = [ gen_vertex() for i in range(n * m) ]
    k_grid.vertices = vertices

    # Generate edges
    rmo = lambda i, j : (i * n) + j
    vtx = lambda i : k_grid.vertices[i]
    for i in range(n):
        for j in range(m):

            cur_idx = rmo(i, j)

            # Add k random edges by distance
            # Sum to calculate probability of long distance edge
            lr_prob_cutoffs = [1]
            max_dist = max(i, n - i - 1) + max(j, m - j - 1)
            for d in range(2, max_dist + 1):
                lr_prob_cutoffs.append(lr_prob_cutoffs[-1] + d ** -q)

            for _ in range(k):
                dist_draw = np.random.uniform(high=lr_prob_cutoffs[-1])

                # Identify distance of target vertex
                found_dist = -1
                for pcum, d in zip(lr_prob_cutoffs, list(range(1, max_dist + 1))):
                    if dist_draw <= pcum:
                        found_dist = d
                        break

                # No point in selecting, all vertices at dist 1 already neighbors
                if found_dist == 1:
                    continue

                # Get vertices found_dist away, select one at random
                fd_vtxs = []
                for it in range(found_dist):

                    # Get vertices in Quad1
                    ur_i = i - (found_dist - it)
                    ur_j = j + (it)
                    if not(ur_i < 0 or ur_j > m - 1):
                        fd_vtxs.append(vtx(rmo(ur_i, ur_j)))

                    # Get vertices in Quad2
                    ul_i = i - (found_dist - it)
                    ul_j = j - (it)
                    if not (ul_i < 0 or ul_j < 0):
                        fd_vtxs.append(vtx(rmo(ul_i, ul_j)))

                    # Get vertices in Quad3
                    bl_i = i + (found_dist - it)
                    bl_j = j - (it)
                    if not (bl_i > n - 1 or bl_j < 0):
                        fd_vtxs.append(vtx(rmo(bl_i, bl_j)))

                    # Get vertices in Quad4
                    br_i = i + (found_dist - it)
                    br_j = j + (it)
                    if not (br_i > n - 1 or br_j > m - 1):
                        fd_vtxs.append(vtx(rmo(br_i, br_j)))

                lr_vtx = np.random.choice(fd_vtxs)
                vtx(cur_idx).edges[lr_vtx] = Edge(p)
                lr_vtx.edges[vtx(cur_idx)] = Edge(p)

            # Only need to add edge for down and right

            # Skip if on last column
            if j != m - 1:
                right_idx = rmo(i, j + 1)
                vtx(cur_idx).edges[vtx(right_idx)] = Edge(p)
                vtx(right_idx).edges[vtx(cur_idx)] = Edge(p)

            # Skip if on bottom row
            if i != n - 1:
                down_idx = rmo(i + 1, j)
                vtx(cur_idx).edges[vtx(down_idx)] = Edge(p)
                vtx(down_idx).edges[vtx(cur_idx)] = Edge(p)

    return k_grid

def reduce_providers_simplest(G):
    """
    Reduces a graph G to the simplest case where exactly one vertex
    has the optimal providers and all else have the worst
    """
    max_prov = max([ vtx.provider for vtx in G.vertices ])
    for vtx in G.vertices:
        vtx.provider = 0
    max_vtx = np.random.randint(len(G.vertices))
    G.vertices[max_vtx].provider = max_prov

def powerlaw_dist_time(G, plaw_exp, plaw_coeff=1):
    """
    Redistribute time per person based on specified powerlaw
    """
    tot_time = sum([ vtx.time for vtx in G.vertices ])
    plaw_times = sample_powerlaw(len(G.vertices), tot_time, plaw_exp, plaw_coeff, True)
    for idx, vtx in enumerate(G.vertices):
        vtx.time = plaw_times[idx]

