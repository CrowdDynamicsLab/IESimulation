import numpy as np

from graph import Graph, Vertex, Edge
from graph_util import gen_const_ratings, ring_slice

def const_kregular(k, n, itrate, time_alloc):
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
    def gen_vertex():
        return Vertex(time_alloc, np.random.choice(provs), global_rank)

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
    init_g = const_kregular(k, n, close_trate, time_alloc)
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
