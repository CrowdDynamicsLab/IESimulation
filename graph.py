"""
Implementation for a graph
Includes methods for generating graphs
"""

import numpy as np

class Vertex:
    """
    Vertex to be used with graph
    Each vertex represents a person in social network
    edges: Dict of edges. Keyed by destination vertex
    time: Amount of time this person has allocated
    provider: Service provider the current person has
    ratings: Rating the current patient gives all providers
    vnum: Vertex number, should be unique for each vertex across
    all graphs generated
    """

    vtx_count = 0

    def __init__(self, time, init_prov, ratings):
        self.edges = {}
        self.time = time
        self.provider = init_prov
        self.prov_rating = ratings
        self.vnum = Vertex.vtx_count
        Vertex.vtx_count += 1

    @property
    def utility(self):
        return self.prov_rating[self.provider]

    @property
    def degree(self):
        return len(self.edges)

    def __repr__(self):
        return 'Vertex {0}'.format(self.vnum)

class Edge:
    """
    Edge between vertices in graph
    Represents connection between two people
    trate: Probability of information transition

    Transmission rate is currently assumed to be a constant
    """

    def __init__(self, trate):
        self.trate = trate

class Graph:
    """
    Graph implementation
    vertices: List of vertices
    """

    def __init__(self):
        self.vertices = []

    @property
    def num_people(self):
        return len(self.vertices)

def gen_const_ratings(provs):
    """
    Returns a dict of ratings of provider : int
    representing ratings of each respective provider
    """
    ratings = np.linspace(0, 1, len(provs))
    np.random.shuffle(ratings)
    return { provs[didx] : ratings[didx] for didx in range(len(provs)) }

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
            cvert.edges[kreg.vertices[ni %n]] = Edge(itrate)
        if k % 2 == 1:
            op_idx = (i + (n // 2)) % n
            cvert.edges[kreg.vertices[op_idx]] = Edge(itrate)

    return kreg

def calc_diameter(k, n):
    """
    For a k regular graph with n vertices, calculates the
    diameter based on the generation scheme used in const_kregular
    """

    #n should always be even
    ring_dist = n // 2
    m = k // 2
    return (ring_dist // m) + (ring_dist % m)
