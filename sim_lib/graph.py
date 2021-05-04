"""
Implementation for a graph
Includes methods for generating graphs
"""

import numpy as np
from collections import OrderedDict, defaultdict

class Vertex:
    """
    Vertex to be used with graph
    Each vertex represents a person in social network
    edges: Dict of edges. Keyed by destination vertex
    vnum: Label/idx for numeric representation of vertex, defaults to global_vnum
    if one not specified
    global_vnum: Global vertex number, should be unique for each vertex across
    all graphs generated
    """

    vtx_count = 0

    def __init__(self, vnum):
        self.edges = OrderedDict()
        self.vnum = vnum

        Vertex.vtx_count += 1

        # For data about the vertex in model
        self.data = None

        # Context : { Attribute : Set( Vertices with attribute ) }
        self.attr_obs = {}
        self.lifetime_visited = set()

        # For params relating to the simulation
        self.sim_params = {}

    ##########################
    # Attribute observations #
    ##########################

    def init_attr_obs(self, G):
        self.attr_obs = { ctxt : defaultdict(set) \
                for ctxt in range(G.sim_params['context_count']) }

    def update_attr_obs(self, G, v):
        
        # Updates with attributes observed in vertex v
        for ctxt in G.data[v]:
            for attr in G.data[v][ctxt]:
                self.attr_obs[ctxt][attr].add(v)

    def attr_likelihood(self, ctxt, attr):

        # Estimates likelihood of observing attribute
        return len(self.attr_obs[ctxt][attr]) / len(self.lifetime_visited)

    @property
    def degree(self):
        return len(self.edges)

    @property
    def nbors(self):
        return list(self.edges.keys())

    @property
    def nbor_set(self):
        return set(self.edges.keys())

    @property
    def sum_edge_util(self):
        return sum([ e.util for e in self.edges.values() ])

    def is_nbor(self, v):
        return v in self.edges

    @property
    def nborhood_degree(self):
        nbor_edges = 0
        for u in self.nbors:
            for w in u.nbors:
                if self.is_nbor(w):
                    nbor_edges += 1
        return nbor_edges

    def __repr__(self):
        return 'Vertex {0}'.format(self.vnum)

class Edge:
    """
    Edge between vertices in graph
    Represents connection between two people
    util: Utility on an edge

    """

    def __init__(self, util):
        self.util = util
        self.data = None

class Graph:
    """
    Graph implementation
    vertices: List of vertices
    """

    def __init__(self):
        self.vertices = []

        # For data about the graph in model
        self.data = None

        # Potential utility matrix
        self.potential_utils = []

    @property
    def num_people(self):
        return len(self.vertices)

    def add_edge(self, u, v):
        """
        Adds edge between u and v
        """
        assert (v in u.edges) == (u in v.edges), 'connection must be symmetric'
        if not self.are_neighbors(u, v):
            u.edges[v] = Edge(self.potential_utils[u.vnum][v.vnum])
            v.edges[u] = Edge(self.potential_utils[v.vnum][u.vnum])

    def remove_edge(self, u, v, reflexive=True):
        """
        Removes edge between u and v if exists
        If reflexive deletes uv and vu, else just deletes edge uv
        """
        edge_util = u.edges[v].util

        if v in u.edges:
            u.edges[v].data = None
            u.edges.pop(v)
        if reflexive and u in v.edges:
            v.edges[u].data = None
            v.edges.pop(u)

    @property
    def edge_count(self):
        return sum([ v.degree for v in self.vertices ]) // 2

    def are_neighbors(self, u, v):
        return (u in v.nbors) and (v in u.nbors)

    @property
    def adj_matrix(self):

        # Returns numpy matrix indexed by vnum
        adj_mat = np.zeros((G.num_people, G.num_people))
        for idx, v in enumerate(G.vertices):
            for u in G.vertices[idx + 1:]:
                if v.is_nbor(u):
                    adj_mat[v.vnum][u.vnum] = 1
        return adj_matrix
