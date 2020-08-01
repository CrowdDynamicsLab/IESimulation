"""
Implementation for a graph
Includes methods for generating graphs
"""

import numpy as np
from collections import OrderedDict

class Vertex:
    """
    Vertex to be used with graph
    Each vertex represents a person in social network
    edges: Dict of edges. Keyed by destination vertex
    time: Amount of time this person has allocated
    provider: Service provider the current person has
    ratings: Rating the current patient gives all providers
    vnum: Label/idx for numeric representation of vertex, defaults to global_vnum
    if one not specified
    global_vnum: Global vertex number, should be unique for each vertex across
    all graphs generated
    interaction_count: The number of interactions this person has had
    """

    vtx_count = 0

    def __init__(self, time, init_prov, ratings, vnum):
        self.edges = OrderedDict()
        self.time = time
        self.provider = init_prov
        self.prov_rating = ratings
        self.vnum = vnum

        # may eventually add per neighbor
        self.interactions = {'total' : 0,
                             'initiated' : 0,
                             'received' : 0}
        self.initial = {'time' : time, 'provider' : init_prov, 'prov_rating' : ratings}

    @property
    def utility(self):
        return self.prov_rating[self.provider]

    @property
    def degree(self):
        return len(self.edges)

    @property
    def nbors(self):
        return list(self.edges.keys())

    @property
    def total_ints(self):
        return self.interactions['total']

    @property
    def init_ints(self):
        return self.interactions['initiated']

    @property
    def recv_ints(self):
        return self.interactions['received']

    def add_init_int(self):
        self.interactions['total'] += 1
        self.interactions['initiated'] += 1

    def add_recv_int(self):
        self.interactions['total'] += 1
        self.interactions['received'] += 1

    def reset(self):
        """
        Reset to initial state
        """
        self.interactions = {'total' : 0,
                             'initiated' : 0,
                             'received' : 0}
        self.edges = OrderedDict()
        self.time = self.initial['time']
        self.provider = self.initial['provider']
        self.prov_rating = self.initial['prov_rating']

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

    def add_edge(self, u, v, p):
        """
        Adds edge between u and v having transmission probability p
        """
        u.edges[v] = Edge(p)
        v.edges[u] = Edge(p)
