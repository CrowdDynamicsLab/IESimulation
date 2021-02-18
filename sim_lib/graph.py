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

        # For params relating to the simulation
        self.sim_params = {}

    @property
    def degree(self):
        return len(self.edges)

    @property
    def nbors(self):
        return list(self.edges.keys())

    @property
    def total_edge_util(self):
        return sum([ e.util for e in self.edges.values() ])

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
            edge_util = self.potential_utils[u.vnum][v.vnum]
            u.edges[v] = Edge(edge_util)
            v.edges[u] = Edge(edge_util)

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
