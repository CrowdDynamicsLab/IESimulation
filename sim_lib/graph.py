"""
Implementation for a graph
Includes methods for generating graphs
"""

from collections import OrderedDict, defaultdict
from functools import cached_property

import numpy as np

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
        self.attr_type = None
        
        # For params relating to the simulation
        self.sim_params = {}

    ##########################
    # Attribute observations #
    ##########################

    @property
    def degree(self):
        return len(self.edges)

    @property
    def nbors(self):
        return list(self.edges.keys())
    
    @property
    def nbor_num_list(self):
        return [ v.vnum for v in self.nbors ]

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

    def utility_values(self, G):

        #Gets attribute, structural utilities
        attr = self.data['total_attr_util'](self, G)
        struct = self.data['struct_util'](self, G)
        return attr, struct

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
        
        self.adj_matrix = None

    @property
    def num_people(self):
        return len(self.vertices)

    def _clear_cache(self):
        self.__dict__.pop('adj_matrix', None)
    
    def add_edge(self, u, v):
        """
        Adds edge between u and v
        """
        assert (v in u.edges) == (u in v.edges), 'connection must be symmetric'
        if not self.are_neighbors(u, v):
            u.edges[v] = Edge(self.potential_utils[u.vnum][v.vnum])
            v.edges[u] = Edge(self.potential_utils[v.vnum][u.vnum])
            self.adj_matrix[u.vnum][v.vnum] = 1
            self.adj_matrix[v.vnum][u.vnum] = 1 

    def remove_edge(self, u, v, reflexive=True):
        """
        Removes edge between u and v if exists
        If reflexive deletes uv and vu, else just deletes edge uv
        """
        if v in u.edges:
            u.edges[v].data = None
            u.edges.pop(v)
            self.adj_matrix[u.vnum][v.vnum] = 0
            self.adj_matrix[v.vnum][u.vnum] = 0
        if reflexive and u in v.edges:
            v.edges[u].data = None
            v.edges.pop(u)
            self.adj_matrix[u.vnum][v.vnum] = 0
            self.adj_matrix[v.vnum][u.vnum] = 0

    @property
    def edge_count(self):
        return sum([ v.degree for v in self.vertices ]) // 2

    def are_neighbors(self, u, v):
        return self.adj_matrix[v.vnum][u.vnum] == 1

    def init_adj_matrix(self):

        # Returns numpy matrix indexed by vnum
        adj_mat = np.zeros((self.num_people, self.num_people))
        for idx, v in enumerate(self.vertices):
            for u in self.vertices[idx + 1:]:
                if v.is_nbor(u):
                    adj_mat[v.vnum][u.vnum] = 1
                    adj_mat[u.vnum][v.vnum] = 1
        self.adj_matrix = adj_mat
        return adj_mat
    
    @property
    def vertex_type_vec(self):
        return np.array([ v.attr_type for v in self.vertices ])

    def nborhood_adj_mat(self, v):
        # https://stackoverflow.com/questions/17740081/given-an-nxn-adjacency-matrix-how-can-one-compute-the-number-of-triangles-in-th
        adj_mat = self.adj_matrix
        nbor_submat = adj_mat[np.ix_(v.nbor_num_list, v.nbor_num_list)]
        return nbor_submat
