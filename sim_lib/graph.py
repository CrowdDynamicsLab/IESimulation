"""
Implementation for a graph
Includes methods for generating graphs
"""

from collections import OrderedDict, defaultdict

import numpy as np
import networkx as nx

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

        # Cache within neighborhood degrees for triangle and social capital 
        self.nbor_degs = {}
        # k-hop reachable
        self.k_hop_reach = { }

    ##########################
    # Attribute observations #
    ##########################
    @property
    def tri_count(self):
        return sum(self.nbor_degs.values()) / 2

    @property
    def disc_nbor_count(self):
        return sum([ 1 for v in self.nbor_degs if self.nbor_degs[v] == 0 ])

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
        self.G_nx = None

        # For data about the graph in model
        self.data = None

        # Potential utility matrix
        self.potential_utils = []
       
        # { Vertex number : set( Vertex numbers ) } 
        self.adj_list = {}

    @property
    def num_people(self):
        return len(self.vertices)
   
    def update_add_nbor_deg(self, u, v):
        # Update neighborhood degrees of u given we add v
        for u_nb in u.nbors:
            if u_nb == v:
                continue
            if self.are_neighbors(u_nb, v):
                u.nbor_deg[u_nb] += 1
                u.nbor_deg[v] += 1
   
    def update_rem_nbor_deg(self, u, v):
        # Update neighborhood degrees of u given we remove v
        for u_nb in u.nbors:
            if self.are_neighbors(u_nb, v):
                u.nbor_deg[u_nb] -= 1

    def add_edge(self, u, v):
        """
        Adds edge between u and v
        """
        if self.are_neighbors(u, v):
            return
        assert (v in u.edges) == (u in v.edges), 'connection must be symmetric'

        u.edges[v] = Edge(self.potential_utils[u.vnum][v.vnum])
        v.edges[u] = Edge(self.potential_utils[v.vnum][u.vnum])
        self.adj_list[u.vnum].append(v.vnum)
        self.adj_list[v.vnum].append(u.vnum)

        self.G_nx.add_edge(u.vnum, v.vnum)

        # Update degree counts
        u.nbor_degs[v] = 0
        v.nbor_degs[u] = 0
        self.update_add_nbor_deg(u, v)
        self.update_add_nbor_deg(v, u)

    def remove_edge(self, u, v, reflexive=True):
        """
        Removes edge between u and v if exists
        If reflexive deletes uv and vu, else just deletes edge uv
        """
        if not self.are_neighbors(u, v):
            return

        #TODO: Why are we checking (0, 0)?
        u.edges[v].data = None
        u.edges.pop(v)
        v.edges[u].data = None
        v.edges.pop(u)
        self.adj_list[u.vnum].remove(v.vnum)
        self.adj_list[v.vnum].remove(u.vnum)

        self.G_nx.remove_edge(u.vnum, v.vnum)

        # Update degree counts
        u.nbor_degs.pop(v)
        v.nbor_degs.pop(u)
        self.update_rem_nbor_deg(u, v)
        self.update_rem_nbor_deg(v, u)

    def k_reachable(self, k):
        # Get the k-reachability for all vertices 
        apsp = nx.all_pairs_shortest_path_length(self.G_nx, k)
        k_reach = { v : set() for v in self.vertices }
        for src, paths in apsp:
            src_vtx = self.vertices[src]
            for tgt, plen in paths.items():
                if plen < 2:
                    continue
                tgt_vtx = self.vertices[tgt]
                k_reach[src_vtx].add(tgt_vtx)
        return k_reach

    @property
    def edge_count(self):
        return sum([ v.degree for v in self.vertices ]) // 2

    def are_neighbors(self, u, v):
        return u.vnum in self.adj_list[v.vnum] and v.vnum in self.adj_list[u.vnum]

    def init_adj_list(self):
        self.G_nx = nx.Graph()
        for v in self.vertices:
            self.G_nx.add_node(v.vnum)
            self.adj_list[v.vnum] = []

        # Returns adjacency list indexed by vnum
        for idx, v in enumerate(self.vertices):
            for u in self.vertices[idx + 1:]:
                if v.is_nbor(u):
                    self.adj_list[v.vnum].add(u.vnum)
                    self.adj_list[u.vnum].add(v.vnum)
                    self.G_nx.add_edge(u.vnum, v.vnum)

        return self.adj_list
    
    @property
    def vertex_type_vec(self):
        return np.array([ v.attr_type for v in self.vertices ])

