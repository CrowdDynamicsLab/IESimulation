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
