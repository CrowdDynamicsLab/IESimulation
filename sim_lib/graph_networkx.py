"""
Implementation for a graph
Includes methods for generating graphs
"""

import networkx as nx

def graph_to_nx(G):
    """
    Converts a graph from sim_lib.graph to a networkx graph (undirected)
    """

    nx_G = nx.Graph()

    for vtx in G.vertices:
        nx_G.add_node(vtx)
        for nbor in vtx.nbors:
            trate = vtx.edges[nbor].trate
            nx_G.add_edge(vtx, nbor, trate=trate)

    return nx_G

