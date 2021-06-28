"""
Implementation for a graph
Includes methods for generating graphs
"""

import networkx as nx
import sim_lib.attr_lib.util as alu

def graph_to_nx(G):
    """
    Converts a graph from sim_lib.graph to a networkx graph (undirected)
    """

    nx_G = nx.Graph()

    for vtx in G.vertices:
        attr_util = vtx.data['total_attr_util'](vtx, G)
        struct_util = vtx.data['struct_util'](vtx, G)
        cost = alu.calc_cost(vtx, G)
        color = vtx.data['color']
        nx_G.add_node(vtx, attr_util=attr_util, struct_util=struct_util, cost=cost, color=color)
        

    for vtx in G.vertices:
        for nbor in vtx.nbors:
            util = vtx.edges[nbor].util
            nx_G.add_edge(vtx, nbor, capacity=1.0, util=util)

    return nx_G

