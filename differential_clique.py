import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import sim_lib.graph as graph
import sim_lib.graph_create as gc
import sim_lib.graph_networkx as gnx
import sim_lib.util as util

_N = 10
_M = 10

_HO_THRESH = 0.2 # Threshold for homophily attribute difference (max diff)

def attribute_graph(n):
    vtx_set = []
    attr_dict = {}

    def has_edge(u, v):

        # Meets homophily requirement
        if abs(attr_dict[u][3] - attr_dict[v][3]) > _HO_THRESH:
            return False

        # Differential filter
        if attr_dict[u][1] > attr_dict[v][1] and attr_dict[u][2] < attr_dict[v][2]:
            return True
        elif attr_dict[u][1] < attr_dict[v][1] and attr_dict[u][2] > attr_dict[v][2]:
            return True
        return False

    for i in range(n):
        vtx = graph.Vertex(0, 0, {0 : 0}, i)
        attr_dict[vtx] = { 1 : np.random.random(),
                2 : np.random.random(),
                3 : np.random.random() }
        vtx_set.append(vtx)

    G = graph.Graph()
    G.vertices = vtx_set
    for u in G.vertices:
        for v in G.vertices:
            if u.vnum == v.vnum:
                continue
            if has_edge(u, v):
                G.add_edge(u, v, 1)

    return G

def kleinberg_sw(n, m):
    kg = gc.kleinberg_grid(n, m, 0, 1, 0, 0)
    dist_mat, _ = util.unweighted_apsp(kg)
    for vtx in kg.vertices:
        for nbor in vtx.nbors:
            if np.random.random() < 1 / (dist_mat[vtx.vnum][nbor.vnum] ** 2):
                kg.add_edge(vtx, nbor, 1)
    return kg

for i in range(10):
    G_attr = attribute_graph_conn(_N * _M)
    G_attr_nx = gnx.graph_to_nx(G_attr)

    G_kg = kleinberg_sw(_N, _M)
    G_kg_nx = gnx.graph_to_nx(G_kg)
