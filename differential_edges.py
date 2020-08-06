import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import sim_lib.graph as graph
import sim_lib.graph_create as gc
import sim_lib.graph_networkx as gnx
import sim_lib.util as util

_N = 10
_M = 10

def attribute_graph(n, init_ep):
    vtx_set = []
    attr_dict = {}

    def has_edge(u, v):
        attr1_diff = attr_dict[u][1] - attr_dict[v][1]
        attr2_diff = attr_dict[u][2] - attr_dict[v][2]

        return np.random.random() < -1 * attr1_diff * attr2_diff

    for i in range(n):
        vtx = graph.Vertex(0, 0, {0 : 0}, i)
        attr_dict[vtx] = { 1 : np.random.random(),
                2 : np.random.random() }
        vtx_set.append(vtx)

    G = gc.erdos_renyi(n, init_ep, 1, 0, vtx_set)

    for u in G.vertices:
        for v in u.nbors:
            if not has_edge(u, v):
                G.remove_edge(u, v)

    trivial_comp = 0
    for v in G.vertices:
        if v.degree == 0:
            trivial_comp += 1
    if trivial_comp > 0:
        print('num trivial comp', trivial_comp)

    return G

def attribute_graph_conn(n, init_ep, retries=3):
    for _ in range(retries):
        G = attribute_graph(n)
        if util.is_connected(G):
            return G
    raise ValueError('Could no create connected attribute graph based on given params')

def kleinberg_sw(n, m):
    kg = gc.kleinberg_grid(n, m, 0, 1, 0, 0)
    dist_mat, _ = util.unweighted_apsp(kg)
    for vtx in kg.vertices:
        for nbor in vtx.nbors:
            if np.random.random() < 1 / (dist_mat[vtx.vnum][nbor.vnum] ** 2):
                kg.add_edge(vtx, nbor, 1)
    return kg

for i in range(10):
    G_attr = attribute_graph(_N * _M, 0.5)
    G_attr_nx = gnx.graph_to_nx(G_attr)
    print(sum([ v.degree for v in G_attr.vertices ]) / 2)

    G_kg = kleinberg_sw(_N, _M)
    G_kg_nx = gnx.graph_to_nx(G_kg)
