import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import sim_lib.graph as graph
import sim_lib.graph_create as gc
import sim_lib.graph_networkx as gnx
import sim_lib.util as util

_N = 5
_M = 5

def attribute_graph(n, num_attr_1, num_attr_2):
    vtx_set = []
    attr_dict = {}

    def has_edge(u, v):
        if attr_dict[u][1] > attr_dict[v][1] and attr_dict[u][2] < attr_dict[v][2]:
            return True
        elif attr_dict[u][1] < attr_dict[v][1] and attr_dict[u][2] > attr_dict[v][2]:
            return True
        return False

    for i in range(n):
        vtx = graph.Vertex(0, 0, {0 : 0}, i)
        attr_dict[vtx] = { 1 : np.random.randint(0, num_attr_1),
                2 : np.random.randint(0, num_attr_2) }
        vtx_set.append(vtx)
    G = graph.Graph()
    G.vertices = vtx_set
    for u in G.vertices:
        for v in G.vertices:
            if u.vnum == v.vnum:
                continue
            if has_edge(u, v):
                G.add_edge(u, v, 1)

    trivial_comp = 0
    for v in G.vertices:
        if v.degree == 0:
            trivial_comp += 1
    if trivial_comp > 0:
        print('num trivial comp', trivial_comp)

    return G

def attribute_graph_conn(n, num_attr_1, num_attr_2, retries=3):
    for _ in range(retries):
        G = attribute_graph(n, num_attr_1, num_attr_2)
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

attr_apl_sum = 0
kg_apl_sum = 0
for i in range(10):
    G_attr = attribute_graph(_N * _M, 10, 10)
    G_attr_nx = gnx.graph_to_nx(G_attr)

    G_kg = kleinberg_sw(_N, _M)
    G_kg_nx = gnx.graph_to_nx(G_kg)

    G_attr_apl = nx.average_shortest_path_length(G_attr_nx)
    G_kg_apl = nx.average_shortest_path_length(G_kg_nx)
    print('attr apl', G_attr_apl)
    print('kg apl', G_kg_apl)

    attr_apl_sum += G_attr_apl
    kg_apl_sum += G_kg_apl

print('avg attr apl', attr_apl_sum / 10)
print('avg kg apl', attr_kg_sum / 10)
