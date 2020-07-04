import json
import csv

import networkx as nx
from scipy.sparse import linalg as ssla

import sim_lib.graph_networkx as gnx
import sim_lib.util as util

ntwk_data = {}
with open('ntwk_data/ntwk_data.json', 'r') as ntd:
    ntwk_data = json.loads(ntd.read())

def calc_stats(util_list, ser_g):
    """
    Calculates diameter, APL, clustering coeff, local eff, second eigval (approx cheeger)
    """
    G = util.json_to_graph(ser_g)
    G_nx = gnx.graph_to_nx(G)

    # Diameter
    diam = nx.diameter(G_nx)

    # APL
    apl = nx.average_shortest_path_length(G_nx)

    # Clustering coeff
    clust_coeff = nx.average_clustering(G_nx)

    # Local effiency
    loc_eff = nx.local_efficiency(G_nx)

    # Second eigenvalue
    lap_mat = nx.normalized_laplacian_matrix(G_nx)
    eigvals = ssla.eigs(lap_mat, return_eigenvectors=False)
    eig2 = abs(sorted(eigvals)[1])

    return [diam, apl, clust_coeff, loc_eff, eig2]

rows = []

for r, r_data in ntwk_data.items():
    for p, p_data in r_data.items():
        utils = p_data['utils']
        graphs = p_data['graphs']

        ntwk_types = list(utils.keys())

        for ntwk in ntwk_types:
            for util_list, ser_g in zip(utils[ntwk], graphs[ntwk]):
                stats = calc_stats(util_list, ser_g)

                final_util = sum(util_list[-1])
                rows.append([p, r, ntwk, final_util] + stats)

with open('ntwk_stats/ntwk_stats.csv', 'w+') as nsc:
    csvwriter = csv.writer(nsc)
    csvwriter.writerow(['p', 'r', 'ntwk', 'final_util', 'diam', 'apl', 'clust_coeff', 'loc_eff', 'eig2'])
    for row in rows:
        csvwriter.writerow(row)

