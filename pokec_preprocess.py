import os
import csv
import random
import json

import numpy as np
import pandas as pd
import networkx as nx

# Preprocess params

# Sampling
n_graphs = 10
sample_size = 150

# Output
out_dir = 'datasets/pokec/sampled/'

# Load data

dataset_dir = 'datasets/pokec/'
edge_file = dataset_dir + 'soc-pokec-relationships.txt'
attr_file = dataset_dir + 'soc-pokec-profiles.txt'

data = {
    'edgelist' : [],
    'users' : {}
}

with open(attr_file, 'r') as af:
    af_reader = csv.reader(af, delimiter='\t')
    for af_row in af_reader:
        vtx_id = af_row[0]
        vtx_gender = af_row[3]
        data['users'][vtx_id] = {}
        data['users'][vtx_id]['gender'] = vtx_gender
        
with open(edge_file, 'r') as ef:
    ef_reader = csv.reader(ef, delimiter='\t')
    for ef_row in ef_reader:
        data['edgelist'].append((ef_row[0], ef_row[1]))

G_nx = nx.from_edgelist(data['edgelist'], create_using=nx.DiGraph)

# Remove non-mutual edges
G_nx = G_nx.to_undirected(reciprocal=True)

# Get largest component
G_comp_vtx = sorted(nx.connected_components(G_nx), key=len, reverse=True)[0]
G_comp = G_nx.subgraph(G_comp_vtx).copy()

def XSN(G_s):
    seed = random.choice(list(G_s.nodes()))
    S = set()
    S.add(seed)
    
    s_nbors = set(G_s.neighbors(seed))
    s_union = s_nbors.union(S)
    while True:
        if len(S) > sample_size:
            break
        max_v = None
        max_v_nbors = None
        max_exp = -1
        for v in s_nbors:
            v_nbors = set(G_s.neighbors(v))
            expansion = len(v_nbors - s_union)
            if expansion > max_exp:
                max_v = v
                max_v_nbors = v_nbors
                max_exp = expansion
        S.add(max_v)
        s_nbors = s_nbors.union(max_v_nbors)
        s_union = s_nbors.union(S)

    return S

for i in range(n_graphs):

    # Expansion sample from largest connected component
    print('Sampling via expansion snowball')
    S = XSN(G_comp)
    print('Sampled')
    S_graph = G_nx.subgraph(S).copy()
    print('Getting adjacency matrix and attributes')
    S_list = list(S)
    S_adjmat = nx.to_numpy_matrix(S_graph, nodelist=S_list).tolist()
    S_attrs = { }
    for idx, s in enumerate(S_list):
        S_attrs[s] = { }
        S_attrs[s]['gender'] = data['users'][s]['gender']
        S_attrs[s]['idx'] = idx
    S_out = {
        'ntwk' : S_adjmat,
        'attrs' : S_attrs
    }
    with open('{od}pokec_{i}.json'.format(od=out_dir, i=i), 'w+') as of:
        of.write(json.dumps(S_out))

    
