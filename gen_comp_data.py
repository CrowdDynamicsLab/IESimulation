from collections import defaultdict, OrderedDict
import math
import json

import networkx as nx
import numpy as np

import sim_lib.util as util
import sim_lib.attr_lib.util as alu
from sim_lib.attr_lib.formation import *

# Overall parameters

save_to = 'data/comp_comparisons.json'

_N = 36
iter_count = 16
num_runs = 10

params = {
    'context_count' : 2, # Needed for simple utility
    'k' : 1, # Needed for simple attribute utility
    'edge_selection' : alu.seq_projection_edge_edit,
    'seed_type' : 'grid', # Type of seed network
    'max_clique_size' : 5,
    'revelation_proposals' : alu.resistance_distance_revelation,
    'util_agg' : lambda a, s, c: a + s, # How to aggregate utility values
    'vtx_types' :
        {
            'type1' : { 'likelihood' : 0.5,
                      'struct_util' : None,
                      'init_attrs' : alu.init_cont_heterophily, # context 1
                      'edge_attr_util' : None,
                      'total_attr_util' : None,
                      'color' : 'blue' },
            'type0' : { 'likelihood' : 0.5,
                      'struct_util' : None,
                      'init_attrs' : alu.init_cont_homophily, # context 0
                      'edge_attr_util' : None,
                      'total_attr_util' : None,
                      'color' : 'red' }
        }
}

type1_count = math.floor(_N * params['vtx_types']['type1']['likelihood'])
vtx_types_list = ['type1'] * type1_count + ['type0'] * (_N - type1_count)
np.random.shuffle(vtx_types_list)
params['type_assignment'] = { i : vtx_types_list[i] for i in range(_N) }

assert sum([ t['likelihood'] for t in params['vtx_types'].values() ]) == 1.0

# Data collection functions 

def get_component_sizes(G_comps):
    comp_sizes = {}
    for gc in G_comps:
        gc_size = gc.number_of_nodes()
        if gc_size in comp_sizes:
            comp_sizes[gc_size] += 1
        else:
            comp_sizes[gc_size] = 1
    return comp_sizes

def get_clique_sizes(G_comps):
    clique_sizes = {}
    for gc in G_comps:
        gc_size = gc.number_of_nodes()
        is_clique = gc.number_of_edges() == (gc_size * (gc_size - 1) / 2)
        if not is_clique:
            continue
        if gc_size in clique_sizes:
            clique_sizes[gc_size] += 1
        else:
            clique_sizes[gc_size] = 1
    return clique_sizes

# Run simulation
# Parameters
similarity_funcs = list(alu.gen_similarity_funcs())
attr_func_named = list(zip(similarity_funcs, ['homophily', 'heterophily']))
theta_values = [0.25, 0.5, 0.75, 1.0][::-1]
struct_funcs = [alu.average_neighborhood_overlap]
struct_func_named = list(zip(struct_funcs, ['embedded']))
seed_types = ['trivial']

# Prepare json data
sim_results = {}

for theta in theta_values:
    sim_results[theta] = {}
    for idx, (sim_func, af_name) in enumerate(attr_func_named):
        sim_results[theta][af_name] = {}
        attr_func = alu.gen_schelling_seg_funcs(theta, 'satisfice')[idx]
        for struct_func, sf_name in struct_func_named:
            sim_results[theta][af_name][sf_name] = {}
            for vtype in ['type0', 'type1']:
                params['vtx_types'][vtype]['struct_util'] = struct_func
                params['vtx_types'][vtype]['edge_attr_util'] = sim_func
                params['vtx_types'][vtype]['total_attr_util'] = attr_func
                    
            for seed in seed_types:
                params['seed_type'] = seed

                sim_results[theta][af_name][sf_name][seed] = {}
                cur_setting_dict = sim_results[theta][af_name][sf_name][seed]
                cur_setting_dict['component'] = []
                cur_setting_dict['clique'] = []
                run_comp_sizes = []
                run_clique_sizes = []
                for _ in range(num_runs):
                    comp_counts = []
                    clique_counts = []
                    G = attribute_network(_N, params)
                    for it in range(iter_count):
                        
                        # Iterate simulation
                        iteration_metadata = calc_edges(G)
                        
                        # Record values
                        G_nx = alu.graph_to_nx(G)
                        G_nx_comp_nodes = nx.algorithms.components.connected_components(G_nx)
                        G_nx_comps = [ G_nx.subgraph(c).copy() for c in G_nx_comp_nodes ]
                        
                        comp_counts.append(get_component_sizes(G_nx_comps))
                        clique_counts.append(get_clique_sizes(G_nx_comps))
                    run_comp_sizes.append(comp_counts)
                    run_clique_sizes.append(clique_counts)
                        
                # Get averages across different runs per iteration
                for it in range(iter_count):
                    total_comp_counts = defaultdict(lambda : 0)
                    total_clique_counts = defaultdict(lambda : 0)
                    for r in range(num_runs):
                        for size, count in run_comp_sizes[r][it].items():
                            total_comp_counts[size] += count
                        for size, count in run_clique_sizes[r][it].items():
                            total_clique_counts[size] += count
                    comp_avgs = { size : count / num_runs for size, count in total_comp_counts.items() }
                    clique_avgs = { size : count / num_runs for size, count in total_clique_counts.items() }
                    cur_setting_dict['component'].append(comp_avgs)
                    cur_setting_dict['clique'].append(clique_avgs)

with open(save_to, 'w+') as sf:
    sf.write(json.dumps(sim_results))
 
