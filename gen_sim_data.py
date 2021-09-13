from collections import defaultdict, OrderedDict
import math

import networkx as nx
import numpy as np
import pandas as pd

import sim_lib.util as util
import sim_lib.attr_lib.util as alu
from sim_lib.attr_lib.formation import *

# Overall parameters

save_to = 'data/satiation_removal_comparisons.csv'

_N = 36
iter_count = 16
num_runs = 10

params = {
    'context_count' : 2, # Needed for simple utility
    'k' : 1, # Needed for simple attribute utility
    'edge_selection' : alu.seq_projection_edge_edit,
    'seed_type' : 'grid', # Type of seed network
    'max_clique_size' : 5,
    'revelation_proposals' : alu.indep_revelation,
    'util_agg' : None, # How to aggregate utility values
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

def get_struct_utils(G):
    return [ u.data['struct_util'](u, G) for u in G.vertices ]

def get_attribute_utils(G):
    return [ u.data['total_attr_util'](u, G) for u in G.vertices ]

def get_degrees(G):
    return [ u.degree for u in G.vertices ]

def get_costs(G):
    return [ alu.calc_cost(u, G) for u in G.vertices ]

def get_attr_changes(md):
    return [ md[v]['attr_delta'] for v in G.vertices ]

def get_struct_changes(md):
    return [ md[v]['struct_delta'] for v in G.vertices ]

def get_cost_changes(md):
    return [ md[v]['cost_delta'] for v in G.vertices ]

def get_proposal_counts(md):
    return [ md[v]['num_proposals'] for v in G.vertices ]

def get_budget_resolution_counts(md):
    br_ind = lambda a : 1 if a == 'budget_resolve' else 0
    return [ br_ind(md[v]['action']) for v in G.vertices ]

# Run simulation
# Parameters
attr_homophily, attr_heterophily = alu.gen_similarity_funcs()
theta_values = [0.0, 0.25, 0.5, 0.75, 1.0][::-1]
struct_func = alu.average_neighborhood_overlap
seed_type = 'trivial'
agg_funcs = [ alu.linear_util_agg, alu.attr_first_agg, alu.struct_first_agg ]
agg_func_named = list(zip(agg_funcs, ['linear', 'attr_first', 'struct_first']))

# Set up df
sim_properties = ['theta', 'agg_func']
sim_metrics = ['struct_util', 'attr_util', 'cost', 'degree',
    'struct_delta', 'attr_delta', 'cost_delta',
    'num_proposals', 'num_budget_resolve'
]
simulation_df = pd.DataFrame(columns=sim_properties + sim_metrics)
sim_graph_funcs = [get_struct_utils, get_attribute_utils, get_costs, get_degrees]
sim_metadata_funcs = [get_struct_changes, get_attr_changes, get_cost_changes,
    get_proposal_counts, get_budget_resolution_counts
]
sim_metric_func_tuples = list(zip(sim_metrics, sim_graph_funcs + sim_metadata_funcs))

for theta in theta_values:
    for idx, (agg_func, af_name) in enumerate(agg_func_named):
        attr_func = alu.gen_schelling_seg_funcs(theta, 'satisfice')[0]
        for vtype in ['type0', 'type1']:
            params['vtx_types'][vtype]['struct_util'] = struct_func
            params['vtx_types'][vtype]['edge_attr_util'] = attr_homophily
            params['vtx_types'][vtype]['total_attr_util'] = attr_func
        params['util_agg'] = agg_func
            
        # Run simulations
        params['seed_type'] = 'trivial'
        
        sim_run_values = {
            'seed' : seed_type,
            'theta' : theta,
            'struct_func' : 'anl',
            'attr_func' : 'homophily',
            'agg_func' : af_name
        }

        sim_run_metrics = { mt : defaultdict(list) for mt in sim_metrics }
        for _ in range(num_runs):
            G = attribute_network(_N, params)
            for it in range(iter_count):
                
                # Iterate simulation
                iteration_metadata = calc_edges(G)
                
                # Record values
                for mname, mfunc in sim_metric_func_tuples:
                    if mfunc in sim_graph_funcs:
                        sim_run_metrics[mname][it].append(mfunc(G))
                    elif mfunc in sim_metadata_funcs:
                        sim_run_metrics[mname][it].append(mfunc(iteration_metadata))
                
        # Get averages across different runs per iteration
        def avg_dict(d):
            it_avg = lambda d, it : np.mean(np.array(d[it]), axis=0)
            return [ np.mean(it_avg(d, it)) for it in range(iter_count) ]
        for mt in sim_metrics:
            sim_run_values[mt] = avg_dict(sim_run_metrics[mt])
        simulation_df = simulation_df.append(sim_run_values, ignore_index=True)
 
simulation_df.to_csv(save_to)

