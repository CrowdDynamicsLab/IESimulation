from collections import defaultdict, OrderedDict
import math

import networkx as nx
import numpy as np
import pandas as pd

import sim_lib.util as util
import sim_lib.attr_lib.util as alu
from sim_lib.attr_lib.formation import *

# Overall parameters

save_to = 'data/sc_sub_comparisons.csv'

_N = 36
iter_count = 16
num_runs = 10

_N = 32

# Simul + some vis parameters
num_iters = 32
add_to_iter = 0
show_last = 0
show_every = 1
log_last = 0
log_every = 0

similarity_homophily, similarity_heterophily = alu.gen_similarity_funcs()
schelling_homophily, schelling_heterophily = alu.gen_schelling_seg_funcs(0.5, 'sat_count')

attr_edge_func = similarity_homophily
attr_total_func = schelling_homophily
struct_func = alu.ball2_size

# Create types
def type_dict(context, color):
    
    #Base color is a rgb list
    base_dict = { 'likelihood' : None,
              'struct_util' : struct_func,
              'init_attrs' : context,
              'edge_attr_util' : attr_edge_func,
              'total_attr_util' : attr_total_func,
              'optimistic' : None,
              'color' : None }
    opt_case = base_dict.copy()
    opt_case['optimistic'] = True
    opt_case['color'] = 'rgb({rgb})'.format(rgb=', '.join([str(c // 3) for c in color]))
    pes_case = base_dict.copy()
    pes_case['optimistic'] = False
    pes_case['color'] = 'rgb({rgb})'.format(rgb=', '.join([ str(c) for c in color ]))
    return [opt_case, pes_case]

ctxt_types = [alu.init_cont_homophily, alu.init_cont_heterophily]
ctxt_base_colors = [[43, 98, 166], [161, 39, 45]]
type_lists = [ type_dict(ctxt, ctxt_color) for ctxt, ctxt_color \
              in zip(ctxt_types, ctxt_base_colors) ]
type_lists_flat = [ td for ctd in type_lists for td in ctd ]
vtx_types = { f'type{idx}' : tl for idx, tl in enumerate(type_lists_flat) }

params = {
    'context_count' : 2, # Needed for simple utility
    'k' : 1, # Needed for simple attribute utility
    'edge_selection' : alu.seq_projection_edge_edit,
    'seed_type' : 'trivial', # Type of seed network
    'max_clique_size' : 5,
    'revelation_proposals' : alu.indep_revelation,
    'util_agg' : alu.attr_first_agg, # How to aggregate utility values
    'vtx_types' : vtx_types
}

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

def get_sub_counts(md):
    sub_ind = lambda a : 1 if a == 'substitution' else 0
    return [ sub_ind(md[v]['action']) for v in G.vertices ]

# Run simulation
# Parameters
attr_homophily, attr_heterophily = alu.gen_similarity_funcs()
theta_values = [0.0, 0.32, 0.65, 1.0][::-1]
optimism_values = [0.0, 0.25, 0.5, 0.75, 1.0][::-1]
struct_func = alu.average_neighborhood_overlap
seed_type = 'trivial'
agg_funcs = [ alu.linear_util_agg, alu.attr_first_agg, alu.struct_first_agg ]
agg_func_named = list(zip(agg_funcs, ['linear', 'attr_first', 'struct_first']))

# Set up df
sim_properties = ['theta', 'p_optim', 'agg_func']
sim_metrics = ['struct_util', 'attr_util', 'cost', 'degree',
    'struct_delta', 'attr_delta', 'cost_delta',
    'num_proposals', 'num_budget_resolve', 'num_subs'
]
simulation_df = pd.DataFrame(columns=sim_properties + sim_metrics)
sim_graph_funcs = [get_struct_utils, get_attribute_utils, get_costs, get_degrees]
sim_metadata_funcs = [get_struct_changes, get_attr_changes, get_cost_changes,
    get_proposal_counts, get_budget_resolution_counts, get_sub_counts
]
sim_metric_func_tuples = list(zip(sim_metrics, sim_graph_funcs + sim_metadata_funcs))

for theta in theta_values:
    for idx, (agg_func, af_name) in enumerate(agg_func_named):
        attr_func = alu.gen_schelling_seg_funcs(theta, 'satisfice')[0]
        for popt in optimism_values:

            # Set vtx types by optimism proportion
            for tl in type_lists_flat:
                if tl['optimistic']:
                    tl['likelihood'] = popt * (2 / len(type_lists_flat))
                else:
                    tl['likelihood'] = (1 - popt) * (2 / len(type_lists_flat))

            assert sum([ t['likelihood'] for t in params['vtx_types'].values() ]) == 1.0

            type_counts = [ int(np.floor(_N * tl['likelihood'])) for tl in type_lists_flat ]
            remaining_tc = _N - sum(type_counts)
            for i in range(remaining_tc):
                type_counts[i] = type_counts[i] + 1
            assert sum(type_counts) == _N, 'Did that work?'
            tc_dict = { f'type{idx}' : tc for idx, tc in enumerate(type_counts) }

            vtx_types_list = np.array([ np.repeat(t, tc) for t, tc in tc_dict.items() ],
                dtype=object)
            vtx_types_list = np.hstack(vtx_types_list)
            np.random.shuffle(vtx_types_list)
            params['type_assignment'] = { i : vtx_types_list[i] for i in range(_N) }

            # Set aggregate utility function
            params['util_agg'] = agg_func
                
            # Run simulations
            params['seed_type'] = 'trivial'
            
            sim_run_values = {
                'theta' : theta,
                'p_optim' : popt,
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

