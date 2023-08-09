import math
import json
import sys
import copy
import multiprocessing as mp

import numpy as np

import sim_lib.attr_lib.util as alu
from sim_lib.attr_lib.formation import calc_edges, calc_edges_global, attribute_network
import ablation_utils as au

############### initializing params ###############

#_N = 10
_N = 77

max_deg = 10

num_iters = 500
max_clique_size = max_deg + 1
ctxt_likelihood = np.round(51/77,3)
sim_iters = 10
#sim_iters = 2
st_count_track = 10
st_count_dev_tol = 0.01

sc_values = np.linspace(0, 1, 9)
ho_values = np.linspace(0, 1, 9)
#sc_values = [0, 1]
#ho_values = [0, 1]
#sc_values = [0.875]
#ho_values = [0.5]

################ run simulation ################

def run_sim(sc_likelihood, ho_likelihood, sim_iters, sub=False):

    # Create type lists
    ctxt_types = [-1, 1]
    ctxt_base_shapes = [0 , 2]
    ctxt_p = [ctxt_likelihood, 1-ctxt_likelihood]
    struct_types = ['em', 'sc']
    attr_types = ['ho', 'he']
    type_itr = [ (ctxt, shape, ct_p, at, st, sc_likelihood, ho_likelihood) \
            for (ctxt, shape, ct_p) in zip(ctxt_types, ctxt_base_shapes, ctxt_p)
                for (at, st) in [(a, s) for a in attr_types for s in struct_types] ]
    type_list = [ au.type_dict(*t_args) for t_args in type_itr ]

    type_counts = [ int(np.floor(_N * tl['likelihood'])) for tl in type_list ]

    remaining_tc = _N - sum(type_counts)
    for i in range(int(remaining_tc)):
        type_counts[i] = type_counts[i] + 1
    assert sum(type_counts) == _N, 'Did that work?'

    tc_dict = { f'type{idx}' : tc for idx, tc in enumerate(type_counts) }
    vtx_types = { f'type{idx}' : tl for idx, tl in enumerate(type_list) }

    params = {
        'context_count' : 2, # Needed for simple utility
        'k' : 1, # Needed for simple attribute utility
        'edge_selection' : alu.seq_edge_sel_silent,
        'seed_type' : 'trivial', # Type of seed network
        'max_clique_size' : max_clique_size,
        'revelation_proposals' : alu.indep_revelation,
        'util_agg' : alu.linear_util_agg, # How to aggregate utility values
        'vtx_types' : vtx_types
    }

    vtx_types_list = np.array([ np.repeat(t, tc) for t, tc in tc_dict.items() ], dtype = 'object')
    vtx_types_list = np.hstack(vtx_types_list)
    #np.random.shuffle(vtx_types_list)
    params['type_assignment'] = { i : vtx_types_list[i] for i in range(_N) }

    # Used for reporting at the end
    final_type_assignments = {}
    for i, ta in params['type_assignment'].items():
        final_type_assignments[i] = copy.deepcopy(vtx_types[ta])
        final_type_assignments[i].pop('struct_util', None)
        final_type_assignments[i].pop('edge_attr_util', None)
        final_type_assignments[i].pop('total_attr_util', None)

    assert math.isclose(sum([ t['likelihood'] for t in params['vtx_types'].values() ]), 1.0)

    # Types created, ready to modify for ablations
    # no locality
    nl_params = copy.deepcopy(params)

    final_networks = {
        'nolocal' : []
    }

    for si in range(sim_iters):

        # Create networks
        G_nl = attribute_network(_N, nl_params)

        st_counts = {
            'nolocal' : []

        }

        ntwk_fin = {
            'nolocal' : False
        }


        for it in range(num_iters):

            # Calculate edges for networks
            if not ntwk_fin['nolocal']:
                calc_edges_global(G_nl)
                nl_st_count = au.count_stable_triads(G_nl)
                if len(st_counts['nolocal']) < st_count_track:
                    st_counts['nolocal'].append(nl_st_count)
                elif np.std(st_counts['nolocal']) <= st_count_dev_tol:
                    ntwk_fin['nolocal'] = True
                    final_networks['nolocal'].append(G_nl.adj_matrix.tolist())
                else:
                    st_counts['nolocal'].pop(0)

            if ntwk_fin['nolocal']:
                break
                
        if not ntwk_fin['nolocal']:
            print('nolocal ablation with param', 'sc', sc_likelihood, 'ho', ho_likelihood, 'did not converge on iteration', si)
            final_networks['nolocal'].append(G_nl.adj_matrix.tolist())

    print('finished ho: ', ho_likelihood, 'sc: ', sc_likelihood)

    return final_networks, final_type_assignments

if __name__ == "__main__":
    pool = mp.Pool(processes=32)

    process_inputs = [ (s, h, sim_iters) for s in sc_values for h in ho_values ]
    sim_res = pool.starmap(run_sim, process_inputs)
    pool.close()
    pool.join()

    for (sc, ho, _), (final_networks, type_assgns) in zip(process_inputs, sim_res):
        ntwk_outname = 'nolocal_data/{n}_{k}_{sc}_{ho}_networks.json'.format(
            n=str(_N), k=str(max_deg), sc=str(sc), ho=str(ho))
        with open(ntwk_outname, 'w+') as out:
            out.write(json.dumps(final_networks))

        types_outname = 'nolocal_data/{n}_{k}_{sc}_{ho}_types.json'.format(
            n=str(_N), k=str(max_deg), sc=str(sc), ho=str(ho))
        with open(types_outname, 'w+') as out:
            out.write(json.dumps(type_assgns))
