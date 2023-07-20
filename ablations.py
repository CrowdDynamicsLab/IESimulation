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

_N = 10
#_N = 150

max_deg = 10

num_iters = 500
max_clique_size = max_deg + 1
ctxt_likelihood = .5
#sim_iters = 10
sim_iters = 2
st_count_track = 10
st_count_dev_tol = 0.01

#sc_values = np.linspace(0, 1, 9)
#ho_values = np.linspace(0, 1, 9)
sc_values = [0, 1]
ho_values = [1]

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

    vtx_types_list = np.array([ np.repeat(t, tc) for t, tc in tc_dict.items() ])
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

    # no budget
    nb_params = copy.deepcopy(params)
    nb_params['max_clique_size'] = _N

    # no locality
    nl_params = copy.deepcopy(params)

    # no structure
    ns_params = copy.deepcopy(params)
    # params['vtx_types'] is a dict of typeIDX : type dict
    for tidx, tdict in ns_params['vtx_types'].items():
        tdict['struct_util'] = lambda v, G : 0.0

    # no attribute
    na_params = copy.deepcopy(params)
    for tidx, tdict in na_params['vtx_types'].items():
        tdict['total_attr_util'] = lambda u, G : 0.0

    final_networks = {
        'nobudget' : [],
        'nolocal' : [],
        'nostruct' : [],
        'noattr' : []
    } 

    for si in range(sim_iters):

        # Create networks
        G_nb = attribute_network(_N, nb_params)
        G_nl = attribute_network(_N, nl_params)
        G_ns = attribute_network(_N, ns_params)
        G_na = attribute_network(_N, na_params)

        st_counts = {
            'nobudget' : [],
            'nolocal' : [],
            'nostruct' : [],
            'noattr' : []
        }

        ntwk_fin = {
            'nobudget' : False,
            'nolocal' : False,
            'nostruct' : False,
            'noattr' : False
        }

        
        for it in range(num_iters):
            
            # Calculate edges for networks

            if not ntwk_fin['nobudget']:
                calc_edges(G_nb)
                nb_st_count = au.count_stable_triads(G_nb)
                if len(st_counts['nobudget']) < st_count_track:
                    st_counts['nobudget'].append(nb_st_count)
                elif np.std(st_counts['nobudget']) <= st_count_dev_tol:
                    ntwk_fin['nobudget'] = True
                    final_networks['nobudget'].append(G_nb.adj_matrix.tolist())
                else:
                    st_counts['nobudget'].pop(0)

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

            if not ntwk_fin['nostruct']:
                calc_edges(G_ns)
                ns_st_count = au.count_stable_triads(G_ns)
                if len(st_counts['nostruct']) < st_count_track:
                    st_counts['nostruct'].append(ns_st_count)
                elif np.std(st_counts['nostruct']) <= st_count_dev_tol:
                    ntwk_fin['nostruct'] = True
                    final_networks['nostruct'].append(G_ns.adj_matrix.tolist())
                else:
                    st_counts['nostruct'].pop(0)

            if not ntwk_fin['noattr']:
                calc_edges(G_na)
                na_st_count = au.count_stable_triads(G_na)
                if len(st_counts['noattr']) < st_count_track:
                    st_counts['noattr'].append(na_st_count)
                elif np.std(st_counts['noattr']) <= st_count_dev_tol:
                    ntwk_fin['noattr'] = True
                    final_networks['noattr'].append(G_na.adj_matrix.tolist())
                else:
                    st_counts['noattr'].pop(0)

            if ntwk_fin['nobudget'] and ntwk_fin['nolocal'] and \
                    ntwk_fin['nostruct'] and ntwk_fin['noattr']:
                break

    print('ho: ', ho_likelihood, 'sc: ', sc_likelihood)

    return final_networks, final_type_assignments

################ run simulation with various params ################

if __name__ == "__main__":
    for sc in sc_values:
        for ho in ho_values:
            final_networks, type_assgns = run_sim(sc, ho, sim_iters)

            ntwk_outname = 'abl_data/{n}_{k}_{sc}_{ho}_networks.json'.format(
                n=str(_N), k=str(max_deg), sc=str(sc), ho=str(ho))
            with open(ntwk_outname, 'w+') as out:
                out.write(json.dumps(final_networks))

            types_outname = 'abl_data/{n}_{k}_{sc}_{ho}_types.json'.format(
                n=str(_N), k=str(max_deg), sc=str(sc), ho=str(ho))
            with open(types_outname, 'w+') as out:
                out.write(json.dumps(type_assgns))

