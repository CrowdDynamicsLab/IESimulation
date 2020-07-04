import json
from math import ceil
import copy
from collections import defaultdict

import numpy as np
import numpy.linalg as nla

from sim_lib.simulation import run_simulation
from sim_lib.graph_create import kleinberg_grid, reduce_providers_simplest, powerlaw_dist_time, powerlaw_dist_util, add_selfedges
from sim_lib.sim_strategies import Uniform, RoundRobin, MWU
import sim_lib.util as util

_DEBUG = True

# Constants
gl = 10 # grid length
regularity = 4

# MWU parameters
lrate = 0.5

def create_kg(p, r):
    return kleinberg_grid(gl, gl, r, p)

def stringify_map(ut_map):
    str_map = {}
    for vtx, vum in ut_map.items():
        vum['from'] = [ v.vnum for v in vum['from'] ]
        str_map[vtx.vnum] = vum
    return str_map

def gen_data(graph_func, strat_params, simplest=False, debug=False):
    r_vals = [ 2 ** k for k in range(1, 8) ][::-1]

    if debug:
        r_vals = [32]

    strategies = ['rr', 'unif', 'se_mwu', 'mwu']

    # Create final dict

    full_data = { 'no_plaw' : {}, 'plaw_r' : {}, 'plaw_ut' : {}, 'plaw_both' : {} }
    plaw_modes = list(full_data.keys())

    data = { pm : {} for pm in plaw_modes }

    for r in r_vals:

        num_p = 10
        num_iter = 10
        if debug:
            num_p = 5
            num_iter = 1

        for pm in plaw_modes:
            data[pm][str(r)] = {}
        
        for p in np.linspace(1, 0, num_p, endpoint=False):
            utils = { pm : { st : [] for st in strategies } for pm in plaw_modes }
            maps = { pm : { st : [] for st in strategies } for pm in plaw_modes }
            graphs = { pm : { st : [] for st in strategies } for pm in plaw_modes }

            init_graphs = { pm : [] for pm in plaw_modes }
            
            for i in range(num_iter):
                G = graph_func(p, r)

                graph_diam = util.calc_diameter(G)

                if simplest:
                    reduce_providers_simplest(G)

                plaw_settings = [('no_plaw', False, False), ('plaw_r', True, False),
                                 ('plaw_ut', False, True), ('plaw_both', True, True)]
               
                for pm, plaw_resources, plaw_utils in plaw_settings:

                    G_init = copy.deepcopy(G)

                    #Get opt to find shortest paths later on
                    opt_seeds = util.opt_vertices(G_init)
                    initial_utils = set([ v.utility for v in G_init.vertices ])
                    initial_seeds = { ut : [ v for v in G_init.vertices if v.utility == ut ] 
                                     for ut in initial_utils }
                    
                    if plaw_resources:
                        powerlaw_dist_time(G_init, 2)

                    if plaw_utils:
                        powerlaw_dist_util(G_init, 2)
                    
                    #Make copy of graph to run MWU on
                    G_mwu = copy.deepcopy(G_init)

                    #Make copy of graph to run round robin
                    G_rr = copy.deepcopy(G_init)

                    #Make copy to run self-edge MWU on
                    G_se = copy.deepcopy(G_init)
                    add_selfedges(G_se)

                    #Make copy of graph for time weighted
                    G_unif = copy.deepcopy(G_init)

                    #Initialize MWU
                    mwu_strat = MWU(**strat_params)
                    mwu_strat.initialize_model(G_mwu)

                    sim_g_mwu, sim_utils_mwu, mwu_ut_map = run_simulation(G_mwu, mwu_strat, True)

                    #Initialize MWU for self edge
                    mwu_self = MWU(**strat_params)
                    mwu_self.initialize_model(G_se)

                    sim_g_se, sim_utils_se, mwu_se_ut_map = run_simulation(G_se, mwu_self, True)
                    
                    #Initialize round robin strategy
                    rr_strat = RoundRobin()
                    rr_strat.initialize_model(G_rr)
                    
                    sim_g_rr, sim_utils_rr, rr_ut_map = run_simulation(G_rr, rr_strat, True)

                    #Initialize time weighted strategy
                    unif_strat = Uniform()
                    unif_strat.initialize_model(G_unif)
                    sim_g_unif, sim_utils_unif, unif_ut_map = run_simulation(G_unif, unif_strat, True)

                    #Get global social welfare at end of simulation normalized by size
                    utils[pm]['mwu'].append(sim_utils_mwu)
                    utils[pm]['se_mwu'].append(sim_utils_se)
                    utils[pm]['rr'].append(sim_utils_rr)
                    utils[pm]['unif'].append(sim_utils_unif)

                    maps[pm]['mwu'].append(stringify_map(mwu_ut_map))
                    maps[pm]['se_mwu'].append(stringify_map(mwu_se_ut_map))
                    maps[pm]['rr'].append(stringify_map(rr_ut_map))
                    maps[pm]['unif'].append(stringify_map(unif_ut_map))

                    graphs[pm]['mwu'].append(util.serialize_graph(sim_g_mwu))
                    graphs[pm]['se_mwu'].append(util.serialize_graph(sim_g_se))
                    graphs[pm]['rr'].append(util.serialize_graph(sim_g_rr))
                    graphs[pm]['unif'].append(util.serialize_graph(sim_g_unif))

                    init_graphs[pm].append(util.serialize_graph(G_init))

            for pm in plaw_modes:
                p_str = str(round(p, 3))
                data[pm][str(r)][p_str] = {}
                data[pm][str(r)][p_str]['utils'] = utils[pm]
                data[pm][str(r)][p_str]['maps']= maps[pm]
                data[pm][str(r)][p_str]['graphs'] = graphs[pm]
                data[pm][str(r)][p_str]['init_graphs'] = init_graphs[pm]
    return data

full_data = gen_data(create_kg, {'lrate' : lrate}, simplest=False)

with open ('kgrid_data/kgrid_data.json', 'w+') as kd:
    kd.write(json.dumps(full_data))

