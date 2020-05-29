import json
from math import ceil
import copy
from collections import defaultdict

import numpy as np
import numpy.linalg as nla

from sim_lib.simulation import run_simulation
from sim_lib.graph_create import kleinberg_grid, reduce_providers_simplest, powerlaw_dist_time, add_selfedges
from sim_lib.sim_strategies import EvenAlloc, MWU
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

def gen_data(graph_func, strat_params, plaw_resources=False, simplest=False, debug=False):
    r_vals = [ 2 ** k for k in range(1, 8) ][::-1]

    if debug:
        r_vals = [32]

    # Create final dict
    data = { 'plaw' : plaw_resources }

    for r in r_vals:
        
        num_p = 10
        num_iter = 10
        if debug:
            num_p = 5
            num_iter = 1

        data[str(r)] = {}
        
        for p in np.linspace(1, 0, num_p, endpoint=False):
            utils = []
            eutils = []
            seutils = []
            
            maps = []
            emaps = []
            semaps = []

            graphs = []
            egraphs = []
            segraphs = []

            for i in range(num_iter):
                G = graph_func(p, r)
                G_init = copy.deepcopy(G)

                graph_diam = util.calc_diameter(G)

                if simplest:
                    reduce_providers_simplest(G)
                
                #Get opt to find shortest paths later on
                opt_seeds = util.opt_vertices(G)
                initial_utils = set([ v.utility for v in G.vertices ])
                initial_seeds = { ut : [ v for v in G.vertices if v.utility == ut ] 
                                 for ut in initial_utils }
                
                if plaw_resources:
                    powerlaw_dist_time(G, 2)
                    
                #Make copy of graph to run even alloc strategy on
                G_even = copy.deepcopy(G)

                #Make copy to run self-edge MWU on
                G_se = copy.deepcopy(G)
                add_selfedges(G_se)

                #Initialize strategy
                sim_strat = MWU(**strat_params)
                sim_strat.initialize_model(G)

                sim_g, sim_utils, mwu_ut_map = run_simulation(G, sim_strat, True)
                
                #Initialize even alloc strategy
                even_strat = EvenAlloc()
                even_strat.initialize_model(G_even)
                
                sim_g_even, sim_utils_even, even_ut_map = run_simulation(G_even, even_strat, True)

                #Initialize MWU for self edge
                mwu_self = MWU(**strat_params)
                mwu_self.initialize_model(G_se)

                sim_g_se, sim_utils_se, mwu_se_ut_map = run_simulation(G_se, mwu_self, True)

                #Get global social welfare at end of simulation normalized by size
                utils.append(sim_utils)
                eutils.append(sim_utils_even)
                seutils.append(sim_utils_se)

                maps.append(stringify_map(mwu_ut_map))
                emaps.append(stringify_map(even_ut_map))
                semaps.append(stringify_map(mwu_se_ut_map))

                graphs.append((util.serialize_graph(G_init),
                    util.serialize_graph(sim_g)))
                egraphs.append((util.serialize_graph(G_init),
                    util.serialize_graph(sim_g_even)))
                segraphs.append((util.serialize_graph(G_init),
                    util.serialize_graph(sim_g_se)))

            p_str = str(round(p, 3))
            data[str(r)][p_str] = {}
            data[str(r)][p_str]['utils'] = { 'mwu' : utils, 'even' : eutils, 'se_mwu' : seutils }
            data[str(r)][p_str]['maps']= { 'mwu' : maps, 'even' : emaps, 'se_mwu' : semaps }
            data[str(r)][p_str]['graphs'] = { 'mwu' : graphs, 'even' : egraphs, 'se_mwu' : segraphs }
    return data

nonplaw_data = gen_data(create_kg, {'lrate' : lrate}, simplest=False)
plaw_data = gen_data(create_kg, {'lrate' : lrate}, simplest=False, plaw_resources=True)

with open ('data/kgrid_data.json', 'w+') as kd:
    kd.write(json.dumps(nonplaw_data))

with open ('data/kgrid_plaw_data.json', 'w+') as kd:
    kd.write(json.dumps(plaw_data))

