import json
from math import ceil
import copy
from collections import defaultdict

import numpy as np
import numpy.linalg as nla

from sim_lib.simulation import run_simulation
from sim_lib.graph_create import create_vtx_set, ring_lattice, watts_strogatz, erdos_renyi, configuration_model, kleinberg_grid
from sim_lib.sim_strategies import Uniform
import sim_lib.util as util

_DEBUG = True

# Constants
n = 100
expec_deg = 5

def stringify_map(ut_map):
    str_map = {}
    for vtx, vum in ut_map.items():
        vum['from'] = [ v.vnum for v in vum['from'] ]
        str_map[vtx.vnum] = vum
    return str_map

def gen_data(debug=False):
    r_vals = [ 2 ** k for k in range(1, 8) ][::-1]

    if debug:
        r_vals = [32]

    networks = {'rl' : ring_lattice, 'ws' : watts_strogatz, 'er' : erdos_renyi,
            'cm' : configuration_model, 'kg' : kleinberg_grid}

    data = {}

    for r in r_vals:

        num_p = 10
        num_iter = 10
        if debug:
            num_p = 5
            num_iter = 1

        data[str(r)] = {}

        for p in np.linspace(1, 0, num_p, endpoint=False):
            utils = { nt : [] for nt in networks.keys() }
            maps = { nt : [] for nt in networks.keys() }
            graphs = { nt : [] for nt in networks.keys() }
            init_graphs = { nt : [] for nt in networks.keys() }
            
            for i in range(num_iter):
                vertices = create_vtx_set(n, r)

                for ntwk, ntwk_func in networks.items():
                    vtx_set = copy.deepcopy(vertices)
                    
                    # Create graph based on current network type
                    if ntwk == 'rl':
                        G = ntwk_func(expec_deg, n, p, r, vtx_set)
                    elif ntwk == 'ws':
                        beta = 0.1 # see ws paper - "spreadability" plateaus around here
                        G = ntwk_func(n, expec_deg, beta, p, r, vtx_set)
                    elif ntwk == 'er':
                        ep = expec_deg / (n - 1)
                        assert ep > np.log(n) / n, 'Edge probability must be greater than ln(n) / n for connectedness'
                        G = erdos_renyi(n, ep, p, r, vtx_set)
                    elif ntwk == 'cm' :
                        deg_seq = [ expec_deg ] * n
                        G = ntwk_func(n, deg_seq, p, r, vtx_set)
                    elif ntwk == 'kg':
                        m = int(n ** 0.5)
                        assert m * m == n, 'For now n must have an integer root'
                        G = kleinberg_grid(m, m, r, p, vtx_set=vtx_set)

                    init_graphs[ntwk].append(util.serialize_graph(G))

                    #Initialize time weighted strategy
                    unif_strat = Uniform()
                    unif_strat.initialize_model(G)
                    sim_g, sim_utils, ut_map = run_simulation(G, unif_strat, True)

                    #Get global social welfare at end of simulation normalized by size
                    utils[ntwk].append(sim_utils)
                    maps[ntwk].append(stringify_map(ut_map))
                    graphs[ntwk].append(util.serialize_graph(sim_g))

            p_str = str(round(p, 3))
            data[str(r)][p_str] = {}
            data[str(r)][p_str]['utils'] = utils
            data[str(r)][p_str]['maps']= maps
            data[str(r)][p_str]['graphs'] = graphs
            data[str(r)][p_str]['init_graphs'] = init_graphs
    return data

full_data = gen_data(debug=False)

with open ('ntwk_data/ntwk_data.json', 'w+') as ndata:
    ndata.write(json.dumps(full_data))

