from math import ceil
from collections import defaultdict
import json

import numpy as np

from simulation import run_simulation
from graph_create import reduce_providers_simplest, powerlaw_dist_time, watts_strogatz, erdos_renyi, configuration_model
import util

def run_watts_strogatz(num_vertices, k, plaw_resources=False):
    ws_data = defaultdict(dict)
    for r in np.linspace(100, 0, 10, endpoint=False):
        for beta in np.linspace(1, 0, 10):
            ws_social_utils = []
            ws_graph_diams = []
            
            cm_social_utils = []
            cm_graph_diams = []
            for p in np.linspace(1, 0, 100, endpoint=False):
                ws_utils = []
                cm_utils = []
                num_iter = 10
                for i in range(num_iter):
                    ring_lat = watts_strogatz(num_vertices, k, beta, p, p, r)
                    
                    degree_seq = [ vtx.degree for vtx in ring_lat.vertices ]

                    ws_graph_diams.append(util.calc_diameter(ring_lat))

                    reduce_providers_simplest(ring_lat)

                    if plaw_resources:
                        powerlaw_dist_time(ring_lat, 2)

                    sim_g, sim_utils = run_simulation(ring_lat)

                    #Get global social welfare at end of simulation normalized by size
                    ws_utils.append(sum(sim_utils[-1]) / len(ring_lat.vertices))
                    
                    #Run configuration model using WS degree seq
                    config_model = configuration_model(num_vertices, degree_seq, p, r)
                    cm_graph_diams.append(util.calc_diameter(config_model))
                    reduce_providers_simplest(config_model)
                    if plaw_resources:
                        powerlaw_dist_time(config_model, 2)
                        
                    csim_g, csim_utils = run_simulation(config_model)
                    cm_utils.append(sum(csim_utils[-1]) / len(config_model.vertices))
                
                ws_social_utils.append((p, ws_utils))
                cm_social_utils.append((p, cm_utils))

            #Write results
            ws_avg_diam = sum(ws_graph_diams) / len(ws_graph_diams)
            cm_avg_diam = sum(cm_graph_diams) / len(cm_graph_diams)
            ws_data[r][beta] = {'ws' : {'utils' : ws_social_utils, 'avg_diam' : ws_avg_diam}}
            ws_data[r][beta] = {'cm' : {'utils' : cm_social_utils, 'avg_diam' : cm_avg_diam}}
    return ws_data

def run_erdos_renyi(num_vertices, k, plaw_resources=False):
    er_data = defaultdict(dict)
    for r in np.linspace(100, 0, 10, endpoint=False):
        for ep in np.linspace(1, 0, 10):
            er_social_utils = []
            er_graph_diams = []
            
            for p in np.linspace(1, 0, 100, endpoint=False):
                er_utils = []
                num_iter = 10
                for i in range(num_iter):
                    ring_lat = erdos_renyi(num_vertices, ep, p, r)
                    
                    er_graph_diams.append(util.calc_diameter(ring_lat))

                    reduce_providers_simplest(ring_lat)

                    if plaw_resources:
                        powerlaw_dist_time(ring_lat, 2)

                    sim_g, sim_utils = run_simulation(ring_lat)

                    #Get global social welfare at end of simulation normalized by size
                    er_utils.append(sum(sim_utils[-1]) / len(ring_lat.vertices))

                er_social_utils.append((p, er_utils))

            er_avg_diam = sum(er_graph_diams) / len(er_graph_diams)
            er_data[r][ep] = {'er' : {'utils' : er_social_utils, 'avg_diam' : er_avg_diam}}
    return er_data

if __name__ == '__main__':
    num_vertices = 100
    reg = 4
   
    ws_data = run_watts_strogatz(num_vertices, reg)

    with open('ws_data.json', 'w+') as wsd:
        wsd.write(json.dumps(ws_data))

    er_data = run_erdos_renyi(num_vertices, reg)
    
    with open('er_data.json', 'w+') as erd:
        erd.write(json.dumps(er_data))
