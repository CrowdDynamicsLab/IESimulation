from math import ceil
from collections import defaultdict
import json
import uuid

import numpy as np

from simulation import run_simulation
from graph_create import reduce_providers_simplest, powerlaw_dist_time, watts_strogatz, erdos_renyi, configuration_model
import util

import argparse

def run_watts_strogatz(num_vertices, k, r_b_pairs, strat, strat_params, plaw_resources=False, save=False):
    ws_data = defaultdict(dict)
    for r, beta in r_b_pairs:
        ws_social_utils = []
        ws_graph_diams = []
        
        cm_social_utils = []
        cm_graph_diams = []
        for p in np.linspace(1, 0, 10, endpoint=False):
            ws_utils = []
            cm_utils = []
            num_iter = 100
            for i in range(num_iter):
                G = watts_strogatz(num_vertices, k, beta, p, p, r)
                
                degree_seq = [ vtx.degree for vtx in G.vertices ]

                ws_diam = util.calc_diameter(G)
                ws_graph_diams.append(ws_diam)

                reduce_providers_simplest(G)

                if plaw_resources:
                    powerlaw_dist_time(G, 2)

                sim_strat = strat(**strat_params)
                sim_strat.initialize_model(G)
                sim_g, sim_utils = run_simulation(G, sim_strat)

                #Get global social welfare at end of simulation normalized by size
                ws_utils.append(sum(sim_utils[-1]) / len(G.vertices))
                
                #Run configuration model using WS degree seq
                config_model = configuration_model(num_vertices, degree_seq, p, r)
                cm_diam = util.calc_diameter(config_model)
                cm_graph_diams.append(cm_diam)
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

        if str(r) not in ws_data['ws']:
            ws_data['ws'][str(r)] = {}
        if str(r) not in ws_data['ws-cm']:
            ws_data['ws-cm'][str(r)] = {}
        if str(round(beta, 3)) not in ws_data['ws'][str(r)]:
            ws_data['ws'][str(r)][str(round(beta, 3))] = {}
        if str(round(beta, 3)) not in ws_data['ws-cm'][str(r)]:
            ws_data['ws-cm'][str(r)][str(round(beta, 3))] = {}
        ws_data['ws'][str(r)][str(round(beta, 3))].update(
                {'utils' : ws_social_utils, 'avg_diam' : ws_avg_diam,
                        'plaw' : plaw_resources})
        ws_data['ws-cm'][str(r)][str(round(beta, 3))].update(
                {'utils' : cm_social_utils, 'avg_diam' : cm_avg_diam,
                        'plaw' : plaw_resources})

    if save:
        file_out = 'ws_data_{0}.json'.format(str(uuid.uuid4()))
        file_out = 'sim_data/{0}'.format(file_out)
        with open(file_out, 'w+') as wsd:
            wsd.write(json.dumps(ws_data))

    return ws_data

def run_erdos_renyi(num_vertices, k, r_ep_pairs, strat, strat_params, plaw_resources=False, save=False):
    er_data = defaultdict(dict)
    for r, ep in r_ep_pairs:
        er_social_utils = []
        er_graph_diams = []
        for p in np.linspace(1, 0, 10, endpoint=False):
            er_utils = []
            num_iter = 100
            for i in range(num_iter):
                G = erdos_renyi(num_vertices, ep, p, r)
                
                er_graph_diams.append(util.calc_diameter(G))

                reduce_providers_simplest(G)

                if plaw_resources:
                    powerlaw_dist_time(G, 2)

                sim_strat = strat(**strat_params)
                sim_strat.initialize_model(G)
                sim_g, sim_utils = run_simulation(G, sim_strat)

                #Get global social welfare at end of simulation normalized by size
                er_utils.append(sum(sim_utils[-1]) / len(G.vertices))

            er_social_utils.append((p, er_utils))

        er_avg_diam = sum(er_graph_diams) / len(er_graph_diams)
        if str(r) not in er_data['er']:
            er_data['er'][str(r)] = {}
        er_data['er'][str(r)].update({str(round(ep, 3)) :
                {'utils' : er_social_utils, 'avg_diam' : er_avg_diam,
                    'plaw' : plaw_resources}})
    if save:
        file_out = 'er_data_{0}.json'.format(str(uuid.uuid4()))
        file_out = 'sim_data/{0}'.format(file_out)
        with open(file_out, 'w+') as erd:
            erd.write(json.dumps(er_data))
    return er_data

