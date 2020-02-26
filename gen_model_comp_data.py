from math import ceil
from collections import defaultdict
import json
import uuid

import numpy as np

from simulation import run_simulation
from graph_create import reduce_providers_simplest, powerlaw_dist_time, watts_strogatz, erdos_renyi, configuration_model
import util

import argparse

def run_watts_strogatz(num_vertices, k, r_b_pairs, plaw_resources=False, save=False):
    ws_data = defaultdict(dict)
    for r, beta in r_b_pairs:
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

                ws_diam = util.calc_diameter(ring_lat)
                ws_graph_diams.append(ws_diam)

                reduce_providers_simplest(ring_lat)

                if plaw_resources:
                    powerlaw_dist_time(ring_lat, 2)

                sim_g, sim_utils = run_simulation(ring_lat)

                #Get global social welfare at end of simulation normalized by size
                ws_utils.append(sum(sim_utils[-1]) / len(ring_lat.vertices))
                
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

        if str(beta) not in ws_data[str(r)]:
            ws_data[str(r)][str(beta)] = {}
        ws_data[str(r)][str(beta)]['ws'] = \
                {'utils' : ws_social_utils, 'avg_diam' : ws_avg_diam}
        ws_data[str(r)][str(beta)]['cm'] = \
                {'utils' : cm_social_utils, 'avg_diam' : cm_avg_diam}

    if save:
        file_out = 'ws_data_{0}.json'.format(str(uuid.uuid4()))
        file_out = 'sim_data/{0}'.format(file_out)
        with open(file_out, 'w+') as wsd:
            wsd.write(json.dumps(ws_data))

    return ws_data

def run_erdos_renyi(num_vertices, k, r_ep_pairs, plaw_resources=False, save=False):
    er_data = defaultdict(dict)
    for r, ep in r_ep_pairs:
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
        er_data[str(r)][str(ep)] = {'er' :
                {'utils' : er_social_utils, 'avg_diam' : er_avg_diam}}

    if save:
        file_out = 'er_data_{0}.json'.format(str(uuid.uuid4()))
        file_out = 'sim_data/{0}'.format(file_out)
        with open(file_out, 'w+') as erd:
            erd.write(json.dumps(er_data))
    return er_data

if __name__ == '__main__':
    num_vertices = 100
    reg = 4
 
    parser = argparse.ArgumentParser(description='Run model simulation')
    parser.add_argument('r_start', metavar='N', type=int,
                        help='Start of r value')
    parser.add_argument('r_end', metavar='N', type=int,
                        help='End of r value')
    parser.add_argument('param_start', metavar='N', type=float,
                        help='Start of other param value')
    parser.add_argument('param_end', metavar='N', type=float,
                        help='End of other param value')
    parser.add_argument('sim_type', metavar='N', type=str,
                        help='Must be one of ws or er')
    parser_args = parser.parse_args()

    r_start = parser_args.r_start
    r_end = parser_args.r_end
    param_start = parser_args.param_start
    param_end = parser_args.param_end
    assert r_start < r_end, 'r start must be less than end'
    assert param_start < param_end, 'Param start must be less than end'
    assert parser_args.sim_type in ['ws', 'er'], 'Sim type must be one of ws or er'

    if parser_args.sim_type == 'ws':
        r_vals = np.arange(r_start, r_end, 10)
        beta_vals = np.arange(param_start, param_end, 0.1)
        ws_json = run_watts_strogatz(num_vertices, reg, zip(r_vals, beta_vals))

        file_out = 'ws_data_{0}.json'.format(str(uuid.uuid4()))
        file_out = 'sim_data/{0}'.format(file_out)
        with open(file_out, 'w+') as wsd:
            wsd.write(json.dumps(ws_json))
    else:
        r_vals = np.arange(r_start, r_end, 10)
        ep_vals = np.arange(param_start, param_end, 0.1)
        er_json = run_erdos_renyi(num_vertices, reg, zip(r_vals, ep_vals))
        
        file_out = 'er_data_{0}.json'.format(str(uuid.uuid4()))
        file_out = 'sim_data/{0}'.format(file_out)
        with open(file_out, 'w+') as erd:
            erd.write(json.dumps(er_json))
