from multiprocessing import Process

import numpy as np

import sim_strategies
import gen_model_comp_data as gmcd

# Global params
num_vertices = 100
reg = 4
r_vals = [ 2.0 ** k for k in range(1, 8) ]

# Split ws processes
beta_vals = [ 0.1 ]

ws_val_pairs = [ (r, b) for r in r_vals for b in beta_vals]
ws_pairs_split = np.array_split(ws_val_pairs, 2)

def run_ws():
    for param_set in ws_pairs_split:
        sim_proc = Process(target=gmcd.run_watts_strogatz,
                args=(num_vertices, reg, param_set, sim_strategies.EvenAlloc, {}, False, True))
        sim_proc.start()
        plaw_proc = Process(target=gmcd.run_watts_strogatz,
                args=(num_vertices, reg, param_set, sim_strategies.EvenAlloc, {}, True, True))
        plaw_proc.start()

# Split er processes
ep_infl_vals = [1 / (num_vertices ** 2), 1 / num_vertices, np.log(num_vertices) / num_vertices ]
ep_vals = [ep_infl_vals[0], *np.linspace(ep_infl_vals[0], ep_infl_vals[1], 4, True)[1:3],
        ep_infl_vals[1], *np.linspace(ep_infl_vals[1], ep_infl_vals[2], 4, True)[1:3],
        ep_infl_vals[2]]

er_val_pairs = [ (r, ep) for r in r_vals for ep in ep_vals]
er_pairs_split = np.array_split(er_val_pairs, 22)

def run_er():
    for param_set in er_pairs_split:
        sim_proc = Process(target=gmcd.run_erdos_renyi,
                args=(num_vertices, reg, param_set, sim_strategies.EvenAlloc, {}, False, True))
        sim_proc.start()
        plaw_proc = Process(target=gmcd.run_erdos_renyi,
                args=(num_vertices, reg, param_set, sim_strategies.EvenAlloc, {}, True, True))
        plaw_proc.start()

if __name__ == '__main__':
    run_ws()
    run_er()
