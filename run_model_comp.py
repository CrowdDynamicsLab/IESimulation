from multiprocessing import Process

import numpy as np

import gen_model_comp_data as gmcd

# Global params
num_vertices = 100
reg = 4
r_vals = np.linspace(10, 100, 10)

# Split ws across 16 processes
beta_vals = np.linspace(1, 0, 10, endpoint=True)[::-1]

ws_val_pairs = [ (r, b) for r in r_vals for b in beta_vals]
ws_pairs_split = np.array_split(ws_val_pairs, 16)

for param_set in ws_pairs_split:
    sim_proc = Process(target=gmcd.run_watts_strogatz, args=(num_vertices, reg, param_set, False, True))
    sim_proc.start()

# Split er across 8 processes
ep_vals = np.linspace(1, 0, 10, endpoint=False)[::-1]

er_val_pairs = [ (r, ep) for r in r_vals for ep in ep_vals]
er_pairs_split = np.array_split(er_val_pairs, 8)

for param_set in er_pairs_split:
    sim_proc = Process(target=gmcd.run_erdos_renyi, args=(num_vertices, reg, param_set, False, True))
    sim_proc.start()
