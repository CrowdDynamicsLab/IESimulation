import graph_create
import numpy as np
from collections import defaultdict

#Util methods
def even_alloc_time(t, n):
    """
    Allocate t amount of time n ways evenly
    Returns list of allocations and amount of time left
    """
    if t == 0:
        return [0] * n, 0

    time_splits = np.linspace(0, t, n + 1, dtype=int)
    allocs = [ time_splits[i + 1] - time_splits[i]\
            for i in range(len(time_splits) - 1) ]
    return allocs

def run_simulation(G):
    """
    Runs a simple simulation of a graph
    No changes are applied to given values such as
    transmission rate
    Moves decided simultaneously, allocations are random
    People do not know their connections providers at any time

    G: Graph to run simulation over
    """

    def graph_utilities():
        return [ v.utility for v in G.vertices ]

    def graph_time_left():
        return sum([ v.time for v in G.vertices ])

    def calc_util():
        return sum(graph_utilities())

    #Get initial utility
    init_util = calc_util()

    social_opt = max(G.vertices[0].prov_rating.values()) * G.num_people

    #Utilities over time
    utilities = []

    #Get vertex ordering
    np.random.shuffle(G.vertices)

    #Dict for nbor idx tracking
    vtx_cur_nbor = defaultdict(int)

    #Allocate time for each person
    t_allocs = {}
    for v in G.vertices:
        time_splits = even_alloc_time(v.time, v.degree)
        t_allocs[v] = { nbor : ta for nbor, ta in zip(v.nbors, time_splits) }

    iter_num = 0
    while True:
        utilities.append(graph_utilities())
        

        #Get amount of interaction time for each person
        min_times = defaultdict(lambda : {})
        for v in G.vertices:

            #Skip if out of time
            if not v.time:
                continue

            #Get next available nbor
            found_nbor = False
            cur_nbor_idx = vtx_cur_nbor[v]
            nbor, nedge = list(v.edges.items())[cur_nbor_idx]
            for it in range(v.degree):
                vtime = t_allocs[v][nbor]
                ntime = t_allocs[nbor][v]
                if vtime and ntime:
                    found_nbor = True
                    break
                vtx_cur_nbor[v] = (cur_nbor_idx + 1) % len(v.edges)
                nbor, nedge = list(v.edges.items())[cur_nbor_idx]

            #If no nbors available this vtx is effectively finished
            if not found_nbor:
                v.time = 0
                continue

            #Reduce time by one for v and nbor
            assert(t_allocs[v][nbor] > 0)
            assert(t_allocs[nbor][v] > 0)
            assert(v.time > 0)
            assert(nbor.time > 0)
            t_allocs[v][nbor] -= 1
            t_allocs[nbor][v] -= 1
            v.time -= 1
            nbor.time -= 1

            #Check if transmission occured, if so transmit info if needed
            if np.random.random() <= nedge.trate:
                if v.utility > nbor.utility:
                    nbor.provider = v.provider
                elif v.utility < nbor.utility:
                    v.provider = nbor.provider

            #Update vtx's nbor idx
            vtx_cur_nbor[v] = (cur_nbor_idx + 1) % len(v.edges)

        cur_util = calc_util()

        iter_num += 1

        # Stopping condition when all time exhausted
        if not graph_time_left():
            break

    return G, utilities

def simple_sim():
    """
    Simulation on a small 4 regular graph
    Low rate of transmission
    """

    #Create graph
    G = graph_create.const_kregular(4, 10, 0.1, 100)

    print("Running simple simulation")
    return run_simulation(G)

def many_runs_fix_vars(size, deg, trans_rate, val_range,
        time_alloc, var, random_time=False, seq=True):
    """
    Run simple simulation over many graphs
    Useful for analysis of network performance as some variable changes
    size: Fixed size
    deg: Fixed regularity degree
    trans_rate: Baseline transmission probability
    val_range: Range of varying variable
    time_alloc: Initial time allocation per vertex. If this is a function it must take
                one variable, the size, and output a time allocation
    var: Parameter to vary
    random_time: Whether time is allocated randomly
    """

    if var != 'size' and var != 'reg' and var != 'trate':
        raise ValueError("fixed must be one of 'size' or 'reg' or 'trate'")

    sim_results = []
    for cur_var in val_range:
        cur_size = cur_var if var == 'size' else size
        cur_deg = cur_var if var == 'reg' else deg
        cur_trate = cur_var if var == 'trate' else trans_rate

        if callable(time_alloc):
            allocated = time_alloc(g_size)
        else:
            allocated = time_alloc

        cur_G = graph_create.const_kregular(cur_deg, cur_size, cur_trate, allocated)
        res_g, res_utils = run_simulation(cur_G, random_time, seq)
        sim_results.append(res_utils)
    return sim_results

def main():
    simple_sim()

if __name__ == '__main__':
    main()
