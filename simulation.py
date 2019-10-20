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
    return list(allocs), 0

def rand_alloc_time(t, n):
    """
    Allocates t amount of time n ways randomly
    Based on dirichlet distribution
    Returns list of allocations and amount of time left
    NOTE: As a rounding error allocs may sum to be slightly
    greater than t
    """
    if t == 0:
        return [0] * n, 0
    
    allocs = np.random.dirichlet(np.ones(n)) * t
    return list(allocs), 0

def calc_trans_prob(tr, ta):
    """
    Takes base transmission rate (tr) and time allocated (ta)
    to determine the probability of information transmission
    NOTE: This needs to be refined
    How should time allocated impact this? Because expected time
    is calculated and compared to time allocated time allocated
    has no direct impact on probability but unless time allocated
    is sufficiently large no transmission will occur
    """
    return tr

def calc_expected_time(tp):
    """
    Given the transmission probability returns the expected
    amount of time for it to occur
    Assumes binomial dist
    """
    if tp == 0:
        return np.infty
    return 1 / tp

def min_expec_time(expec_times):
    """
    Gets the minimum expected time in dict of dicts
    vertex : { nbor : expected time }
    """
    vtex_mins = [ min(nbor.values()) for nbor in expec_times.values() ]
    return min(vtex_mins)

def simul_info_transfer(G, time_expected):
    #Simultanenusly subtract all expected time
    #Current min expected time
    cur_met = min_expec_time(time_expected)
    while (cur_met > 0):
        for v in np.random.permutation(G.vertices):
            for nbor in v.edges:
                time_expected[v][nbor] -= cur_met

                #Expected time has passed
                if time_expected[v][nbor] <= 0:
                    if v.utility < nbor.utility:
                        v.provider = nbor.provider
        cur_met = min_expec_time(time_expected)

def seq_info_transfer(G, min_times):
    # Sequentially iterate over each vertex and attempt info transmission

    trate = next(iter(G.vertices[0].edges.values())).trate
    for v in np.random.permutation(G.vertices):
        for nbor in np.random.permutation(list(v.edges.keys())):
            if min_times[v][nbor] == -1 or min_times[nbor][v] == -1:
                continue

            for t_iter in range(min_times[v][nbor]):
                if np.random.rand() < trate:

                    #Info swapped, change utilities if need be
                    if v.utility < nbor.utility:
                        v.provider = nbor.provider
                    elif nbor.utility < v.utility:
                        nbor.provider = v.provider

                    #No need to continue talking
                    min_times[v][nbor] = -1
                    min_times[nbor][v] = -1
                    break

            #Allocated time used up, don't double count
            min_times[v][nbor] = -1
            min_times[nbor][v] = -1

def run_simulation(G, random_time=False, seq=True):
    """
    Runs a simple simulation of a graph
    No changes are applied to given values such as
    transmission rate
    Moves decided simultaneously, allocations are random
    People do not know their connections providers at any time

    G: Graph to run simulation over
    seq: If true then runs transmission step sequentially, else simultaneously
    """

    def graph_utilities():
        return [ v.utility for v in G.vertices ]

    def calc_util():
        return sum(graph_utilities())

    #Set up time allocation method
    alloc_time = None
    if random_time:
        alloc_time = rand_alloc_time
    else:
        alloc_time = even_alloc_time

    #Get initial utility
    init_util = calc_util()

    social_opt = max(G.vertices[0].prov_rating.values()) * G.num_people

    #Utilities over time
    utilities = []

    prev_util = init_util
    iter_num = 0
    while True:
        utilities.append(graph_utilities())
        
        #Simultaneously allocate time for each person
        t_allocs = {}
        for v in G.vertices:

            #Time is reallocated at the beginning of each iteration
            t_allocs[v], t_left = alloc_time(v.time, v.degree)

        #Get amount of interaction time for each person
        num_nonzero = 0
        time_expected = defaultdict(lambda : {})
        min_times = defaultdict(lambda : {})
        for v in G.vertices:
            for nbor, nedge in v.edges.items():

                #In simultaneous case
                if v in time_expected[nbor] or nbor in time_expected[v]:
                    continue

                #In sequential case
                if v in min_times[nbor] or nbor in min_times[v]:
                    continue

                #Amount of time agreed to spend
                vtime = t_allocs[v].pop()
                ntime = t_allocs[nbor].pop()
                min_time = min(vtime, ntime)

                if seq:
                    min_times[v][nbor] = min_time
                    min_times[nbor][v] = min_time
                else:
                    #Get expected time to transmit information
                    tprob = calc_trans_prob(nedge.trate, min_time)
                    texpec = calc_expected_time(tprob)
                    time_expected[v][nbor] = texpec
                    time_expected[nbor][v] = texpec

                if min_time > 0:
                    num_nonzero += 1

        if not num_nonzero:
            break

        #Transfer information
        if seq:
            seq_info_transfer(G, min_times)
        else:
            simul_info_transfer(G, time_expected)

        cur_util = calc_util()

        if prev_util == cur_util:
            break

        prev_util = cur_util
        iter_num += 1
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
