import graph
import numpy as np
from collections import defaultdict

#Util methods
def alloc_time(t, n):
    """
    Allocate t amount of time n ways evenly
    Returns list of allocations and amount of time left
    """
    if t == 0:
        return [0] * n, 0

    allocs = np.linspace(0, t, n)
    return list(allocs), 0

def calc_trans_prob(tr, ta):
    """
    Takes base transmission rate (tr) and time allocated (ta)
    to determine the probability of information transmission
    """
    return tr ** ta

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

def run_simulation(G):
    """
    Runs a simple simulation of a graph
    No changes are applied to given values such as
    transmission rate
    Moves decided simultaneously, allocations are random
    People do not know their connections providers at any time

    G: Graph to run simulation over
    """

    def calc_util():
        return sum([ v.utility for v in G.vertices ])

    #Get initial utility
    init_util = calc_util()

    social_opt = max(G.vertices[0].prov_rating.values()) * G.num_people
    print("Optimal social welfare: {0}".format(social_opt))

    print("Initial utility: {0}".format(init_util / social_opt))

    prev_util = init_util
    iter_num = 0
    while True:
        
        #Simultaneously allocate time for each person
        t_allocs = {}
        for v in G.vertices:
            t_allocs[v], t_left = alloc_time(v.time, v.degree)
            v.time = t_left

        #Get amount of interaction time for each person
        num_nonzero = 0
        time_expected = defaultdict(lambda : {})
        for v in G.vertices:
            for nbor, nedge in v.edges.items():
                if v in time_expected[nbor] or nbor in time_expected[v]:
                    continue

                #Amount of time agreed to spend
                vtime = t_allocs[v].pop()
                ntime = t_allocs[nbor].pop()
                min_time = min(vtime, ntime)

                #Get expected time to transmit information
                tprob = calc_trans_prob(nedge.trate, min_time)
                texpec = calc_expected_time(tprob)
                time_expected[v][nbor] = texpec
                time_expected[nbor][v] = texpec

                v.time += max(vtime - texpec, 0)
                nbor.time += max(ntime - texpec, 0)

                if min_time > 0:
                    num_nonzero += 1

        if not num_nonzero:
            print("All time expended, ending")
            break

        #Transfer information

        #Current min expected time
        cur_met = min_expec_time(time_expected)
        while (cur_met > 0):
            for v in G.vertices:
                for nbor in v.edges:
                    time_expected[v][nbor] -= cur_met

                    #Expected time has passed
                    if time_expected[v][nbor] <= 0:
                        if v.utility < nbor.utility:
                            v.provider = nbor.provider
            cur_met = min_expec_time(time_expected)

        cur_util = calc_util()
        print("At {0} utility was {1}".format(iter_num, cur_util / social_opt))

        if prev_util == cur_util:
            print("No improvement, ending")
            break

        prev_util = cur_util
        iter_num += 1
    return G

def simple_sim():
    """
    Simulation on a small 4 regular graph
    Low rate of transmission
    """

    #Create graph
    G = graph.const_kregular(4, 10, 0.1, 100)

    print("Running simple simulation")
    return run_simulation(G)

def main():
    simple_sim()

if __name__ == '__main__':
    main()
