from collections import defaultdict

import numpy as np

import sim_lib.graph_create as graph_create

def run_simulation(G, strategy, util_times=False):
    """
    Runs a simple simulation of a graph
    No changes are applied to given values such as
    transmission rate
    Moves decided simultaneously, allocations are random
    People do not know their connections providers at any time

    G: Graph to run simulation over
    strategy: Resource allocation strategy to use - must be initialized
    util_times: If True return times at which each person received their
    best util and who they received it from
    """

    assert (not strategy.initialize), 'strategy must be initialized'

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

    util_time_map = { v : { 'from' : [v], 'ut' : v.utility, 'iter' : 0, 'int_cnt' : 0 } \
            for v in G.vertices }

    iter_num = 1
    while iter_num < max([v.time for v in G.vertices]) + 1:
        global_util = graph_utilities()
        utilities.append(global_util)
        if sum(global_util) == social_opt:
            break

        #Get amount of interaction time for each person
        min_times = defaultdict(lambda : {})
        for v in G.vertices:

            #Skip if out of time
            if not v.time or (v.degree == 0):
                v.time = 0
                continue

            #If no nbors will become available this vertex is effectively done
            #NOTE: This will have to change when ability to add edges is introduced
            if sum([nbor.time for nbor in v.nbors]) == 0:
                v.time = 0
                continue

            #Get next available nbor from strategy
            avail_nbors = strategy.get_available_nbor(v)

            if not avail_nbors:
                continue

            nbor, nedge = avail_nbors

            #For self selection do nothing
            if nbor.vnum == v.vnum:
                continue

            #Decrement time for v and nbor
            v.time -= 1
            nbor.time -= 1

            #Increment interaction count for both vertices
            v.add_init_int()
            nbor.add_recv_int()

            #Save data needed to calculate cost of action
            nbor_prev_util = nbor.utility
            v_prev_util = v.utility

            #Check if transmission occured, if so transmit info if needed
            if np.random.random() <= nedge.trate:
                if v.utility > nbor.utility:
                    util_time_map[nbor]['from'] = util_time_map[v]['from'].copy()
                    util_time_map[nbor]['from'].append(nbor)
                    util_time_map[nbor]['ut'] = v.utility
                    util_time_map[nbor]['iter'] = iter_num
                    util_time_map[nbor]['int_cnt'] = nbor.total_ints
                    nbor.provider = v.provider
                elif v.utility < nbor.utility:
                    util_time_map[v]['from'] = util_time_map[nbor]['from'].copy()
                    util_time_map[v]['from'].append(v)
                    util_time_map[v]['ut'] = nbor.utility
                    util_time_map[v]['iter'] = iter_num
                    util_time_map[v]['int_cnt'] = v.total_ints
                    v.provider = nbor.provider

            #Update time allocations in strategy
            strategy.update_time_alloc(v_prev_util, v, nbor_prev_util, nbor)

        cur_util = calc_util()

        iter_num += 1

        # Stopping condition when all time exhausted
        if graph_time_left() == 0:
            break

    if util_times:
        return G, utilities, util_time_map

    return G, utilities

