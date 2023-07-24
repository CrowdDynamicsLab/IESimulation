"""
Util functions used in attribute search network
"""
import math
from collections import defaultdict
import time

import numpy as np
from scipy.sparse import linalg as scp_sla
from scipy.sparse.csgraph import connected_components as conn_comp_func
import networkx as nx

from tabulate import tabulate

#######################
# Numerical functions #
#######################

# Approximately strictly greater, x > y
def asg(x, y):
    if math.isclose(x, y):
        return False
    return x > y

##########################
# Attr utility functions #
##########################

def homophily(u, v, G):
    return 1 if u.attr_type == v.attr_type else 0

def heterophily(u, v, G):
    return 1 if u.attr_type != v.attr_type else 0

def agg_attr_util(u, G):
    return min(u.sum_edge_util / G.sim_params['max_degree'], 1.0)

###########################
# Structural (normalized) #
###########################

def satisfice(theta):
    def struct(sfunc):
        def swrapper(*args, **kwargs):
            sutil = sfunc(*args, **kwargs)
            if sutil >= theta:
                return 1.0
            return sutil
        return swrapper
    return struct

def triangle_count(v, G):

    # Number of triangles that v is a part of
    clique_deg = G.sim_params['max_clique_size'] - 1
    max_triangles = (clique_deg * (clique_deg - 1)) / 2
    triangle_cnt = np.sum(G.nborhood_adj_mat(v)) / 2

    return min(1.0, triangle_cnt / max_triangles)

def num_disc_nbors(v, G):

    nbor_mat = G.nborhood_adj_mat(v)
    nbor_deg = np.sum(nbor_mat, axis=0)
    num_con = np.count_nonzero(nbor_deg)
    num_disc = v.degree - num_con
    return min(1.0, num_disc / G.sim_params['max_degree'])

def num_nbor_comp_scipy(v, G):

    nbor_mat = G.nborhood_adj_mat(v)
    conn_comps = conn_comp_func(nbor_mat)
    num_comps = conn_comps[0]

    return min(1.0, num_comps / G.sim_params['max_degree'])

def num_nbor_comp_nx(v, G):

    nbor_mat = G.nborhood_adj_mat(v)
    G_nx = nx.from_numpy_array(nbor_mat)
    num_comps = nx.number_connected_components(G_nx)

    return min(1.0, num_comps / G.sim_params['max_degree'])


##################
# Cost functions #
##################

def calc_cost(u, G, ignore_indirect=True):
    if ignore_indirect:
        return u.degree / G.sim_params['max_degree']

    direct_cost = G.sim_params['direct_cost']
    indirect_cost = G.sim_params['indirect_cost']

    total_direct_cost = u.degree * direct_cost
    total_indirect_cost = (u.nborhood_degree / 2) * indirect_cost

    return total_direct_cost + total_indirect_cost

def calc_all_costs(G):
    degree_vec = np.sum(G.adj_matrix, axis=1)
    return degree_vec / G.sim_params['max_degree']

def remaining_budget(u, G):
    u_cost = calc_cost(u, G)
    return 1 - u_cost

####################
# Edge calculation #
####################

# Seq selection functions
def seq_edge_sel_logged(G, edge_proposals):
    return seq_projection_edge_edit(G, edge_proposals, log=True)

def seq_edge_sel_silent(G, edge_proposals):
    return seq_projection_edge_edit(G, edge_proposals, log=False)

def seq_projection_edge_edit(G, edge_proposals, allow_early_drop=True, assume_accept=True, log=True):
    # Sequentially (non-random) pick one edge to propose to via projection
    # of multiobjective optimization function
    # Assumes even split of coefficients

    #print(edge_proposals)

    util_agg = G.sim_params['util_agg']

    # Agents who have proposed to v
    add_candidates = { v : [] for v in G.vertices }
    for u, u_prop in edge_proposals.items():
        if u_prop is not None:
            add_candidates[u_prop].append(u)

    # For each candidate store its max cost inc action and its cost non-inc action
    v_move = { }

    for v in G.vertices:

        action_dict = {'add': 0, 'delete': 0, 'swap': 0, 'resolve': 0}

        #NOTE: the init max val would be negative for an optimistic agent
        max_inc_cand = None
        max_inc_val = 0
        max_dec_cand = None
        max_dec_val = 0
        max_ninc_val = 0
        max_ninc_cand = None

        # No options
        if len(v.nbors) == 0 and len(add_candidates[v]) == 0:
            continue

        cur_attr_util = v.data['total_attr_util'](v, G)
        cur_struct_util = v.data['struct_util'](v, G)
        cur_cost = calc_cost(v, G)

        # Satiated
        cur_agg_util = util_agg(
            cur_attr_util,
            cur_struct_util,
            cur_cost, v, G
        )


        # No point checking proposal/single drop values if over budget anyways
        if remaining_budget(v, G) < 0:
            raise ValueError('Agents should never have negative budget')

        if cur_agg_util >= 2.0:
            continue

        can_add_nbor = remaining_budget(v, G) > 0

        # If proposals are not assumed to be accepted then the expected utility of doing nothing is 0
        do_nothing_val = 0

        # If we assume proposals are accepted then add the proposal edge for checking
        if assume_accept and edge_proposals[v] is not None:
            G.add_edge(v, edge_proposals[v])
            attr_change = G.potential_utils[v.vnum][edge_proposals[v].vnum] / G.sim_params['max_degree']
            cost_change = 1  / G.sim_params['max_degree']
            agg_util = util_agg(
                cur_attr_util + attr_change,
                v.data['struct_util'](v, G),
                cur_cost + cost_change,
                v, G
            )
            do_nothing_val = agg_util - cur_agg_util

        for u in v.nbors:

            # Consider drop choices
            if not allow_early_drop and can_add_nbor:
                # Disallow early drops
                break

            G.remove_edge(v, u)
            attr_change = -1 * G.potential_utils[v.vnum][u.vnum] / G.sim_params['max_degree']
            cost_change = -1 / G.sim_params['max_degree']
            agg_util = util_agg(
                cur_attr_util + attr_change,
                v.data['struct_util'](v, G),
                cur_cost + cost_change,
                v, G
            )
            agg_change = agg_util - cur_agg_util

            if asg(agg_change, max_dec_val):
                max_dec_val = agg_change
                max_dec_cand = ('d', u)

            G.add_edge(v, u)

        # Drop proposal edge for addition checking
        if assume_accept and edge_proposals[v] is not None:
            G.remove_edge(v, edge_proposals[v])

        for u in add_candidates[v]:

            # Consider edge formation choices
            if G.are_neighbors(u, v):
                print(u, v, add_candidates[v], u.nbors, v.nbors)
                raise ValueError('Neighbors should not propose to one another')

            # It is possible that u proposed to v and v proposed to u
            if assume_accept and edge_proposals[v] is not None and edge_proposals[v] != u:
                G.add_edge(v, edge_proposals[v])

            G.add_edge(v, u)

            # Would have budget to add (remaining_budget assumes add here)
            if remaining_budget(v, G) > 0:

                attr_change = G.potential_utils[v.vnum][u.vnum] / G.sim_params['max_degree']
                cost_change = 1  / G.sim_params['max_degree']
                agg_util = util_agg(
                    cur_attr_util + attr_change,
                    v.data['struct_util'](v, G),
                    cur_cost + cost_change,
                    v, G
                )
                agg_change = agg_util - cur_agg_util

                if asg(agg_change, max_inc_val):
                    max_inc_val = agg_change
                    max_inc_cand = ('a', u)

            G.remove_edge(v, u)

            if assume_accept and edge_proposals[v] is not None and edge_proposals[v] != u:
                G.remove_edge(v, edge_proposals[v])

        # Set cost non-increasing action
        if asg(max_dec_val, do_nothing_val):
            max_ninc_cand = max_dec_cand
            max_ninc_val = max_dec_val
        else:
            max_ninc_val = do_nothing_val

        if (G.sim_params['max_degree'] - v.degree) < 1 and edge_proposals[v] is not None:
            print('Agent v budget', remaining_budget(v, G), 'limit', 1 / G.sim_params['max_degree'])
            raise ValueError('If agent budget was 0 should not have proposed')
        elif (G.sim_params['max_degree'] - v.degree) <= 1 and edge_proposals[v] is not None:
            v_move[v] = max_ninc_cand
        elif asg(max_inc_val, max_ninc_val) or math.isclose(max_inc_val, max_ninc_val):
            v_move[v] = max_inc_cand
        else:
            v_move[v] = max_ninc_cand

    for v, cand_tuple in v_move.items():
        if cand_tuple is None:
            continue
        action = cand_tuple[0]
        cand = cand_tuple[1]
        if action == 'd':
            G.remove_edge(v, cand)
            action_dict['delete'] = action_dict['delete'] + 1
        elif action == 'a':
            G.add_edge(v, cand)
            action_dict['add'] = action_dict['add'] + 1
    return v_move

# Utility aggregation functions
def linear_util_agg(a, s, c, v, G):
    return a + s

# Revelation proposal sets
def indep_revelation(G):

    # Do not allow self revelation
    rand_sel = np.random.randint(low=0, high=G.num_people, size=G.num_people)
    self_sel = np.arange(0, G.num_people)

    self_match = np.arange(len(rand_sel))[rand_sel == self_sel]
    while len(self_match) > 0:
        new_rand_sel = np.random.randint(low=0, high=G.num_people, size=len(self_match))
        for ni, i in enumerate(self_match):
            rand_sel[i] = new_rand_sel[ni]
        self_match = np.arange(len(rand_sel))[rand_sel == self_sel]

    return rand_sel

#######################
# Networkx Conversion #
#######################

def graph_to_nx(G, with_labels=True):
    """
    Converts a graph from sim_lib.graph to a networkx graph (undirected)
    """

    nx_G = nx.Graph()

    for vtx in G.vertices:
        if with_labels:
            attr_util = vtx.data['total_attr_util'](vtx, G)
            struct_util = vtx.data['struct_util'](vtx, G)
            cost = calc_cost(vtx, G)
            #color = vtx.data['color']
            shape = vtx.data['shape']
            struct = vtx.data['struct']
            nx_G.add_node(vtx,
                attr_util=attr_util,
                struct_util=struct_util,
                struct = struct,
                cost=cost,
                #color=color,
                shape=shape)
        else:
            nx_G.add_node(vtx)


    for vtx in G.vertices:
        for nbor in vtx.nbors:
            util = vtx.edges[nbor].util
            nx_G.add_edge(vtx, nbor, capacity=1.0, util=util)

    return nx_G
