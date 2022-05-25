"""
Util functions used in attribute search network
"""
import math
from collections import defaultdict
import time

import numpy as np
from scipy.sparse import linalg as scp_sla
import networkx as nx

from tabulate import tabulate

##########################
# Attr utility functions #
##########################

def gen_similarity_funcs():
    def homophily(u, v, G):
        return 1 if u.attr_type == v.attr_type else 0

    def heterophily(u, v, G):
        return 1 if u.attr_type == v.attr_type else 0

    return homophily, heterophily

def gen_attr_util_func(thresh):
    def util_func(u, G):
        if thresh == 0.0:
            return 1.0
            # return 1.0 if u.sum_edge_util > 0 else 0.0
        thresh_count = math.ceil(G.sim_params['max_degree'] * thresh)
        if u.sum_edge_util >= thresh_count:
            return 1.0
        return min(u.sum_edge_util / thresh_count, 1.0)
    return util_func

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

def subset_budget_resolution(v, G, util_agg):

    # Brings v under budget by removing subset of edges incident to v
    cur_attr_util = v.data['total_attr_util'](v, G)
    cur_struct_util = v.data['struct_util'](v, G)
    cur_cost = calc_cost(v, G)
    cur_util = util_agg(cur_attr_util, cur_struct_util, cur_cost, v, G)
    while remaining_budget(v, G) < 0:
        min_util_loss = np.inf
        drop_candidate = None

        # Out of place shuffle
        shuffled_nbors = np.random.choice(v.nbors, v.degree, replace=False)
        for nbor in shuffled_nbors:
            G.remove_edge(v, nbor)
            pot_attr_util = v.data['total_attr_util'](v, G)
            pot_struct_util = v.data['struct_util'](v, G)
            pot_cost = calc_cost(v, G)
            potential_util = util_agg(pot_attr_util, pot_struct_util, pot_cost, v, G)
            util_loss = cur_util - potential_util
            G.add_edge(v, nbor)
            if util_loss < min_util_loss:
                min_util_loss = util_loss
                drop_candidate = nbor
        G.remove_edge(v, drop_candidate)
        cur_util -= min_util_loss
    assert remaining_budget(v, G) >= 0, 'did not resolve budget'

# Seq selection functions
def seq_projection_single_selection(G, edge_proposals, log):
    return seq_projection_edge_edit(G, edge_proposals, substitute=_allow_sub, log=log)

def seq_edge_sel_logged(G, edge_proposals):
    return seq_projection_edge_edit(G, edge_proposals, log=True)

def seq_edge_sel_silent(G, edge_proposals):
    return seq_projection_edge_edit(G, edge_proposals, log=False)

def seq_projection_edge_edit(G, edge_proposals, substitute=True, allow_early_drop=True, log=True):
    # Sequentially (non-random) pick one edge to propose to via projection
    # of multiobjective optimization function
    # Assumes even split of coefficients

    #if log:
        #NOTE: Consider adding back once not global
        #print('proposals:', edge_proposals)

    util_agg = G.sim_params['util_agg']

    proposed_by = { v : [ u for u, u_prop in edge_proposals.items() if v == u_prop ] \
            for v in G.vertices }

    # Prepare metadata collection for analysis
    metadata = { }

    for v in G.vertices:
        metadata[v] = { }
        metadata[v]['num_proposals'] = 0

        attr_util_deltas = []
        struct_util_deltas = []
        cost_deltas = []

        # No options
        if len(v.nbors) == 0 and len(proposed_by[v]) == 0:
            if log:
                print('-----------------------------------------')
                print(v, 'had no nbors or props')
                print('-----------------------------------------')
            metadata[v]['action'] = 'none'
            metadata[v]['attr_delta'] = 0
            metadata[v]['struct_delta'] = 0
            metadata[v]['cost_delta'] = 0
            continue
            
        cur_attr_util = v.data['total_attr_util'](v, G)
        cur_struct_util = v.data['struct_util'](v, G)
        cur_cost = calc_cost(v, G)

        # Satiated
        cur_util_agg = G.sim_params['util_agg'](
            cur_attr_util,
            cur_struct_util,
            cur_cost, v, G
        )

        # No point checking proposal/single drop values if over budget anyways
        if remaining_budget(v, G) < 0:

            # Budget resolution
            if log:
                print('has to resolve budget via subset drop')
            subset_budget_resolution(v, G, util_agg)
            metadata[v]['action'] = 'budget_resolve'
            metadata[v]['attr_delta'] = v.data['total_attr_util'](v, G) - cur_attr_util
            metadata[v]['struct_delta'] = v.data['struct_util'](v, G) - cur_struct_util
            metadata[v]['cost_delta'] = cur_cost - calc_cost(v, G)
            continue
            
        if cur_util_agg == 2.0:
            if log:
                print('-----------------------------------------')
                print(v, 'satiated, doing nothing')
                print('-----------------------------------------')
            metadata[v]['action'] = 'satiated'
            metadata[v]['attr_delta'] = 0
            metadata[v]['struct_delta'] = 0
            metadata[v]['cost_delta'] = 0
            continue

        candidates = []

        #can_add_nbor = remaining_budget(v, G) >= G.sim_params['direct_cost']
        can_add_nbor = remaining_budget(v, G) > 0
        for u in v.nbors:

            # Consider drop choices
            if not allow_early_drop and can_add_nbor:

                # Disallow early drops
                break

            #u_cur_au = u.data['total_attr_util'](u, G)
            #u_cur_su = u.data['struct_util'](u, G)
            #u_cur_agg_util = util_agg(u_cur_au, u_cur_su, calc_cost(u, G), u, G)

            G.remove_edge(v, u)
            attr_change = -1 * G.potential_utils[v.vnum][u.vnum] / G.sim_params['max_degree']
            struct_change = v.data['struct_util'](v, G) - cur_struct_util
            attr_util_deltas.append(attr_change)
            struct_util_deltas.append(struct_change)
            cost_deltas.append(1 / G.sim_params['max_degree'])

            """
            # Check dropped vtx util change
            u_drop_au = u.data['total_attr_util'](u, G) - u_cur_au
            u_drop_su = u.data['struct_util'](u, G) - u_cur_su
            u_drop_cost = calc_cost(u, G)
            u_drop_agg_util = util_agg(u_drop_au, u_drop_su, u_drop_cost, u, G)

            # Require that drops be mutually beneficial
            #if u_drop_agg_util - u_cur_agg_util < 0:
            #   G.add_edge(v, u)
            #   continue

            attr_util_deltas.append(v.data['total_attr_util'](v, G) - cur_attr_util)
            struct_util_deltas.append(v.data['struct_util'](v, G) - cur_struct_util)

            # Ordered so that reduction in cost is positive
            cost_deltas.append(cur_cost - calc_cost(v, G))
            """
            G.add_edge(v, u)

            candidates.append(u)

        for u in proposed_by[v]:

            # Consider edge formation choices
            if u in v.nbors:
                continue

            G.add_edge(v, u)

            # Would have budget to add (remaining_budget assumes add here)
            if remaining_budget(v, G) >= 0:
                attr_change = G.potential_utils[v.vnum][u.vnum] / G.sim_params['max_degree']
                attr_util_deltas.append(attr_change)
                struct_util_deltas.append(v.data['struct_util'](v, G) - cur_struct_util)
                cost_deltas.append(-1  / G.sim_params['max_degree'])
                metadata[v]['num_proposals'] += 1
                candidates.append(u)

            if (not allow_early_drop and can_add_nbor) or not substitute:

                # If can still add do not allow a substitution
                # Edge case when no available vertex at direct cost
                G.remove_edge(v, u)
                continue

            for w in v.nbors:

                # Check substitution
                if u == w:
                    continue

                #w_cur_au = u.data['total_attr_util'](u, G)
                #w_cur_su = u.data['struct_util'](u, G)
                #w_cur_agg_util = util_agg(w_cur_au, w_cur_su, calc_cost(w, G), w, G)

                G.remove_edge(v, w)

                """
                # Check dropped vtx util change
                w_drop_au = w.data['total_attr_util'](u, G) - w_cur_au
                w_drop_su = w.data['struct_util'](u, G) - w_cur_su
                w_drop_cost = calc_cost(w, G)
                w_drop_agg_util = util_agg(w_drop_au, w_drop_su, w_drop_cost, w, G)

                # Require that drops be mutually beneficial
                #if w_drop_agg_util - w_cur_agg_util < 0:
                    #G.add_edge(v, w)
                    #continue
                """

                if remaining_budget(v, G) >= 0:

                    # Disallow substitution if over budget
                    attr_util_deltas.append(v.data['total_attr_util'](v, G) - cur_attr_util)
                    struct_util_deltas.append(v.data['struct_util'](v, G) - cur_struct_util)

                    # Ordered so that reduction in cost is positive
                    cost_deltas.append(cur_cost - calc_cost(v, G))

                    candidates.append((u, w))
                G.add_edge(v, w)
            G.remove_edge(v, u)

        if len(candidates) == 0:
            if log:
                print('\n-------------------------------------------------------------------')
                header = np.array([('Vertex', 'Degree','Budget Used'), (str(v.vnum), str(v.degree), str(calc_cost(v,G)))])
                print(tabulate(header))
                print(v, 'had no options')
            metadata[v]['action'] = 'none'
            metadata[v]['attr_delta'] = 0
            metadata[v]['struct_delta'] = 0
            metadata[v]['cost_delta'] = 0
            continue

        candidate_value_points = list(zip(attr_util_deltas, struct_util_deltas, cost_deltas))
        norm_values = [ util_agg(a, s, c, v, G) for a, s, c in candidate_value_points ]
        max_val_candidate_idx = np.argmax(norm_values)
        max_val_candidate = candidates[ max_val_candidate_idx ]

        if log:
            print('\n-------------------------------------------------------------------')
            header = np.array([('Vertex', 'Degree','Budget Used'), (str(v.vnum), str(v.degree), str(calc_cost(v,G)))])
            print(tabulate(header, headers="firstrow"))
            candidate_value_rounded = [ (round(a, 2), round(s, 2), round(c, 2)) for (a, s, c) in candidate_value_points ]
            data = np.array([candidates, candidate_value_rounded]).T
            print('\n', tabulate(data, headers=['Candidates', '(Attr Util, Struct Util, Cost)']))
            #print([ s + c for a, s, c in candidate_value_points ])
            print('\nmax:', max_val_candidate)
            if not v.data['optimistic'] and norm_values[max_val_candidate_idx] <= 0:
                print('chose do nothing')
            elif type(max_val_candidate) != tuple and max_val_candidate not in v.nbors:
                print('chose to add', max_val_candidate)
            elif type(max_val_candidate) == tuple:
                u, w = max_val_candidate
                print('chose to substitute', w, 'for', u)
            elif allow_early_drop and max_val_candidate in v.nbors and norm_values[max_val_candidate_idx] > 0:
                print('chose to early drop', max_val_candidate)
            else:
                print('chose do nothing')

        if remaining_budget(v, G) < 0:
            raise(ValueError, "Reached end of edge selection when budget resolution should have been enforced")

        # Either optimistic or max val move is stirctly positive
        max_val = norm_values[max_val_candidate_idx]
        if (max_val < 0) or (not v.data['optimistic'] and max_val == 0):

            # No non-negative change candidates
            metadata[v]['action'] = 'none'
            metadata[v]['attr_delta'] = 0
            metadata[v]['struct_delta'] = 0
            metadata[v]['cost_delta'] = 0
            continue
        elif type(max_val_candidate) != tuple and max_val_candidate not in v.nbors:

            # Edge formation
            metadata[v]['action'] = 'addition'
            G.add_edge(v, max_val_candidate)
        elif type(max_val_candidate) == tuple:

            # Substitution
            metadata[v]['action'] = 'substitution'
            add_vtx, rem_vtx = max_val_candidate
            G.add_edge(v, add_vtx)
            G.remove_edge(v, rem_vtx)
        elif allow_early_drop and max_val_candidate in v.nbors and max_val > 0:

            # Early single drop
            metadata[v]['action'] = 'drop'
            G.remove_edge(v, max_val_candidate)
        else:

            # When dropping gives 0 utility change
            metadata[v]['action'] = 'none'
            metadata[v]['attr_delta'] = 0
            metadata[v]['struct_delta'] = 0
            metadata[v]['cost_delta'] = 0
            continue

        # Add additional iteration metadata
        metadata[v]['attr_delta'] = attr_util_deltas[max_val_candidate_idx]
        metadata[v]['struct_delta'] = struct_util_deltas[max_val_candidate_idx]
        metadata[v]['cost_delta'] = cost_deltas[max_val_candidate_idx]

    return metadata

# Utility aggregation functions
def linear_util_agg(a, s, c, v, G):
    return a + s

def attr_first_agg(a, s, c, v, G):
    if v.data['total_attr_util'](v, G) == 1.0:
        return a + s
    return a

def struct_first_agg(a, s, c, v, G):
    if v.data['struct_util'](v, G) == 1.0:
        return a + s
    return s

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

#########################
# Measurement functions #
#########################

def random_walk_length(u, G):

    # Uses budget calculation to get length of walk that u should take on G
    walk_budget = remaining_budget(u, G)
    return math.floor(walk_budget / 0.1)

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
            nx_G.add_node(vtx,
                attr_util=attr_util,
                struct_util=struct_util,
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
