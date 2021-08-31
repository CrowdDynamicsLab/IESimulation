"""
Util functions used in attribute search network
"""
import math
from collections import defaultdict

import numpy as np
from scipy.sparse import linalg as scp_sla
import networkx as nx

from tabulate import tabulate  

#######################
# Init attr functions #
#######################

def init_indicator_attrs(v, G):
    total_ctxts = list(range(G.sim_params['context_count']))
    ctxts = np.random.choice(total_ctxts, size=G.sim_params['k'])
    return { ctxt : { 0 } for ctxt in ctxts }

def init_binary_homophily(v, G):
    req = G.sim_params['context_count'] == 2 and G.sim_params['k'] == 1
    msg = "Must have 2 total contexts and 1 allowed context"
    assert req, msg
    return { 0 : { 1 } }

def init_binary_heterophily(v, G):
    req = G.sim_params['context_count'] == 2 and G.sim_params['k'] == 1
    msg = "Must have 2 total contexts and 1 allowed context"
    assert req, msg
    return { 1 : { 1 } }

def init_cont_homophily(v, G):
    req = G.sim_params['context_count'] == 2 and G.sim_params['k'] == 1
    msg = "Must have 2 total contexts and 1 allowed context"
    assert req, msg
    return { 0 : { np.random.random() } }

def init_cont_heterophily(v, G):
    req = G.sim_params['context_count'] == 2 and G.sim_params['k'] == 1
    msg = "Must have 2 total contexts and 1 allowed context"
    assert req, msg
    return { 1 : { np.random.random() } }

##########################
# Attr utility functions #
##########################

# Utility directional u => v
def binary_homophily(u, v, G):
    assert 0 in G.data[u], 'u must have context 0 (homophily) by assumption'
    if 0 in G.data[v]:
        return 1 if G.data[u] == G.data[v] else 0
    return 0

def binary_heterophily(u, v, G):
    assert 1 in G.data[u], 'u must have context 1 (heterophily) by assumption'
    return 1 if 0 in G.data[v] else 0

def cont_homophily(u, v, G):
    assert 0 in G.data[u], 'u must have context 0 (homophily) by assumption'
    if 0 in G.data[u] and 0 in G.data[v]:
        u_val = -1
        v_val = -1
        for val in G.data[u][0]:
            u_val = val
            break
        for val in G.data[v][0]:
            v_val = val
            break
        return 1 - abs(u_val - v_val)
    return 0.0

def cont_heterophily(u, v, G):
    assert 1 in G.data[u], 'u must have context 1 (heterophily) by assumption'
    if 1 in G.data[u] and 0 in G.data[v]:
        u_val = -1
        v_val = -1
        for val in G.data[u][1]:
            u_val = val
            break
        for val in G.data[v][0]:
            v_val = val
            break
        return 1 - abs(u_val - v_val)
    return 0.0

def gen_similarity_funcs():
    def similarity_score(u, v, G):
        similar_ctxts = 0
        for ctxt in G.data[u]:
            if ctxt in G.data[v]:
                similar_ctxts += 1
        return similar_ctxts / G.sim_params['k']

    def homophily(u, v, G):
        return similarity_score(u, v, G)

    def heterophily(u, v, G):
        return 1 - similarity_score(u, v, G)

    return homophily, heterophily

def gen_schelling_seg_funcs(thresh, calc_method='satisfice'):
    
    # Generates a homphily function and a heterophily function where homophily
    # desires roughly `thresh` proportion neighbors to be similar

    def schelling_balance(u, G):
        if u.degree == 0:
            return 0.0
        
        if calc_method == 'proportion':
            return 1 - abs((u.sum_edge_util / u.degree) - thresh)
        elif calc_method == 'buckets':
            if thresh == 0.0:
                return (u.degree - u.sum_edge_util) / (G.sim_params['max_degree'] * (1 - thresh))
            elif thresh == 1.0:
                return u.sum_edge_util / (G.sim_params['max_degree'] * thresh)
            strat_term = u.sum_edge_util / (G.sim_params['max_degree'] * thresh)
            rem_term = (u.degree - u.sum_edge_util) / (G.sim_params['max_degree'] * (1 - thresh))
            return strat_term + rem_term
        elif calc_method == 'satisfice':
            prop = u.sum_edge_util / u.degree
            if prop >= thresh:
                return 1.0
            return prop
        raise ValueError(f"calc_method '{calc_method}' argument value is invalid, check spelling")

    return schelling_balance, schelling_balance

def exp_surprise(u, v, G):
    total_surprise = 0
    matches = []
    for ctx in G.data[u]:
        if ctx in G.data[v]:
            total_surprise += np.log2(u.data[ctx] * v.data[ctx])
            matches.append(ctx)
    return 1 - 2 ** (total_surprise)

def simple_sigmoid(u, v, G, scale=1.0):

    # Simple sigmoid, |X| / (1 + |X|) where X is intersecting
    # contexts. Does not account for rarity of context
    match_count = 0
    for ctx, attrs in G.data[u].items():
        if ctx in G.data[v]:
            match_count += len(attrs.intersection(G.data[v][ctx]))
    return match_count / (scale + match_count)

def discrete_pareto_pdf(k, alpha=2, sigma=1):
    
    # PDF of discrete Pareto distribution (Pareto II)
    k_1 = ( 1 / ( 1 + k / sigma) ) ** alpha
    k_2 = ( 1 / ( 1 + ( (k + 1) / sigma ) ) ) ** alpha
    return k_1 - k_2

###########################
# Structural (normalized) #
###########################

def direct_util_buffer(ut_func):
    # TODO: Figure out how to resolve this
    norm = 1
    def ut_func_wrapper(v, G):
        struct_ut = ut_func(v, G)
        epsilon = 2 ** -10 # Pretty arbitrary choice
        direct_util = G.sim_params['direct_cost'] + epsilon
        return (struct_ut + (v.degree * direct_util)) / norm
    return ut_func_wrapper

def neighborhood_density(v, G):
    # Density is already "normalized", clique is density 1
    if len(v.nbors) == 0:
        return 0

    nbor_edges = 0
    for u in v.nbors:
        for w in u.nbors:
            if v.is_nbor(w):
                nbor_edges += 1
    return (nbor_edges + (len(v.nbors) * 2)) / (len(v.nbors) * (len(v.nbors) + 1))

def potential_density(v, G):

    # Actually degree in the end
    nbor_edges = 0
    for u in v.nbors:
        for w in u.nbors:
            if v.is_nbor(w):
                nbor_edges += 1
    max_clique = G.sim_params['max_clique_size']
    potential_degree = max_clique * (max_clique - 1)
    return nbor_edges / potential_degree

def average_neighborhood_overlap(v, G):
    if v.degree == 0:
        return 0.0

    v_nbors = v.nbor_set
    nbor_overlaps = []
    for u in v_nbors:
        u_nbors = set(u.nbors)
        shared = len(u_nbors.intersection(v_nbors))
        total_nbors = len(u_nbors) + len(v_nbors) - shared - 2
        if shared + total_nbors == 0:
            nbor_overlaps.append(0.0)
        else:
            nbor_overlaps.append(shared / total_nbors)
    return sum(nbor_overlaps) / len(v_nbors)

def ball2_size(v, G):
    if len(v.nbors) == 0:
        return 0

    visited = set()
    queue = list(zip(v.nbors, [ 1 ] * len(v.nbors)))
    for u, depth in queue:
        if u in visited:
            continue

        visited.add(u)

        if depth == 2:
            continue

        for u_prime in u.nbors:
            queue.append((u_prime, depth + 1))
    size = len(visited)

    max_degree = math.floor(1 / G.sim_params['direct_cost'])
    max_ball_size = min(G.num_people, max_degree * (max_degree - 1) + max_degree)

    return size / max_ball_size

def degree_util(v, G):
    if v.degree == 0:
        return 0
    
    # Gives utility based on degree normalized by potential by cost
    max_degree = min(math.floor( 1 / G.sim_params['direct_cost'] ), G.num_people)
    util_degree = v.degree / max_degree
    return util_degree + v.degree * (2 ** -10)

##################
# Cost functions #
##################

def calc_cost(u, G):
    direct_cost = G.sim_params['direct_cost']
    indirect_cost = G.sim_params['indirect_cost']

    total_direct_cost = u.degree * direct_cost
    total_indirect_cost = (u.nborhood_degree / 2) * indirect_cost

    return total_direct_cost + total_indirect_cost

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
    cur_util = util_agg(cur_attr_util, cur_struct_util, cur_cost)
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
            potential_util = util_agg(pot_attr_util, pot_struct_util, pot_cost)
            util_loss = cur_util - potential_util
            G.add_edge(v, nbor)
            if util_loss < min_util_loss:
                min_util_loss = util_loss
                drop_candidate = nbor
        G.remove_edge(v, drop_candidate)
        cur_util -= min_util_loss

# Seq selection functions
def seq_projection_single_selection(G, edge_proposals, log):
    return seq_projection_edge_edit(G, edge_proposals, substitute=_allow_sub, log=log)

def seq_edge_sel_logged(G, edge_proposals):
    return seq_projection_edge_edit(G, edge_proposals, log=True)

def seq_edge_sel_silent(G, edge_proposals):
    return seq_projection_edge_edit(G, edge_proposals, log=False)

def seq_projection_edge_edit(G, edge_proposals, substitute=True, allow_early_drop=True, log=False):
    # Sequentially (non-random) pick one edge to propose to via projection
    # of multiobjective optimization function
    # Assumes even split of coefficients

    #if log:
        #NOTE: Consider adding back once not global
        #print('proposals:', edge_proposals)

    util_agg = G.sim_params['util_agg'] 

    proposed_by = { v : [ u for u, u_props in edge_proposals.items() if v in u_props ] \
            for v in G.vertices }

    # Prepare metadata collection for analysis
    metadata = { }

    for v in G.vertices:
        metadata[v] = { }
        metadata[v]['num_proposals'] = 0

        attr_util_deltas = []
        struct_util_deltas = []
        cost_deltas = []

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

        candidates = []

        can_add_nbor = remaining_budget(v, G) >= G.sim_params['direct_cost']
        for u in v.nbors:

            # Consider drop choices
            if not allow_early_drop and can_add_nbor:

                # Disallow early drops
                break

            G.remove_edge(v, u)
            attr_util_deltas.append(v.data['total_attr_util'](v, G) - cur_attr_util)
            struct_util_deltas.append(v.data['struct_util'](v, G) - cur_struct_util)

            # Ordered so that reduction in cost is positive
            cost_deltas.append(cur_cost - calc_cost(v, G))
            G.add_edge(v, u)

            candidates.append(u)

        for u in proposed_by[v]:

            # Consider edge formation choices
            if u in v.nbors:
                continue

            G.add_edge(v, u)

            # Would have budget to add (remaining_budget assumes add here)
            if remaining_budget(v, G) >= 0:
                attr_util_deltas.append(v.data['total_attr_util'](v, G) - cur_attr_util)
                struct_util_deltas.append(v.data['struct_util'](v, G) - cur_struct_util)
                cost_deltas.append(cur_cost - calc_cost(v, G))
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

                G.remove_edge(v, w)
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
        norm_values = [ util_agg(a, s, c) for a, s, c in candidate_value_points ]
        max_val_candidate_idx = np.argmax(norm_values)
        max_val_candidate = candidates[ max_val_candidate_idx ]

        if log:
            print('\n-------------------------------------------------------------------')
            header = np.array([('Vertex', 'Degree','Budget Used'), (str(v.vnum), str(v.degree), str(calc_cost(v,G)))]) 
            print(tabulate(header, headers="firstrow"))
            data = np.array([candidates, candidate_value_rounded]).T
            print('\n', tabulate(data, headers=['Candidates', '(Attr Util, Struct Util, Cost)']))
            #print([ s + c for a, s, c in candidate_value_points ])
            print('\nmax:', max_val_candidate)
            if remaining_budget(v, G) >= 0 and norm_values[max_val_candidate_idx] < 0:
                print('chose do nothing')
            elif type(max_val_candidate) == tuple:
                u, w = max_val_candidate
                print('chose to substitute', w, 'for', u)
            elif max_val_candidate in v.nbors:
                print('chose to early drop', max_val_candidate)
            else:
                print('chose to add', max_val_candidate)

        if remaining_budget(v, G) >= 0 and norm_values[max_val_candidate_idx] < 0:

            # No non-negative change candidates
            metadata[v]['action'] = 'none'
            metadata[v]['attr_delta'] = 0
            metadata[v]['struct_delta'] = 0
            metadata[v]['cost_delta'] = 0
            continue
        elif type(max_val_candidate) == tuple:

            # Substitution
            metadata[v]['action'] = 'substitution'
            add_vtx, rem_vtx = max_val_candidate
            G.add_edge(v, add_vtx)
            G.remove_edge(v, rem_vtx)
        elif allow_early_drop and max_val_candidate in v.nbors:

            # Early single drop
            metadata[v]['action'] = 'drop'
            G.remove_edge(v, max_val_candidate)
        else:

            # Edge formation
            metadata[v]['action'] = 'addition'
            G.add_edge(v, max_val_candidate)

        # Add additional iteration metadata
        metadata[v]['attr_delta'] = attr_util_deltas[max_val_candidate_idx]
        metadata[v]['struct_delta'] = struct_util_deltas[max_val_candidate_idx]
        metadata[v]['cost_delta'] = cost_deltas[max_val_candidate_idx]

    return metadata

# Revelation proposal sets 

def indep_revelation(G):
    edge_proposals = {}
    vtx_set = set(G.vertices)
    for v in vtx_set:
        non_nbors = vtx_set.difference(v.nbor_set.union({ v }))
        chosen_vtx = np.random.choice(list(non_nbors))
        edge_proposals[v] = [ chosen_vtx ]
    return edge_proposals

def non_ball2_revelation(G):

    # Add a vertex from V \ Ball_2(v) to v's proposals
    edge_proposals = {}
    vtx_set = set(G.vertices)
    for v in vtx_set:
        edge_proposals[v] = []
        ball_2 = v.nbor_set.union(*[ u.nbor_set for u in v.nbors ])
        ball_2.add(v)
        indep_choice_set = vtx_set.difference(ball_2)
        if len(indep_choice_set) == 0:
            continue
        chosen_vtx = np.random.choice(list(indep_choice_set))
        edge_proposals[v] = [ chosen_vtx ]
    return edge_proposals

def resistance_distance_revelation(G, dc_teleport_chance=0.2):

    # Reveals a vertex from V_v \ Ball_2(v) in component of v based on expected hitting time
    # dc_teleport_chance is sum likelihood of revealing vertex outside of component of v
    # except when size of said component is 0. Then this is equivalent to independent selection.

    edge_proposals = {}

    G_nx = graph_to_nx(G)
    G_nx_components = [ G_nx.subgraph(cc).copy() for cc in \
            nx.algorithms.components.connected_components(G_nx) ]
    vtx_set = set(G.vertices)
    for v in vtx_set:
        edge_proposals[v] = []
        ball_2 = v.nbor_set.union(*[ u.nbor_set for u in v.nbors ])
        ball_2.add(v)
        v_cc = next(cc for cc in G_nx_components if v in cc.nodes)

        if len(v_cc.nodes) == len(ball_2):

            # Case when connected component is contained in Ball2(v)
            indep_choice_set = vtx_set.difference(ball_2)
            if len(indep_choice_set) == 0:
                continue
            edge_proposals[v] = [ np.random.choice(list(indep_choice_set)) ]
        else:
            v_cc_node_set = set(v_cc.nodes)

            if len(G_nx_components) > 1 and np.random.random() <= dc_teleport_chance:

                # Only applicable when multiple components
                non_comp_choices = vtx_set.difference(v_cc_node_set)
                edge_proposals[v] = [ np.random.choice(list(non_comp_choices)) ]
                continue

            comp_non_ball2 = list(v_cc_node_set.difference(ball_2))
            comp_rd = [ nx.algorithms.distance_measures.resistance_distance(v_cc, v, c_v)
                    for c_v in comp_non_ball2 ]
            comp_rd_inv = [ 1 / (rd ** 2) for rd in comp_rd ]
            comp_rd_inv_sum = sum(comp_rd_inv)
            comp_rd_inv_prob = [ rdi / comp_rd_inv_sum for rdi in comp_rd_inv ]
            edge_proposals[v] = [ np.random.choice(comp_non_ball2, p=comp_rd_inv_prob) ]
    return edge_proposals

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

def graph_to_nx(G):
    """
    Converts a graph from sim_lib.graph to a networkx graph (undirected)
    """

    nx_G = nx.Graph()

    for vtx in G.vertices:
        attr_util = vtx.data['total_attr_util'](vtx, G)
        struct_util = vtx.data['struct_util'](vtx, G)
        cost = calc_cost(vtx, G)
        color = vtx.data['color']
        nx_G.add_node(vtx, attr_util=attr_util, struct_util=struct_util, cost=cost, color=color)
        

    for vtx in G.vertices:
        for nbor in vtx.nbors:
            util = vtx.edges[nbor].util
            nx_G.add_edge(vtx, nbor, capacity=1.0, util=util)

    return nx_G


