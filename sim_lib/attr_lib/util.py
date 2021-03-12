"""
Util functions used in attribute search network
"""
import math
from collections import defaultdict

import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm

import sim_lib.graph_networkx as gnx

##########################
# Edge utility functions #
##########################

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

def total_inv_likelihood(u, v, G):
    # Gives utility as the summed inverse likelihood of shared attributes
    # In discrete choice terms it is 1_u^T theta 1_v where theta is the
    # diagonal matrix with inverse attribute likelihoods

    sum_likelihood = 0
    for ctx, u_attrs in G.data[u].items():
        if ctx in G.data[v]:
            for attr in u_attrs.intersection(G.data[v][ctx]):
                sum_likelihood += 1 / discrete_pareto_pdf(attr)
    return sum_likelihood / G.sim_params['k']

def total_inv_frequency(u, v, G):
    # Sum of inverse frequency amongst all vertices

    sum_inv_freq = 0
    for ctx, u_attrs in G.data[u].items():
        if ctx in G.data[v]:
            for attr in u_attrs.intersection(G.data[v][ctx]):
                num_occurences = len([ vtx for vtx in G.vertices \
                        if ctx in G.data[vtx] and attr in G.data[vtx][ctx] ])
                sum_inv_freq += len(G.vertices) / num_occurences
    return sum_inv_freq

# Structural (normalized)
def neighborhood_density(v, G):
    # Density is already "normalized", clique is density 1
    if len(v.nbors) == 0:
        return 0
    if len(v.nbors) == 1:
        return 1
    nbor_edges = 0
    for u in v.nbors:
        for w in u.nbors:
            if v.is_nbor(w):
                nbor_edges += 1
    return nbor_edges / (len(v.nbors) * (len(v.nbors) - 1))

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

    max_ball_size = sum([ (G.sim_params['vtx_budget'] // u.data['direct_cost']) - 1 \
            for u in v.nbors ]) + len(v.nbors)

    return size / max_ball_size

##############################
# Edge probability functions #
##############################

def marginal_logistic(u, util, scale=2 ** -4):
    log_func = lambda x : (2 / (1 + np.exp(-1 * scale * x))) - 1
    return (log_func(u.total_edge_util + util) - log_func(u.total_edge_util)) ** 0.5

def logistic(u, util, scale=2 ** -4):
    log_func = lambda x : (2 / (1 + np.exp(-1 * scale * x))) - 1
    return (log_func(util)) ** 0.5

def const_one(u, util):
    return 1.0

##################
# Cost functions #
##################

def calc_cost(u, G):
    direct_cost = u.data['direct_cost']
    indirect_cost = u.data['indirect_cost']

    u_subgraph_degree = 0
    nbor_set = set(u.nbors)
    for v in nbor_set:
        u_subgraph_degree += len(nbor_set.intersection(set(v.nbors)))
    assert u_subgraph_degree % 2 == 0, 'sum of degrees must be even'
    edge_count = u_subgraph_degree / 2
    total_direct_cost = len(nbor_set) * direct_cost
    total_indirect_cost = edge_count * indirect_cost

    return (total_direct_cost + total_indirect_cost) / G.sim_params['vtx_budget']

def remaining_budget(u, G):
    u_cost = calc_cost(u, G)
    return 1 - u_cost

####################
# Edge calculation #
####################

def inv_util_edge_calc(G, edge_candidates):

    # Reset graph
    for v in G.vertices:
        for u in v.nbors:
            G.remove_edge(u, v)

    for u, v in edge_candidates:
        G.add_edge(u, v)

    for v in G.vertices:
        if remaining_budget(v, G) >= 0:
            continue
        inv_util_set = [ 1 / G.potential_utils[v.vnum][u.vnum] for u in v.nbors ]
        dropped_nbor = np.random.choice(v.nbors,
                p= [ iut / sum(inv_util_set) for iut in inv_util_set ])
        G.remove_edge(v, dropped_nbor)

def greedy_simul_set_proposal(G, edge_candidates, dunbar=150):

    # Each vertex greedily selects top edges within budget then keep checking if
    # all other vertices agree until everyone is satisfied
    # This assumes that there is no indirect cost

    max_degree = dunbar // max([ u.data['direct_cost'] for u in G.vertices ])

    # Reset graph
    for v in G.vertices:
        for u in v.nbors:
            G.remove_edge(u, v)

    # Put edge candidates into dict
    candidates = defaultdict(list)
    for u, v in edge_candidates:
        candidates[u].append(v)
        candidates[v].append(u)

    # Gets ordered candidates per vertex
    util_cands = {}
    for v in candidates:
        util_cands[v] = list(zip(candidates[v], \
                [ G.potential_utils[v.vnum][u.vnum] for u in candidates[v] ]))
        util_cands[v].sort(key=lambda k : k[1], reverse=True)
        nbor_candidates, ec_vals = zip(*util_cands[v])
        util_cands[v] = nbor_candidates

    # Loose upper bound on max number of iters possible
    for i in range(len(G.vertices) ** 2):
        proposals = { v : util_cands[v][:max_degree - v.degree] for v in util_cands }
        had_add_edge = False
        for v in proposals:
            for u in proposals:
                if u != v and v in proposals[u] and u.degree < max_degree:
                    G.add_edge(u, v)
                    had_add_edge = True
        if not had_add_edge:
            break

def greedy_simul_sequence_proposal(G, edge_candidates):
    
    # Each vertex constructs sequence of proposals up until constraints are meant
    # Sequence is ordered greedily

    # Reset graph
    for v in G.vertices:
        for u in v.nbors:
            G.remove_edge(u, v)

    candidates = defaultdict(list)
    for u, v in edge_candidates:
        candidates[u].append(v)
        candidates[v].append(u)

    sequences = defaultdict(list)
    for v in G.vertices:
        sequences[v] = list(zip(candidates[v], \
                [ G.potential_utils[v.vnum][u.vnum] for u in candidates[v] ]))
        sequences[v].sort(key=lambda k : k[1], reverse=True)
        sequences_vtx, vtx_values = zip(*sequences[v])
        sequences[v] = sequences_vtx

    for i in range(len(G.vertices)):
        # Check for mutuals up until index i
        had_addition = False
        proposals = { v : sequences[v][:i + 1] for v in sequences }
        for v in proposals:
            for u in proposals[v]:
                if not G.are_neighbors(u, v) and v in proposals[u]:
                    G.add_edge(u, v)
                    if remaining_budget(u, G) < 0:
                        G.remove_edge(u, v)
                    else:
                        had_addition = True
        if not had_addition:
            break

def iter_drop_max_objective(G, edge_proposals):
    # While over budget, drop edges that result in greatest net objective
    # After within budget, drop edges until reach local max objective

    obj = lambda v, G : G.sim_params['vtx_util'](v, v.total_edge_util) - calc_cost(v, G)

    def steepest_hill_iter(v, G):
        max_obj_val = obj(v, G) 
        max_obj_edge = None
        for u in v.nbors:
            G.remove_edge(v, u)
            current_obj = obj(v, G)
            G.add_edge(v, u)
            if current_obj > max_obj_val:
                max_obj_edge = u
                max_obj_val = current_obj
        if max_obj_edge is not None:
            G.remove_edge(v, max_obj_edge)

    # Assume all proposals are accepted
    for v in edge_proposals:
        for u in edge_proposals[v]:
            G.add_edge(u, v)

    # Resolve budget and do local optimization
    for _ in range(G.num_people):
        for v in G.vertices:
            if remaining_budget(v, G) < 0:
                max_obj_val = np.NINF
                max_obj_edge = None
                for u in v.nbors:
                    G.remove_edge(v, u)
                    current_obj = obj(v, G) 
                    G.add_edge(v, u)
                    if current_obj >= max_obj_val:
                        max_obj_val = current_obj
                        max_obj_edge = u
                G.remove_edge(v, max_obj_edge)
            else:
                steepest_hill_iter(v, G)

def seq_projection_single_selection(G, edge_proposals):
    # Sequentially (non-random) pick one edge to propose to via projection
    # of multiobjective optimization function
    # Assumes even split of coefficients

    print('-----------------------------------------')

    proposed_by = { v : [ u for u, u_props in edge_proposals.items() if v in u_props ] \
            for v in G.vertices }

    for v in G.vertices:
        attr_util_deltas = []
        struct_util_deltas = []
        cost_deltas = []

        if len(v.nbors) == 0 and len(proposed_by[v]) == 0:
            continue

        candidates = [ u for u in v.nbors ]

        cur_attr_util = v.total_edge_util
        cur_struct_util = v.data['struct_util'](v, G)
        cur_cost = calc_cost(v, G)

        for u in v.nbors:
            G.remove_edge(v, u)
            attr_util_deltas.append(v.total_edge_util - cur_attr_util)
            struct_util_deltas.append(v.data['struct_util'](v, G) - cur_struct_util)

            # Ordered so that reduction in cost is positive
            cost_deltas.append(cur_cost - calc_cost(v, G))
            G.add_edge(v, u)

        for u in proposed_by[v]:
            if u in candidates:
                continue

            G.add_edge(v, u)
            if remaining_budget(v, G) >= 0:
                attr_util_deltas.append(v.total_edge_util - cur_attr_util)
                struct_util_deltas.append(v.data['struct_util'](v, G) - cur_struct_util)
                cost_deltas.append(cur_cost - calc_cost(v, G))
                candidates.append(u)
            G.remove_edge(v, u)

        # Normalize attr util by maximum values in current t

        max_attr_util = G.sim_params['edge_util_func'](v, v, G)
        attr_util_deltas = [ aud / max_attr_util for aud in attr_util_deltas ]

        candidate_value_points = list(zip(attr_util_deltas, struct_util_deltas, cost_deltas))
        norm_values = [ sum([a, s, c]) for a, s, c in candidate_value_points ]
        max_val_candidate_idx = np.argmax(norm_values)
        max_val_candidate = candidates[ max_val_candidate_idx ]


        print(v, 'degree', v.degree)
        print(candidate_value_points)
        print([ sum([a, s, c]) for a, s, c in candidate_value_points ])
        print('chose', max_val_candidate_idx, candidate_value_points[max_val_candidate_idx])
        print('chosen vtx', max_val_candidate)
        if norm_values[max_val_candidate_idx] < 0:
            print('chose do nothing')
        elif max_val_candidate in v.nbors:
            print('chose to drop')
        else:
            print('chose to add')
        print("###########################")

        if norm_values[max_val_candidate_idx] < 0:
            continue
        elif max_val_candidate in v.nbors:
            G.remove_edge(v, max_val_candidate)
        else:
            G.add_edge(v, max_val_candidate)

# Non-neighbor and visited vertex set proposals

def indep_context_proposal(G, v, copy_attr=True):
    # Independently select a vertex in G that has a context shared with v
    # to be proposed to by v. Adds selected vertex to v's visited set
    # Select in proportion to context count (for trivial will usually be IID)
    v_context_counts = [ ( ctx, len(G.data[v][ctx]) ) for ctx in G.data[v] ]
    v_contexts, context_counts = zip(*v_context_counts)
    context_probs = [ cnt / sum(context_counts) for cnt in context_counts ]
    search_context = np.random.choice(v_contexts, p=context_probs)

    valid_vertices = [ u for u in G.vertices if \
            (search_context in G.data[u]) and u.vnum != v.vnum ]

    if len(valid_vertices) == 0:
        return

    proposed_vertex = np.random.choice(valid_vertices)
    v.data['visited'].add(proposed_vertex)

    if copy_attr:
        G.data[v] = G.sim_params['attr_copy'](v, proposed_vertex, G)

#########################
# Measurement functions #
#########################

def random_walk_length(u, G):

    # Uses budget calculation to get length of walk that u should take on G
    walk_budget = remaining_budget(u, G)
    return math.floor(walk_budget / 0.1)

###########################
# Attribute distributions #
###########################

def discrete_pareto_val(alpha=2, sigma=1):

    # From Buddana Kozubowski discrete Pareto
    # alpha "shape" sigma "size"
    std_exp_val  = np.random.exponential(scale=1.0)
    gamma_val = np.random.gamma(alpha, scale=sigma)
    return np.ceil(std_exp_val / gamma_val) - 1.0

#####################
# Attribute copying #
#####################

def indep_attr_copy(u, v, G):
    # Copy an attribute from v to u by random selection

    v_contexts = list(G.data[v].keys())
    v_context_sizes = [ len(G.data[v][vctxt]) for vctxt in v_contexts ]
    v_context = np.random.choice(v_contexts,
            p=[ csize / sum(v_context_sizes) for csize in v_context_sizes ])
    v_attr = np.random.choice(list(G.data[v][v_context]))

    u_context_map = G.data[u].copy()
    if v_context in u_context_map:
        u_context_map[v_context].add(v_attr)
        return u_context_map
    else: # Case where context may be switched
        u_context_map[v_context] = { v_attr }
        u_contexts = list(u_context_map.keys())
        u_context_sizes = [ len(u_context_map[uctxt]) for uctxt in u_contexts ]
        context_count = G.sim_params['k']
        u_context_set = np.random.choice(u_contexts,
                size=context_count, replace=False,
                p=[ csize / sum(u_context_sizes) for csize in u_context_sizes ])
        u_context_map = { ctxt : u_context_map[ctxt] for ctxt in u_context_set }

    return u_context_map

def freq_attr_copy(u, v, G):
    # Supposes specific strategy: copy attributes that your neighbors have
    # Weigh by increase to utility if you take that attribute

    u_context_map = G.data[u]

    def shared_attr_util(u, v, G, ctxt, attr):
        assert attr not in u_context_map[ctxt], 'Func only used when attr not present'

        pre_add_util = sum([ G.sim_params['edge_util_func'](u, w, G) for w in u.nbors ])
        u_context_map[ctxt].add(attr)
        post_add_util = sum([ G.sim_params['edge_util_func'](u, w, G) for w in u.nbors ])
        u_context_map[ctxt].remove(attr)
        return post_add_util - pre_add_util

    def diff_attr_util(u, v, G, v_ctxt, v_attr, u_ctxt):
        assert v_ctxt not in u_context_map, 'Only used when v_ctxt not in u'
        assert u_ctxt in u_context_map, 'u_ctxt must be in u'

        pre_add_util = sum([ G.sim_params['edge_util_func'](u, w, G) for w in u.nbors ])
        u_ctxt_attrs = u_context_map.pop(u_ctxt)
        u_context_map[v_ctxt] = { v_attr }
        post_add_util = sum([ G.sim_params['edge_util_func'](u, w, G) for w in u.nbors ])
        u_context_map.pop(v_ctxt)
        u_context_map[u_ctxt] = u_ctxt_attrs
        return post_add_util - pre_add_util

    u_contexts = set(u_context_map.keys())
    v_contexts = set(G.data[v].keys())

    sum_gain = 0

    # Get common context attributes
    # These are monotonic
    shared_ctxt_attributes = { ctxt : { attr : 0 for attr in G.data[v][ctxt] } \
            for ctxt in v_contexts.intersection(u_contexts) }
    for ctxt, attr_map in shared_ctxt_attributes.items():
        for attr in attr_map.keys():
            if attr in u_context_map[ctxt]:
                continue
            attr_map[attr] = shared_attr_util(u, v, G, ctxt, attr)
            sum_gain += attr_map[attr]

    # Get new context attributes
    # These may not be monotonic so we must drop options here
    diff_ctxt_attributes = { ctxt : { attr : 0 for attr in G.data[v][ctxt] } \
            for ctxt in v_contexts.difference(u_contexts) }
    for v_ctxt, attr_map in diff_ctxt_attributes.items():
        attr_map_drop = []
        for v_attr in attr_map.keys():
            max_gain_u_ctxt = None
            max_gain = -1
            for u_ctxt in u_contexts:
                drop_gain = diff_attr_util(u, v, G, v_ctxt, v_attr, u_ctxt)
                if max_gain_u_ctxt == None or drop_gain > max_gain:
                    max_gain_u_ctxt = u_ctxt
                    max_gain = drop_gain
            if max_gain < 0:
                attr_map_drop.append(v_attr)
            else:
                attr_map[v_attr] = (max_gain, max_gain_u_ctxt)
                sum_gain += max_gain
        for attr in attr_map_drop:
            attr_map.pop(attr)

    if sum_gain == 0:
        return u_context_map

    # Select an attribute based on largest gain
    attribute_candidates = {}
    for ctxt, attr_map in shared_ctxt_attributes.items():
        for attr, gain in attr_map.items():
            attribute_candidates[(ctxt, attr)] = gain / sum_gain
    for ctxt, attr_map in diff_ctxt_attributes.items():
        for attr, (gain, u_ctxt) in attr_map.items():
            attribute_candidates[(ctxt, attr)] = gain / sum_gain
    ctxt_attr_pairs = list(attribute_candidates.keys())
    attr_gains = [ attribute_candidates[ca] for ca in ctxt_attr_pairs ]
    chosen_idx = np.random.choice(len(ctxt_attr_pairs), p=attr_gains)
    chosen_ctxt, chosen_attr = ctxt_attr_pairs[chosen_idx]

    if chosen_ctxt in diff_ctxt_attributes:
        _, drop_context = diff_ctxt_attributes[chosen_ctxt][chosen_attr]
        u_context_map.pop(drop_context)
        u_context_map[chosen_ctxt] = { chosen_attr }
    else:
        assert chosen_ctxt in u_context_map, 'chosen ctxt must be in u_context_map'
        u_context_map[chosen_ctxt].add(chosen_attr)

    return u_context_map

