"""
Util functions used in attribute search network
"""
import math
from collections import defaultdict

import numpy as np
from scipy.sparse import linalg as scp_sla
import networkx as nx
import networkx.algorithms.community as nx_comm

import sim_lib.graph_networkx as gnx

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

def gen_schelling_seg_funcs(frac):
    
    # Generates a homphily function and a heterophily function where homophily
    # desires `frac` proportion neighbors to be similar
    def schelling_homophily(u, G):
        if u.degree == 0:
            return 0.0
        overall_similarity = u.sum_edge_util / u.degree
        if overall_similarity < frac:
            return 0.0
        return overall_similarity

    def schelling_heterophily(u, G):
        if u.degree == 0:
            return 0.0
        overall_dissimilarity = u.sum_edge_util / u.degree
        if overall_dissimilarity < frac:
            return 0.0
        return overall_dissimilarity

    return schelling_homophily, schelling_heterophily

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
    v_nbors = v.nbors
    nborhoods = []
    for u in v_nbors:
        nborhoods.append(set(u.nbors))
    nbor_intersect = len(set(v_nbors).intersection(*nborhoods))
    nbor_union = len(set(v_nbors).union(*nborhoods))
    if nbor_union + nbor_intersect == 0:
        return 0
    return nbor_intersect / nbor_union

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
    total_indirect_cost = u.nborhood_degree / 2 * indirect_cost

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
    return seq_projection_edge_edit(G, edge_proposals, substitute=False, log=log)

def seq_edge_sel_logged(G, edge_proposals, substitute=True, allow_early_drop=False):
    return seq_projection_edge_edit(G, edge_proposals, substitute, allow_early_drop, True)

def seq_edge_sel_silent(G, edge_proposals, substitute=True, allow_early_drop=False):
    return seq_projection_edge_edit(G, edge_proposals, substitute, allow_early_drop, False)

def seq_projection_edge_edit(G, edge_proposals, substitute=True, allow_early_drop=True, log=False):
    # Sequentially (non-random) pick one edge to propose to via projection
    # of multiobjective optimization function
    # Assumes even split of coefficients

    if log:
        print('-----------------------------------------')
        #NOTE: Consider adding back once not global
        #print('proposals:', edge_proposals)

    util_agg = lambda a, s, c: a 

    proposed_by = { v : [ u for u, u_props in edge_proposals.items() if v in u_props ] \
            for v in G.vertices }

    for v in G.vertices:
        attr_util_deltas = []
        struct_util_deltas = []
        cost_deltas = []

        if len(v.nbors) == 0 and len(proposed_by[v]) == 0:
            continue

        candidates = []

        cur_attr_util = v.data['total_attr_util'](v, G)
        cur_struct_util = v.data['struct_util'](v, G)
        cur_cost = calc_cost(v, G)

        can_add_nbor = remaining_budget(v, G) >= G.sim_params['direct_cost']
        for u in v.nbors:
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
            if u in v.nbors:
                continue

            G.add_edge(v, u)

            # Would have budget to add (remaining_budget assumes add here)
            if remaining_budget(v, G) >= 0:
                attr_util_deltas.append(v.data['total_attr_util'](v, G) - cur_attr_util)
                struct_util_deltas.append(v.data['struct_util'](v, G) - cur_struct_util)
                cost_deltas.append(cur_cost - calc_cost(v, G))
                candidates.append(u)

            G.remove_edge(v, u)

            if (not allow_early_drop and can_add_nbor) or not substitute:

                # If can still add do not allow a substitution
                # Edge case when no available vertex at direct cost
                continue

            for w in v.nbors:
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

        if len(candidates) == 0:
            if log:
                print(v, 'degree', v.degree, 'budget', calc_cost(v, G))
                print(v, 'had no options')
            print("###########################")
            continue

        #TODO: Attribute normalization! 

        candidate_value_points = list(zip(attr_util_deltas, struct_util_deltas, cost_deltas))
        #TODO: Change after understanding struct/attr utility alone
        norm_values = [ util_agg(a, s, c) for a, s, c in candidate_value_points ]
        max_val_candidate_idx = np.argmax(norm_values)
        max_val_candidate = candidates[ max_val_candidate_idx ]

        if log:
            print(v, 'degree', v.degree, 'budget', calc_cost(v, G))
            print(candidate_value_points)
            print([ s + c for a, s, c in candidate_value_points ])
            print(candidates)
            print('chose', max_val_candidate_idx, candidate_value_points[max_val_candidate_idx], norm_values[max_val_candidate_idx])
            print('chosen vtx', max_val_candidate)
            if remaining_budget(v, G) < 0:
                print('has to resolve budget via subset drop')
            elif remaining_budget(v, G) >= 0 and norm_values[max_val_candidate_idx] <= 0:
                print('chose do nothing')
            elif type(max_val_candidate) == tuple:
                u, w = max_val_candidate
                print('chose substitute', w, ' for ', u)
            elif max_val_candidate in v.nbors:
                print('chose to early drop')
            else:
                print('chose to add')
            print("###########################")

        if remaining_budget(v, G) < 0:
            subset_budget_resolution(v, G, util_agg)
        elif remaining_budget(v, G) >= 0 and norm_values[max_val_candidate_idx] <= 0:
            continue
        elif type(max_val_candidate) == tuple:
            add_vtx, rem_vtx = max_val_candidate
            G.add_edge(v, add_vtx)
            G.remove_edge(v, rem_vtx)
        elif allow_early_drop and max_val_candidate in v.nbors:
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

