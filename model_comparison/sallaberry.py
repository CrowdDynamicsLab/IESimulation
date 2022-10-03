import math

import numpy as np
import networkx as nx

class Tree:
    def __init__(self, T_mat, degs):
        self.mat = T_mat
        self.degs = degs

def gen_G(k, min_size, max_size):

    # Returns G with k cliques of random size between min and max
    C = []
    sizes = np.random.randint(min_size, max_size + 1, size=k)
    N = np.sum(sizes)
    G = np.zeros((N, N))
    v_cnt = 0
    for c_i in range(k):
        c_i_size = sizes[c_i]
        for i in range(v_cnt, v_cnt + c_i_size):
            for j in range(i + 1, v_cnt + c_i_size):
                G[i][j] = 1
                G[j][i] = 1
        C.append(list(range(v_cnt, v_cnt + c_i_size)))
        v_cnt += c_i_size
    return G, C

def sf_assoc(G):

    # Generate a BA network with roughly double number of edges (from paper ex)
    tot_deg = np.sum(G)
    n = len(G)

    # want m s.t. nm - m^2 = tot_deg
    m = min(np.roots([-1, n, -1 * tot_deg]))

    # Overestimate
    m = math.ceil(m)

    ba = nx.barabasi_albert_graph(n, m)

    sfDeg = {}
    G_vtx = list(range(n))
    Gpr_vtx = list(range(n))
    for _ in range(n):
        v_idx = np.random.randint(0, len(G_vtx))
        vpr_idx = np.random.randint(0, len(Gpr_vtx))
        v = G_vtx.pop(v_idx)
        vpr = Gpr_vtx.pop(vpr_idx)
        
        if np.sum(G[v]) < ba.degree[vpr]:
            sfDeg[v] = ba.degree[vpr]
        else:
            sfDeg[v] = np.sum(G[v])

    return sfDeg

def gen_T(k):

    mean_deg = math.ceil(math.log(k))

    T_mat = [ [ [i] for i in range(k) ] ]
    deg = [ [ 1 for i in range(k) ] ]
    while True:
        splits = []
        cur_split = []
        degs = []
        to_split = len(T_mat[-1])
        if to_split == 1:
            break
        elif to_split == 2:
            T_mat.append([T_mat[-1][0] + T_mat[-1][1]])
            deg.append([2])
            break
        last_split = -1
        for i in range(to_split):
            cur_split.extend(T_mat[-1][i])
            rval = np.random.random()
            if rval <= (1 / mean_deg):
                splits.append(cur_split)
                cur_split = []
                degs.append(i - last_split)
                last_split = i

        if len(cur_split) > 0:
            splits.append(cur_split)
            degs.append(to_split - 1 - last_split)

        if len(T_mat[-1]) != sum(degs):
            print('ERROR')
            print('T mat')
            print(T_mat)
            print('degs')
            print(degs)
        assert len(T_mat[-1]) == sum(degs), 'split degree not equal to number of elements!'

        T_mat.append(splits)
        deg.append(degs)

    # low prob but remove duplicates
    dups = []
    for l in range(len(T_mat) - 2, 0, -1):
        if T_mat[l] == T_mat[l + 1]:
            dups.append(l)
    for l in dups:
        T_mat.pop(l)
        deg.pop(l)

    return Tree(T_mat, deg)

def P_ij(i, j, T):
    d = len(T.mat)
    n = len(T.mat[0])

    # Start at root, find first split with i and j 
    conn_lvl = -1
    conn_idx = -1
    for l in range(d - 1, 0, -1):
        for si, split in enumerate(T.mat[l]):
            if i in split and j in split:
                conn_lvl = l
                conn_idx = si

    # Get probability of going to connection split
    p = 1
    for l in range(1, conn_lvl):
        for si, split in enumerate(T.mat[l]):
            if i in split:
                p *= 1 / T.degs[l][si]
                break

    # Probability at connection inner node
    if conn_lvl == d - 1:
        p *= 1 / (T.degs[conn_lvl][conn_idx] - 1)
    else:
        p *= 1 / T.degs[conn_lvl][conn_idx]

    # Traverse down to j
    for l in range(conn_lvl - 1, 0, -1):
        for si, split in enumerate(T.mat[l]):
            if j in split:
                p *= 1 / T.degs[l][si]
                break

    return p

def node_merges(v, sfDeg, avgDeg):
    return math.floor(sfDeg[v] / avgDeg)

def clique_merges(c, sfDeg, avgDeg):
    return sum([ node_merges(v, sfDeg, avgDeg) for v in c ])

def merge(c_a, c_b, sfDeg, mc):

    # Pick u
    c_a_choices = [ e for e in c_a if mc[e] > 0 ]
    if len(c_a_choices) == 0:
        return None
    u = np.random.choice(c_a_choices)
    mc[u] -= 1

    # Pick v
    tot_wgt = sum([ sfDeg[e] for e in c_b ])
    c_b_wgt = [ sfDeg[e] / tot_wgt for e in c_b ]
    v = np.random.choice(c_b, p=c_b_wgt)

    return u, v

def get_clusters(v, mw):
    all_merges = set()
    visited = set()
    return get_merges(v, mw, all_merges, visited)

def get_merges(v, mw, am, visited):
    if v in visited:
        return set()

    visited.add(v)
    for u in mw[v]:
        am = am.union(get_merges(u, mw, am, visited))
        am.add(u)
    return am

def run_model(k, min_size, max_size):
    G, C = gen_G(k, min_size, max_size)
    sfDeg = sf_assoc(G)

    n = len(G)

    avgDeg = np.mean(np.sum(G, axis=0))

    T = gen_T(k)

    # Randomly assign cluster indices to tree leaf indices
    perm = np.random.permutation(k)
    t_c_map = { i : c for i, c in enumerate(perm) }
    c_t_map = { }
    for t, c in t_c_map.items():
        c_t_map[c] = t

    merge_counts = { v : node_merges(v, sfDeg, avgDeg) for v in range(n) }

    # Run clique merge
    merges = set()
    merges_with = { v : set() for v in range(n) }
    for a, c_a in enumerate(C):
        for b, c_b in enumerate(C):
            if a == b:
                continue

            # Pairwise_Merges
            cmerges = clique_merges(c_a, sfDeg, avgDeg)
            p_wgt = P_ij(c_t_map[a], c_t_map[b], T)
            num_merges = math.floor(cmerges * p_wgt)
            for _ in range(num_merges):
                merge_pair = merge(c_a, c_b, sfDeg, merge_counts)
                if merge_pair is not None:
                    u, v = merge_pair
                    merges.add(frozenset(merge_pair))
                    merges_with[u].add(v)
                    merges_with[v].add(u)

    # If u merges with v and v merges with w then uvw are all merged
    vtx_clusters = []
    for v in range(n):
        skip = False
        for vc in vtx_clusters:
            if v in vc:
                skip = True
                break
        if skip:
            continue
        cluster = get_clusters(v, merges_with)
        cluster.add(v)
        vtx_clusters.append(cluster)
    assert len(set().union(*vtx_clusters)) == n, 'all vertices should be accounted for in vertex cluster sets'

    # Take boolean sum of rows in each cluster, return as final G
    npr = len(vtx_clusters)
    cl_mat = np.zeros((npr, npr))
    for vci, vc in enumerate(vtx_clusters):
        #print(vc)
        uci = vci + 1
        for uc in vtx_clusters[vci + 1:]:
            #print(uc)
            if cl_mat[vci][uci] == 1:
                continue
            for v in vc:
                v_nbors = np.where(G[v] != 0)[0]
                for u in uc:
                    if u in v_nbors:
                        print(vci, uci)
                        cl_mat[vci][uci] = 1
                        cl_mat[uci][vci] = 1
            uci += 1
    return cl_mat

run_model(8, 2, 5)
