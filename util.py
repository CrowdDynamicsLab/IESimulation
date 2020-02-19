import numpy as np
from math import ceil, factorial

def gen_const_ratings(provs):
    """
    Returns a dict of ratings of provider : int
    representing ratings of each respective provider
    """
    ratings = np.linspace(0, 1, len(provs))
    #np.random.shuffle(ratings)
    return { provs[didx] : ratings[didx] for didx in range(len(provs)) }

def is_connected(G):
    """
    Runs BFS to check if G is connected
    """
    found = set()
    queue = [G.vertices[0]]
    found.add(G.vertices[0])

    while queue:
        cur_vtx = queue[0]
        queue.pop(0)

        for nbor in cur_vtx.nbors:
            if nbor not in found:
                queue.append(nbor)
                found.add(nbor)

    if len(found) != len(G.vertices):
        return False
    return True

def calc_diameter(G):
    """
    Runs floyd warshall and gets diam
    """

    n = len(G.vertices)
    dist = np.full((n,n), np.infty)

    for vtx in G.vertices:
        dist[vtx.vnum][vtx.vnum] = 0
        for nbor in vtx.nbors:
            dist[vtx.vnum][nbor.vnum] = 1

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return int(np.max(dist))

def ring_slice(ring, start_idx, end_idx):
    """
    Given some buffer ring, returns a slice from start_idx to end_idx
    """
    rsize = len(ring)

    if end_idx < start_idx:
        end_idx = rsize + end_idx
    ring_slice = []
    for i in range(start_idx, end_idx):
        ring_slice.append(ring[i % rsize])
    return ring_slice

def expected_conv_rate_simp(p, r, k, diam):
    """
    Simplest case with equal resource distribution and even
    allocation amongst neighbors
    Returns expected number of iterations before convergence

    There are at most r/2 iterations and at least diameter
    iterations so we only need to iterate from diameter to r/2
    """
    expec = 0
    max_iter = ceil(r / 2)
    for i in range(diam, max_iter + 1):
        i_choose = factorial(max_iter) / (factorial(max_iter - i) * factorial(i))
        prob = i_choose * (p ** i) * ((1 - p) ** (max_iter - i))
        expec += prob * i
    return expec

def sample_powerlaw(n, tot_val, exp, coeff=1, discretize=False):
    """
    Draws n samples from a powerlaw distribution defined by (coeff * x ** exp)
    x_min is the lowest possible value sampled
    Samples sum to tot_val

    requires that exp > 0 and x_min > 0

    Takes ceiling of 
    """

    plaw_raw = coeff * np.random.pareto(exp, n)
    plaw_sum = np.sum(plaw_raw)
    scale_fac = tot_val / plaw_sum
    scaled = scale_fac * plaw_raw
    if discretize:
        return [ ceil(samp) for samp in scaled ]
    else:
        return scaled
