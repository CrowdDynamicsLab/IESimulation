import numpy as np

def gen_const_ratings(provs):
    """
    Returns a dict of ratings of provider : int
    representing ratings of each respective provider
    """
    ratings = np.linspace(0, 1, len(provs))
    np.random.shuffle(ratings)
    return { provs[didx] : ratings[didx] for didx in range(len(provs)) }

def calc_diameter(k, n):
    """
    For a k regular graph with n vertices, calculates the
    diameter based on the generation scheme used in const_kregular
    """

    #n should always be even
    ring_dist = n // 2
    m = k // 2
    return (ring_dist // m) + (ring_dist % m)

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
