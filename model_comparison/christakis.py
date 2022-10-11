import numpy as np
import networkx as nx

def U(i, j, X, F, G, eps_ij, theta):
    
    # theta = (b_0, b_1, omega, alpha), global
    # b_0 scalar, b_1 vector, omega diag matrix, alpha vector of 4 elements
    b_0, b_1, omega, alpha = theta

    # delta is degree of preference over C (global scaling factor over # classes in paper)
    
    # eps_ij contains unobserved util

    x_i = X[i]
    x_j = X[j]
    partner_prefs = b_1 * x_j
    disutil = (x_i - x_j) * omega * (x_i - x_j)
    ntwk_eff = alpha[0] * F[j] + alpha[1] * (F[j]**2) + \
        alpha[2] * (1 if G[i, j] == 2 else 0) + alpha[3] * (1 if G[i, j] == 3 else 0)
    error = eps_ij
    return b_0 + partner_pref - disutil + ntwk_eff + error

def update_G(G, i, j):

    # Update shortest path matrix G after adding edge ij
    G[i, j] = 1
    G[j, i ] = 1
    size = len(G)
    for x in range(size):
        for y in range(size):
            if x == y:
                continue
            xijy_len = G[x, i] + 1 + G[j, y]
            xjiy_len = G[x, j] + 1 + G[i, y]
            min_len = min(xijy_len, xjiy_len)
            G[x, y] = min(G[x, y], min_len)

def run_sim(X, theta):

    # Eps a matrix with independent type 1 extreme value between all i, j
    # TODO: REPLACE
    Eps = 

    N = X.shape[0]
    D = np.zeros((N, N))
    F = [0] * N
    G = np.ones((N, N)) * np.inf

    pairs = [ (i, j) for i in range(_N) for j in range(i + 1, _N) ]
    np.random.shuffle(pairs)

    for p in pairs:
        i_util = U(p[0], p[1], X, F, G, Eps[p[0], p[1]], theta)
        j_util = U(p[1], p[0], X, F, G, Eps[p[1], p[0]], theta)

        # Should be strict?
        if i_util > 0 and j_util > 0:
            D[i, j] = 1
            D[j, i] = 1
            F[i] += 1
            F[j] += 1
            update_G(G, i, j)

    return D
