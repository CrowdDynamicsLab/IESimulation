import numpy as np

# We will take edges_i and triangles_ijk as our structural parameters based on Mele's results

def util(i, j, G, chars, z, theta):
    # Utility of agent i
    # Page 13
    n = len(chars)
    alpha, beta, gamma = theta
    alpha_val = None
    if z[i] == z[j]:
        alpha_val = alpha[z[i]]
    else:
        alpha_val = alpha[-1]

    rooms1 = 0
    rooms2 = 0
    if chars[i] == chars[j]  and chars[i] == 1:
        rooms1 = beta[0]
    if chars[i] == chars[j]  and chars[i] == 2:
        rooms2 = beta[1]

    tri = 0
    for r in range(n):
        tri += G[j][r] * G[r][i] * gamma[z[i]]

    return alpha_val + rooms1 + rooms2 + tri

def potential(G, z, theta, chars):

    # Page 10
    alpha, beta, gamma = theta
    n = G.shape[0]
    term1 = 0
    term2 = 0
    for i in range(n):
        for j in range(n):
            term1 += G[i][j] * util(i, j, G, chars, z, theta)
            for r in range(n):
                term2 += G[i][j] * G[j][r] * G[i][r] * gamma[z[i]]
    return term1 + (term2 / 6)

def run_model(alpha, beta, gamma, chars):
    # Let K = # communities
    # Each agent put into an unobserved community by multinomial
    # Assume K = 3 from paper

    # If agent i, j in C_k, cost is alpha_k
    # Otherwise it is alpha_b (a constant)
    # Thus alpha is a k+1 length vector

    # Beta is a weight on the covariates

    # Chars is a n x M characteristics matrix

    # In each iteration randomly select a pair
    # Run until potential function stops increasing
    n = len(chars)
    z = np.random.randint(0, 3, n)
    theta = (alpha, beta, gamma)

    G = np.zeros((n, n))

    min_iters = 100
    it = 0
    cur_potential = potential(G, z, theta, chars)
    while True:
        i, j = np.random.randint(0, n, 2)
        if i == j:
            continue

        print(it, i, j)
        i_util = util(i, j, G, chars, z, theta)
        j_util = util(j, i, G, chars, z, theta)
        if i_util + j_util > 0:
            G[i][j] = 1
            G[j][i] = 1

        new_potential = potential(G, z, theta, chars)
        if new_potential <= cur_potential and it >= min_iters:
            break
        cur_potential = new_potential
        it += 1

    return G

n = 150
chars = np.random.randint(1, 3, n)
alpha = [-1.5, -0.1, -0.1, -0.1]
beta = [1.0, 1.0]
gamma = [1.2, 0.8, 1.6]
run_model(alpha, beta, gamma, chars)
