import numpy as np

# We will take edges_i and triangles_ijk as our structural parameters based on Mele's results

def util(i, j, G, chars, z, theta):
    # Utility of agent i
    # Page 13
    n = chars.shape[0]
    alpha, beta, gamma = theta
    alpha_val = None
    if z[i] == z[j]:
        alpha_val = alpha[z[i]]
    else:
        alpha_val = alpha[-1]
    
    rooms1 = 0
    rooms2 = 0
    if chars[i][0] == chars[j][0]  and chars[i][0] == 1:
        rooms1 = beta[0]
    if chars[i][1] == chars[j][1]  and chars[i][1] == 1:
        rooms2 = beta[1]
    
    tri = 0
    for r in range(n):
        tri += G[j][r] * G[r][i] * gamma[z[i]]

    return alpha_val + rooms1 + rooms2 + tri

def potential(G, z, theta):

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
    n = chars.shape[0]
    z = np.random.randint(0, 3, n)
    theta = (alpha, beta, gamma)

    G = np.zeros((n, n))
   
    min_iters = 100 
    it = 0
    cur_potential = potential(G, z, theta)
    while True:
        i, j = np.random.randint(0, n, 2)
        if i == j:
            continue

        i_util = util(i, j, G, chars, z, theta)
        j_util = util(j, i, G, chars, z, theta)
        if i_util + j_util > 0:
            G[i][j] = 1
            G[j][i] = 1

        new_potential = potential(G, z, theta)
        if new_potential <= cur_potential and it >= min_iters:
            break
        cur_potential = new_potential
        it += 1

    return G

alpha = [-1, -0.75, -0.5, -1.25]
beta = [1, 1.2]
gamma = [0.969, 1.573, 0.995]
n = 20
chars = np.zeros((n, 2))
char_vals = np.random.randint(0, 2, n)
for idx, v in enumerate(char_vals):
    chars[idx][v] = 1

G = run_model(alpha, beta, gamma, chars)
print(G)

