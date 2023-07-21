import math

import numpy as np
from scipy.optimize import linprog

# Solve EMD using custom distance matrix

# define the distance function between two ordered pairs (a_1, d_1) and (a_2, d_2)
# for associativity and density respectively

def norm_distance(a_1, d_1, a_2, d_2):
    return abs(a_1 - a_2) + ( abs(d_1 - d_2) ** 0.5 )
    #return abs(a_1 - a_2) + abs(d_1 - d_2)

def emd(dist1, dist2, dist_func=norm_distance):
    # each dist is a list of tuples of the form
    # [ ( (a_1, d_1), freq_1 ),..., ( (a_n, d_n), freq_n ) ]
    # where (a_i, d_i) is the associativity and density value
    # and freq_i is the relative frequency

    # LP from here: https://en.wikipedia.org/wiki/Earth_mover%27s_distance

    # some numerical checks

    # size of dist1
    n = len(dist1)

    # size of dist2
    m = len(dist2)

    # sum of the weights in this case is 1 so
    # the final constraint in the LP is equality to 1
    eq_const_mat = np.ones((1, n * m))

    # inequality constraint
    # put the dist1 weights first
    dist1_weights = np.array([ tp[1] for tp in dist1 ])
    dist2_weights = np.array([ tp[1] for tp in dist2 ])
    ineq_const_bounds = np.concatenate((dist1_weights, dist2_weights))

    # const mat is binary, sets whether the flow is summed or not
    ineq_const_mat = np.zeros((n + m, n * m))
    for i in range(n):
        for j in range(m):
            ineq_const_mat[i][(i * m) + j] = 1
    for j in range(m):
        for i in range(n):
            ineq_const_mat[n + j][(i * m) + j] = 1

    # The distance matrix D
    D = np.zeros( (n, m) )
    for i in range(n):
        for j in range(m):
            op1 = dist1[i][0]
            op2 = dist2[j][0]
            D[i][j] = dist_func(op1[0], op1[1], op2[0], op2[1])

    # These should be 1 but due to numerical issues will be slightly off
    dist1_weight = np.sum(dist1_weights)
    dist2_weight = np.sum(dist2_weights)
    weight_const = min(dist1_weight, dist2_weight)
    assert math.isclose(weight_const, 1), 'Total weight should be close to 1'

    c = np.reshape(D, (n * m, ))

    res = linprog(c,
        A_ub=ineq_const_mat,
        b_ub=ineq_const_bounds,
        A_eq=eq_const_mat,
        b_eq=weight_const,
        method='highs',
        #bounds=(0, None)
        bounds=(0, 1)
    )


    flows = res.x
    obj_val = res.fun

    return obj_val / sum(flows)
