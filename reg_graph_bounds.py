from math import sqrt

def num_reachable(d, k):
    """
    Number of people reachable in distance d
    on a k regular graph
    """
    total = 0
    for i in range(1, d + 1):
        total += k * ((k - 1) ** (i - 1))
    return total

def prob_has_opt(n, d, k):
    return 1 - ((1 - ( 1 / sqrt(n) )) ** num_reachable(d, k))


