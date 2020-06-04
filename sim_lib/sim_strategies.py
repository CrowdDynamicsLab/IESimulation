from collections import defaultdict

import numpy as np

from sim_lib.sim_strategy_abc import Strategy

class Uniform(Strategy):
    def __init__(self):
        super().__init__()

    def initialize_model(self, G):
        """
        Set weights based on time allocated to each vertex split evenly
        """
        for vtx in G.vertices:
            t = vtx.time
            n = len(vtx.nbors)

            time_splits = np.linspace(0, t, n + 1, dtype=int)
            allocs = [ time_splits[i + 1] - time_splits[i] \
                    for i in range(len(time_splits) - 1) ]

            self.time_allocs[vtx] = { nbor : ta for nbor, ta \
                    in zip(vtx.nbors, allocs) }

        self.initialize = False

    def get_available_nbor(self, v):
        nbors = list(self.time_allocs[v].keys())

        avail = lambda n : self.time_allocs[v][n] > 0 and self.time_allocs[n][v] > 0
        nbors = [ nb for nb in nbors if avail(nb) ]
        if len(nbors) == 0:
            return None

        nbor = np.random.choice(nbors)

        return nbor, v.edges[nbor]

class RoundRobin(Strategy):
    def __init__(self):
        super().__init__()

    def initialize_model(self, G):
        """
        Allocate t amount of time n ways evenly
        Here t vtx.time and n = |N(vtx)|
        Returns list of allocations and amount of time left

        NOTE: params is passed to conform to model function requirements
        """
        for vtx in G.vertices:
            t = vtx.time
            n = len(vtx.nbors)

            time_splits = np.linspace(0, t, n + 1, dtype=int)
            allocs = [ time_splits[i + 1] - time_splits[i] \
                    for i in range(len(time_splits) - 1) ]

            self.time_allocs[vtx] = { nbor : ta for nbor, ta \
                    in zip(vtx.nbors, allocs) }

        self.initialize = False

class MWU(Strategy):
    def __init__(self, lrate):
        assert lrate > 0, 'Learning rate must be positive'
        super().__init__()

        self.vtx_weights = defaultdict(lambda : {})
        self.lrate = lrate

    # Overload get_available_nbor to sample from distribution by weights
    def get_available_nbor(self, v):
        tot_weight = sum(self.vtx_weights[v].values())
        nbors = list(self.vtx_weights[v].keys())
        probs = [ self.vtx_weights[v][nb] / tot_weight for nb in nbors ]
        nbor = np.random.choice(nbors, p=probs)
        return nbor, v.edges[nbor]

    def update_time_alloc(self, v_prev_util, v_cur, nbor_prev_util, nbor_cur):

        # Update weights for pair
        vcost = self.get_cost(v_prev_util, v_cur)
        self.vtx_weights[v_cur][nbor_cur] = \
            self.vtx_weights[v_cur][nbor_cur] * np.exp(-1 * self.lrate * vcost)
        ncost = self.get_cost(nbor_prev_util, nbor_cur)
        self.vtx_weights[nbor_cur][v_cur] = \
                self.vtx_weights[nbor_cur][v_cur] * np.exp(-1 * self.lrate * ncost)

        # Recalculate time allocs based on updated weights
        v_weight_sum = sum(self.vtx_weights[v_cur].values())
        nbor_weight_sum = sum(self.vtx_weights[nbor_cur].values())
        for n in v_cur.nbors:
            new_alloc = round(v_cur.time * self.vtx_weights[v_cur][n] / v_weight_sum)
            self.time_allocs[v_cur][n] = new_alloc
        for n in nbor_cur.nbors:
            new_alloc = round(nbor_cur.time * self.vtx_weights[nbor_cur][n] / nbor_weight_sum)
            self.time_allocs[nbor_cur][n] = new_alloc

    def initialize_model(self, G):
        for vtx in G.vertices:
            self.vtx_weights[vtx] = { nbor : 1.0 for nbor in vtx.nbors }

            weights = [ w / vtx.degree for w in self.vtx_weights[vtx].values() ]

            # Get initial even alloc
            self.time_allocs[vtx] = { nbor : round(wgt * vtx.time) for
                    nbor, wgt in zip(vtx.nbors, weights) }

        self.initialize = False

    def get_cost(self, v_prev_util, v_cur):
        """
        Gives cost incurred by one state change
        cost in [-1,1]
        Update by Hedge algorithm
        """
        if v_prev_util < v_cur.utility:
            
            # Better, give 0 cost
            return -0.5

        elif v_prev_util > v_cur.utility:

            # Worse, give positive cost
            return 1

        else:

            # No change, give positive cost of resource use
            return 0.5

