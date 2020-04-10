from abc import ABC, abstractmethod
from collections import defaultdict

class Strategy(ABC):
    def __init__(self):

        # Flags whether the strategy needs to be initialized
        self.initialize = True

        # Time each vertex allocates to its neighbors
        self.time_allocs = {}

        # Which neighbor the vertex is going to communicate with next
        # Stores indices
        self.vtx_cur_nbor = defaultdict(lambda : 0)
        super().__init__()

    @abstractmethod
    def update_time_alloc(self, v_prev, v_cur, nbor_prev, nbor_cur):
        pass

    @abstractmethod
    def initialize_model(self, **kwargs):
        pass

    def get_available_nbor(self, v):
        found_nbor = False
        nbor, nedge = list(v.edges.items())[self.vtx_cur_nbor[v]]
        for it in range(v.degree):
            vtime = self.time_allocs[v][nbor]
            ntime = self.time_allocs[nbor][v]
            if (vtime > 0 and v.time > 0) and (ntime > 0 and nbor.time > 0):
                found_nbor = True
                break
            self.vtx_cur_nbor[v] = (self.vtx_cur_nbor[v] + 1) % len(v.edges)
            nbor, nedge = list(v.edges.items())[self.vtx_cur_nbor[v]]

        if not found_nbor:
            return None

        self.vtx_cur_nbor[v] = (self.vtx_cur_nbor[v] + 1) % len(v.edges)

        assert self.time_allocs[v][nbor] > 0
        assert self.time_allocs[nbor][v] > 0
        assert v.time > 0, f"{self.time_allocs[v]} {v.time}"
        assert nbor.time > 0, f"{self.time_allocs[nbor]} {nbor.time}"

        self.time_allocs[v][nbor] -= 1
        self.time_allocs[nbor][v] -= 1
        v.time -= 1
        nbor.time -= 1

        return nbor, nedge
