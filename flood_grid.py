import sim_lib.graph_create as gc
from sim_lib.graph import Vertex

flatten = lambda l : [ l_item for l_row in l for l_item in l_row ]

vtx_grid = [ [ Vertex(0, 0, { 0 : 0}, i + (j * 9)) for i in range(9) ] for j in range(9) ] 
vtx_grid_flat = flatten(vtx_grid)

grid = gc.kleinberg_grid(9, 9, 0, 1, k=0, q=0, vtx_set=vtx_grid_flat)
grid_center = vtx_grid[4][4]

##################################
# Add edges
##################################

# Add edge to far center of quadrants
grid.add_edge(grid_center, vtx_grid[2][6], 1)
grid.add_edge(grid_center, vtx_grid[2][2], 1)
grid.add_edge(grid_center, vtx_grid[6][6], 1)
#grid.add_edge(grid_center, vtx_grid[6][2], 1)

##################################
# Run flood fill
##################################
def flood_fill(G, seed):
    """
    Deterministic flood fill (probability 1 of spread)
    """

    flood_set = set(seed)

    iter_cnt = 0
    while True:
        if len(flood_set) == len(G.vertices):
            break

        f_nborhood_grid = [ f_v.nbors for f_v in flood_set ]
        f_nborhood = set(flatten(f_nborhood_grid))
        flood_set = flood_set.union(f_nborhood)

        print(len(flood_set))

        iter_cnt += 1

    return iter_cnt

print(flood_fill(grid, [grid_center]))
