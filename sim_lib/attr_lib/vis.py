import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import sim_lib.graph_networkx as gnx
import sim_lib.attr_lib.util as attr_util

def draw_graph(G_attr):
    G_attr_nx = gnx.graph_to_nx(G_attr)
    G_attr_vtx_pos = nx.drawing.layout.spring_layout(G_attr_nx)
    
    colors = [ v.data['color'] if 'color' in v.data else 'b' for v in G_attr.vertices ]
    
#    node_sizes = [300*(max(attr_util.remaining_budget(v, G_attr), 0) + 2 ** -10) for v in G_attr.vertices ]
    node_sizes = [150*v.data['total_attr_util'](v, G_attr)+50 for v in G_attr.vertices ]
    
    # Draw graph
    plt.figure(figsize=(15,15))
    nx.draw_networkx(G_attr_nx, pos=G_attr_vtx_pos, node_color=colors,
            node_size=node_sizes, width=0.6, with_labels=True)
    plt.show()
