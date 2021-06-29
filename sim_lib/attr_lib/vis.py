import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go

import sim_lib.attr_lib.util as attr_util

def draw_graph(G_attr):
    G_attr_nx = attr_util.graph_to_nx(G_attr)
    G_attr_vtx_pos = nx.drawing.layout.spring_layout(G_attr_nx)
    
    colors = [ v.data['color'] if 'color' in v.data else 'b' for v in G_attr.vertices ]
    node_sizes = []

#    node_sizes = [300*(max(attr_util.remaining_budget(v, G_attr), 0) + 2 ** -10) for v in G_attr.vertices ]
    for v in G_attr.vertices:
        v_attr_util, v_struct_util = v.utility_values(G_attr)
        v_cost = attr_util.calc_cost(v, G_attr)
        v_agg_util = G_attr.sim_params['util_agg'](v_attr_util, v_struct_util, v_cost)
        node_sizes.append(200 * v_agg_util + 25)
    
    # Draw graph
    plt.figure(figsize=(15,15))
    nx.draw_networkx(G_attr_nx, pos=G_attr_vtx_pos, node_color=colors,
            node_size=node_sizes, width=0.6, with_labels=True)
    plt.show()

def make_edge(x, y):
    return  go.Scatter(x         = x,
                       y         = y,
                       line = dict(color = 'black', width = 1),
                       mode      = 'lines')

def graph_vis(G):
    G_nx = attr_util.graph_to_nx(G)
    pos = nx.spring_layout(G_nx)
    edge_trace = []
    for edge in G_nx.edges():

        char_1 = edge[0]
        char_2 = edge[1]

        x0, y0 = pos[char_1]
        x1, y1 = pos[char_2]

        trace  = make_edge([x0, x1, None], [y0, y1, None])

        edge_trace.append(trace)
    
    node_trace = go.Scatter(x         = [],
                            y         = [],
                            textposition = "top center",
                            textfont_size = 10,
                            hoverinfo = 'text',
                            hovertext = [],
                            text      = [],
                            mode      = 'markers+text',
                            marker    = dict(color = [],
                                             size  = [],
                                             line  = None))
    for node in G_nx.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['marker']['size'] += tuple([6*(G_nx.nodes()[node]['struct_util'] + G_nx.nodes()[node]['attr_util'])+2])
        node_info = str(node) + '\n Structural utility: ' + str(np.round(G_nx.nodes()[node]['struct_util'],2)) + '\n Attribute utility: ' + str(np.round(G_nx.nodes()[node]['attr_util'],2)) + '\n Cost: ' + str(np.round(G_nx.nodes()[node]['cost'],2)) 
        node_trace['hovertext'] += tuple([node_info]) 
        node_trace['text'] += tuple([node])
        node_trace['marker']['color'] += tuple([G_nx.nodes()[node]['color']])

    
    layout = go.Layout(xaxis = {'showgrid': False, 'zeroline': False}, yaxis = {'showgrid': False, 'zeroline': False})

    # Create figure
    fig = go.Figure(layout = layout)
    for trace in edge_trace:
        fig.add_trace(trace)
    fig.add_trace(node_trace)
    fig.update_layout(showlegend = False)
    fig.update_xaxes(showticklabels = False)
    fig.update_yaxes(showticklabels = False)
    fig.show()

