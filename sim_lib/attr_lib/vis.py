import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import plotly.graph_objects as go

import sim_lib.attr_lib.util as attr_util

def draw_graph(G_attr, partition, info_string):
    G = attr_util.graph_to_nx(G_attr)
    pos = nx.spring_layout(G)

    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)

    #nodes = nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
    #edges = nx.draw_networkx_edges(G, pos, alpha=0.5)

    node_sizes = []
    labels_array = []

    for v in G_attr.vertices:
        v_attr_util, v_struct_util = v.utility_values(G_attr)
        node_sizes.append(20 * (v_attr_util + v_struct_util))

    nx.draw(G, pos, with_labels = False, nodelist = partition.keys(), node_size=node_sizes, cmap=cmap, node_color=list(partition.values()), font_size = 5)
    save_string = 'figures/networks/community_' + info_string + '.png'
    plt.savefig(save_string, dpi = 1000)
    plt.close('all')

def make_edge(x, y):
    return  go.Scatter(x         = x,
                       y         = y,
                       line = dict(color = 'black', width = 1),
                       mode      = 'lines')

def graph_vis(G, name, string):
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


    layout = go.Layout(xaxis = {'showgrid': False, 'zeroline': False, 'title': string},
                        yaxis = {'showgrid': False, 'zeroline': False},
                        autosize = False,
                        width = 1500,
                        height = 1500)

    # Create figure
    fig = go.Figure(layout = layout)
    for trace in edge_trace:
        fig.add_trace(trace)
    fig.add_trace(node_trace)
    fig.update_layout(showlegend = False)
    fig.update_layout(title = name)
    fig.update_xaxes(showticklabels = False)
    fig.update_yaxes(showticklabels = False)
    #fig.show()
    plot_name = 'figures/networks/' + name + '.png'
    fig.write_image(plot_name)
