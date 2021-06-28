import plotly.graph_objects as go
import networkx as nx
import sim_lib.graph_networkx as gnx
import numpy as np

def make_edge(x, y):
    return  go.Scatter(x         = x,
                       y         = y,
                       line = dict(color = 'black', width = 1),
                       mode      = 'lines')

def graph_vis(G):
    G_nx = gnx.graph_to_nx(G)
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