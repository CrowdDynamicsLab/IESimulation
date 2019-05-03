import copy

import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import numpy as np

def calc_kreg_pos(n, r):
    """
    Calculates coordinates for a cycle graph
    with radius r
    """
    
    #Calculate polar coordinates then convert to cartesian
    angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    xpos = r * np.cos(angles)
    ypos = r * np.sin(angles)
    return xpos, ypos

def init_edge_traces():
    """
    Initialize edge traces with no data
    """

    edge_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        hoverinfo='text',
        mode='lines',
        line=dict(width=0.5, color="#888"))
    return edge_trace

def init_vertex_traces():
    """
    Initialize vertex traces with no data
    """

    vtx_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        hoverinfo='text',
        mode='markers',
        marker=dict(
            size=10,
            line=dict(width=2),
            color=[],
            colorscale='Electric',
            showscale=True,
            reversescale=True))
    return vtx_trace

def populate_traces_init(G, radius, vtx_trace, edge_trace):
    """
    Adds initial data to edge and vertex traces
    """

    graph_xpos, graph_ypos = calc_kreg_pos(G.num_people, radius)
    vtx_positions = zip(G.vertices, graph_xpos, graph_ypos)

    vertex_pos = {}

    #Create vertices
    for vert, xpos, ypos in vtx_positions:
        vtx_trace['x'] += tuple([xpos])
        vtx_trace['y'] += tuple([ypos])
        vtx_text = '{0} {1}'.format(str(vert), str(vert.utility))
        vtx_trace['text'] += tuple([vtx_text])
        vertex_pos[vert.vnum] = (xpos, ypos)
        vtx_trace['marker']['color'] += tuple([vert.utility])
            
    #Create edges
    for vert in G.vertices:
        vert_x, vert_y = vertex_pos[vert.vnum]
        for nbor in vert.edges:
            nbor_x, nbor_y = vertex_pos[nbor.vnum]
            edge_trace['x'] += tuple([vert_x, nbor_x, None])
            edge_trace['y'] += tuple([vert_y, nbor_y, None])
            
            #Increase darkness based on trate
            trans_rate = vert.edges[nbor].trate
            edge_trace['text'] += tuple([str(trans_rate)])

    return vtx_trace, edge_trace


def plot_graph(G, radius):
    """
    Plots graph G with radius size in each ring
    """

    edge_trace = init_edge_traces()
    vtx_trace = init_vertex_traces()

    vtx_trace, edge_trace = populate_traces_init(G, radius, vtx_trace, edge_trace)

    g_fig = go.Figure(data=[edge_trace, vtx_trace],
                      layout=go.Layout(
                        title="Information Elicitation Simulation",
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False,
                                  showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False,
                                  showticklabels=False),
                        updatemenus= [{'type': 'buttons',
                                       'buttons': [{'label': 'Play Simulation',
                                       'method': 'animate',
                                       'args': [None]}]}]),
                      frames=[])

    return g_fig

def update_vtx_trace_text(text, util):
    return text[:text.find(' ') + 1] + str(util)

def animate_simulation(g_fig, utils):
    """
    g_fig is a plotly Figure
    utils represents utilites over time of the simulation
    """
    edge_trace = g_fig['data'][0]
    vtx_trace = g_fig['data'][1]
    vtx_copies = [ copy.deepcopy(vtx_trace) for i in range(len(utils)) ]
    for vcpy, util in zip(vtx_copies, utils):

        #Update text
        update_txt = lambda txt, u : txt[:txt.find(' ') + 1] + str(u)
        vtx_text = [ update_txt(txt, u) for txt, u in zip(vcpy['text'], util) ]
        vcpy['text'] = vtx_text

        #Update color
        vcpy['marker']['color'] = tuple(util)

    #Add frames to animation
    sim_frames = [ dict(data=[edge_trace, sim_vtx]) for sim_vtx in vtx_copies ]
    sim_frames.append(dict(data=[edge_trace, vtx_trace]))
    g_fig['frames'] = sim_frames
    return g_fig
