import copy

import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import numpy as np

#GENERAL PLOT METHODS
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

def init_trace_values(G, radius, vtx_trace, edge_trace):
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

def create_empty_plot():
    edge_trace = init_edge_traces()
    vtx_trace = init_vertex_traces()
    g_fig = go.Figure(data=[edge_trace, vtx_trace],
                      layout=go.Layout(
                        title="Information Elicitation Simulation",
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False,
                                  showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False,
                                  showticklabels=False),
                      updatemenus=[dict(
                          buttons=[dict(args=[None],
                                        label='Play',
                                        method='animate')],
                          pad={'r':10, 't':87},
                          showactive=False,
                          type='buttons')]),
                      frames=[])
    return g_fig

def plot_graph(G, radius):
    """
    Plots graph G with radius size in each ring
    """

    edge_trace = init_edge_traces()
    vtx_trace = init_vertex_traces()

    vtx_trace, edge_trace = init_trace_values(G, radius, vtx_trace, edge_trace)
    
    #Figure def taken from plotly tutorial
    g_fig = go.Figure(data=[edge_trace, vtx_trace],
                      layout=go.Layout(
                        title="Information Elicitation Simulation",
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False,
                                  showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False,
                                  showticklabels=False),
                      updatemenus=[dict(
                          buttons=[dict(args=[None],
                                        label='Play',
                                        method='animate')],
                          pad={'r':10, 't':87},
                          showactive=False,
                          type='buttons')]),
                      frames=[])

    return g_fig

def animate_simulation(g_fig, utils):
    """
    g_fig is a plotly Figure
    utils represents utilites over time of the simulation
    """

    #Vertex values
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
    g_fig['frames'] = sim_frames
    return g_fig

#STATS OVER TIME
def plot_util_avg(utils):
    """
    Plots mean and median of utils over time
    """
    iterations = list(range(len(utils)))
    means = np.mean(utils, axis=1)
    medians = np.median(utils, axis=1)
    mean_trace = go.Scatter(x=iterations, y=means, mode='lines', name='Mean')
    med_trace = go.Scatter(x=iterations, y=medians, mode='lines', name='Median')
    avg_fig = go.Figure(data=[mean_trace, med_trace],
                        layout=go.Layout(
                            title='Mean and median over time',
                            showlegend=True,
                            xaxis=dict(title=dict(text='Iteration')),
                            yaxis=dict(title=dict(text='Utility'))))
    return avg_fig

def plot_util_minmax(utils):
    """
    Plots min and max of utils over time
    """
    iterations = list(range(len(utils)))
    minvals = np.max(utils, axis=1)
    maxvals = np.min(utils, axis=1)
    max_trace = go.Scatter(x=iterations, y=maxvals, mode='lines', name='Mean')
    min_trace = go.Scatter(x=iterations, y=minvals, mode='lines', name='Median')
    minmax_fig = go.Figure(data=[max_trace, min_trace],
                        layout=go.Layout(
                            title='Min and max over time',
                            showlegend=True,
                            xaxis=dict(title=dict(text='Iteration')),
                            yaxis=dict(title=dict(text='Utility'))))
    return minmax_fig

def plot_util_std(utils):
    """
    Plots std of utils over time
    """
    iterations = list(range(len(utils)))
    stdvals = np.std(utils, axis=1)
    std_trace = go.Scatter(x=iterations, y=stdvals, mode='lines', name='Std')
    std_fig = go.Figure(data=[std_trace],
                        layout=go.Layout(
                            title='Standard deviation over time',
                            showlegend=True,
                            xaxis=dict(title=dict(text='Iteration')),
                            yaxis=dict(title=dict(text='Utility'))))
    return std_fig

def plot_util_optimality(utils):
    """
    Plots percent optimality (current overall utility over max possible)
    over time
    """
    iterations = list(range(len(utils)))
    max_util = len(utils[0])

    util_sums = np.sum(utils, axis=1)
    optvals = util_sums / max_util
    opt_trace = go.Scatter(x=iterations, y=optvals, mode='lines', name='Percent')
    opt_fig = go.Figure(data=[opt_trace],
                        layout=go.Layout(
                            title='Percent optimality over time',
                            showlegend=True,
                            xaxis=dict(title=dict(text='Iteration')),
                            yaxis=dict(title=dict(text='Percent optimality'))))
    return opt_fig

