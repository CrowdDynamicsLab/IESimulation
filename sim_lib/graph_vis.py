import copy
import math

import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import numpy as np

# Methods for plotting graphs and changes in util over epochs

def calc_rl_pos(n, r):
    """
    Calculates coordinates for a cycle graph
    with radius r
    """
    
    #Calculate polar coordinates then convert to cartesian
    angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    xpos = r * np.cos(angles)
    ypos = r * np.sin(angles)
    return xpos, ypos

def calc_grid_pos(n, m, l, w):
    """
    Calculates coordinates for a n x m grid with dimension l x w
    """
    xpos = np.linspace(0, w, m)
    xpos = [ [xp] for xp in xpos ]
    xpos = np.repeat(xpos, n, 1).T.flatten()

    # Shift ypos down by appropriate amount
    ypos = np.linspace(0, l, n)
    ypos = [ [yp] for yp in ypos ]
    ypos = np.repeat(ypos, m, 1).flatten()
    
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
            colorscale='hot',
            cmax=1,
            cmin=0,
            showscale=True,
            reversescale=True))
    return vtx_trace

def init_graph_fig_dict(edge_trace, vtx_trace):
    g_fig_dict = {}
    g_fig_dict['data'] = [edge_trace, vtx_trace]
    g_fig_dict['frames'] = []
    g_fig_dict['layout'] = dict(
                        title="Information Elicitation Simulation",
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False,
                                  showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False,
                                  showticklabels=False),
                        updatemenus=[
                              { "buttons": [
                                { "args": [None, {"frame": {"duration": 500, "redraw": False},
                                                "fromcurrent": True, "transition": {"duration": 300,
                                                                                    "easing": "quadratic-in-out"}}],
                                "label": "Play",
                                "method": "animate" },
                                { "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                                  "mode": "immediate",
                                                  "transition": {"duration": 0}}],
                                "label": "Pause",
                                "method": "animate" } ],
                                "direction": "left",
                                "pad": {"r": 10, "t": 87},
                                "showactive": False,
                                "type": "buttons",
                                "x": 0.1,
                                "xanchor": "right",
                                "y": 0,
                                "yanchor": "top" }
                          ])

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 15},
            "prefix": "Epoch:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    g_fig_dict['layout']['sliders'] = [sliders_dict]

    return g_fig_dict

def init_trace_values(G, radius, vtx_trace, edge_trace, mode='ring'):
    """
    Adds initial data to edge and vertex traces
    """

    if mode == 'ring':
        graph_xpos, graph_ypos = calc_rl_pos(G.num_people, radius)
    elif mode == 'grid':
        grid_n = math.floor(math.sqrt(len(G.vertices)))
        grid_m = len(G.vertices) // grid_n
        while grid_n > 1:
            if grid_m * grid_n == len(G.vertices):
                break
            grid_n -= 1
            grid_m = len(G.vertices) // grid_n

        graph_xpos, graph_ypos = calc_grid_pos(grid_n, grid_m, radius, radius)
    else:
        raise ValueError("Mode must be one of 'ring' or 'grid'")

    # Sort vertices by vertex num
    vnum_map = { vtx.vnum : vtx for vtx in G.vertices }
    sorted_vnums = sorted(list(vnum_map.keys()))
    sorted_vtxs = [ vnum_map[vnum] for vnum in sorted_vnums ]

    vtx_positions = zip(sorted_vtxs, graph_xpos, graph_ypos)

    vertex_pos = {}

    # Create vertices
    for vert, xpos, ypos in vtx_positions:
        vtx_trace['x'] += tuple([xpos])
        vtx_trace['y'] += tuple([ypos])
        vtx_text = '{0}: {1}'.format(str(vert), str(vert.utility))
        vtx_trace['text'] += tuple([vtx_text])
        vertex_pos[vert.vnum] = (xpos, ypos)
        vtx_trace['marker']['color'] += tuple([vert.utility - 0.5])

    # Create edges
    for vert in sorted_vtxs:
        vert_x, vert_y = vertex_pos[vert.vnum]
        for nbor in vert.edges:
            nbor_x, nbor_y = vertex_pos[nbor.vnum]
            edge_trace['x'] += tuple([vert_x, nbor_x, None])
            edge_trace['y'] += tuple([vert_y, nbor_y, None])
            
            trans_rate = vert.edges[nbor].trate
            edge_trace['text'] += tuple([str(trans_rate)])

    return vtx_trace, edge_trace

def create_empty_plot():
    #NOTE: may not be needed
    edge_trace = init_edge_traces()
    vtx_trace = init_vertex_traces()

    g_fig_dict = init_graph_fig_dict(edge_trace, vtx_trace)
    g_fig = go.Figure(g_fig_dict)
    return g_fig

def vis_G(G, radius, utils, mode='ring', ntwk_name=''):
    """
    Creates visualization of graph G with radius size
    Arranges vertices as 'mode' type graph
    utils is list of utils indexed by vertex in order of G.vertices over the run of the simulation
    """

    edge_trace = init_edge_traces()
    vtx_trace = init_vertex_traces()

    vtx_trace, edge_trace = init_trace_values(G, radius, vtx_trace, edge_trace, mode)

    vnum_map = { vtx.vnum : vtx for vtx in G.vertices }
    sorted_vnums = sorted(list(vnum_map.keys()))
    vnum_idx = { vtx.vnum : idx for idx, vtx in enumerate(G.vertices) }
    sorted_utils = [ [ epoch_uts[vnum_idx[vnum]] for vnum in sorted_vnums ] for epoch_uts in utils ]

    g_fig_dict = init_graph_fig_dict(edge_trace, vtx_trace)

    # Vertex values
    edge_trace, vtx_trace = g_fig_dict['data']

    vtx_trace_copies = [ copy.deepcopy(vtx_trace) for i in range(len(utils)) ]

    vis_frames = []

    # Only need to update text and color for each vertex trace copy, edges are static
    for epoch, (vt_cpy, util) in enumerate(zip(vtx_trace_copies, sorted_utils)):

        # Update text
        update_txt = lambda txt, u : txt[:txt.find(':') + 1] + str(u)
        vtx_text = [ update_txt(txt, u) for txt, u in zip(vt_cpy['text'], util) ]
        vt_cpy['text'] = vtx_text

        # Update color
        vt_cpy['marker']['color'] = tuple(util)

        slider_step = { 'args' : [
            [epoch],
            { 'frame' : {'duration': 300, 'redraw' : False}, 'mode' : 'immediate',
              'transition' : { 'duration' : 3000 }}
            ],
            'label' : epoch,
            'method' : 'animate' }
        g_fig_dict['layout']['sliders'][0]['steps'].append(slider_step)
        vis_frames.append( dict(data=[edge_trace, vt_cpy], name=str(epoch)) )

    # Add frames to animation
    g_fig_dict['frames'] = vis_frames

    g_fig_dict['layout']['title'] = f'Information Elicitation Sim (ntwk={ntwk_name})'

    g_fig = go.Figure(g_fig_dict)
    return g_fig

