import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import numpy as np

init_notebook_mode(connected=True)

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

def plot_graph(G, radius):
    """
    Plots graph G with radius size in each ring
    """
    edge_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        hoverinfo='text',
        mode='lines+markers',
        line=dict(width=0.5, color="#888"))

    node_trace = go.Scatter(
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

    graph_xpos, graph_ypos = calc_kreg_pos(G.num_people, radius)
    vtx_positions = zip(G.vertices, graph_xpos, graph_ypos)

    vertex_pos = {}

    #Create vertices
    for vert, xpos, ypos in vtx_positions:
        node_trace['x'] += tuple([xpos])
        node_trace['y'] += tuple([ypos])
        node_trace['text'] += tuple([str(vert)])
        vertex_pos[vert.vnum] = (xpos, ypos)
        node_trace['marker']['color'] += tuple([vert.utility])
            
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
          
    g_fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title="Information Elicitation Simulation",
                       showlegend=False,
                       hovermode='closest',
                       xaxis=dict(showgrid=False, zeroline=False,
                                  showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False,
                                  showticklabels=False)))

    iplot(g_fig)
