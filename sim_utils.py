from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import numpy as np
from math import ceil

from simulation import run_simulation
from graph_create import configuration_model, reduce_providers_simplest, powerlaw_dist_time
import util

def plot_utils(social_utils, r, regularity, diam):
    fig = go.Figure()

    #Add averages trace
    prk_vals = [ p * r / regularity for (p, uts) in social_utils ]
    ut_vals = [ np.average(uts) for (p, uts) in social_utils ]
    fig.add_trace(go.Scatter(x=prk_vals,
        y=ut_vals, mode='lines+markers', hoverinfo='skip',
        name='avg social welfare rate'))
    
    #Add individual points trace
    indiv_p_vals = []
    indiv_ut_vals = []
    for (p, uts) in social_utils:
        indiv_ut_vals.extend(uts)
        indiv_p_vals.extend([p * r / regularity] * len(uts))
    fig.add_trace(go.Scatter(x=indiv_p_vals,
        y=indiv_ut_vals, mode='markers', hoverinfo='skip', opacity=0.7,
        marker=dict(color='LightSkyBlue'), name='social welfare per p'))
    
    fig.layout.update(showlegend=False)
    title_fmt = "Average Social Welfare (Normalized) vs p*r/k (r={0}, diam={1})"
    plot_title = title_fmt.format(r, diam)
    fig.layout.update(title=plot_title, showlegend=True,
        xaxis=dict(title='p*r/k'), yaxis=dict(title='avg social welfare'),
        plot_bgcolor='rgba(0,0,0,0)')

    iplot(fig)

def run_sim(num_vertices, regularity, graph_func, plaw_resources=False):
    graph_diam = None
    for r in np.linspace(100, 0, 10, endpoint=False):
        social_utils = []
        for p in np.linspace(1, 0, 10, endpoint=False):
            utils = []
            num_iter = 10
            for i in range(num_iter):
                ring_lat = graph_func(num_vertices, regularity, p, r)

                if not graph_diam:
                    graph_diam = util.calc_diameter(ring_lat)

                simp_ring_lat = reduce_providers_simplest(ring_lat)
                
                if plaw_resources:
                    powerlaw_dist_time(ring_lat, 2)

                sim_g, sim_utils = run_simulation(ring_lat)

                #Get global social welfare at end of simulation normalized by size
                utils.append(sum(sim_utils[-1]) / len(ring_lat.vertices))
            social_utils.append((p, utils))

        plot_utils(social_utils, r, regularity, graph_diam)
