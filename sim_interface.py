from abc import ABC, abstractmethod

from plotly.offline import iplot
import ipywidgets as widgets
from IPython.display import display, clear_output

import gen_vis
import graph
import simulation

class Simulator(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def update_graph(self, **args):
        pass
    
class SimpleSimulator(Simulator):
    def __init__(self):
        
        #Setup variable factors
        self.num_vertices = -1
        self.regularity = -1
        
        self.vtx_widget = widgets.IntText(value=8,
                                    description='Vertex Count (Rounded to Even)',
                                    disabled=False)
        self.reg_widget = widgets.IntText(value=4,
                                    description='Graph Regularity',
                                    disabled=False)
        self.trans_prob_widget = widgets.FloatSlider(value=0.1,
                                    min=0.0,
                                    max=1.0,
                                    step=0.01,
                                    description='Transmission Probability',
                                    disabled=False,
                                    orientation='horizontal',
                                    continuous_update=False,
                                    readout=True)
        self.time_alloc_widget = widgets.IntText(value=100,
                                    description='Initial Time Allocation',
                                    disabled=False)
        self.random_time_widget = widgets.ToggleButton(value=False,
                                    description='Randomize Time Allocation',
                                    disabled=False,
                                    button_style='info',
                                    tooltip='Random time allocation',
                                    icon='check')
        self.seq_alloc_widget = widgets.ToggleButton(value=False,
                                    description='Sequential Time Allocation',
                                    disabled=False,
                                    button_style='info',
                                    tooltip='Sequential time allocation',
                                    icon='check')
        
        #Update button
        self.update_button = widgets.Button(
                                    description='Update Graph',
                                    disabled=False,
                                    button_style='',
                                    tooltip='Update Graph')
        self.update_button.on_click(self.update_graph)
        
        #Initial graph
        self.graph = None
        
        #Initial plot
        self.plot_fig = gen_vis.create_empty_plot()
        self.radius = 5
        
        self.render_visuals()
        
    def render_visuals(self, utils=None):
        """
        Renders visual components
        If utils is None then does not render stats
        Otherwise expects list of lists of utils per iteration
        """

        #Render widgets
        display(self.vtx_widget)
        display(self.reg_widget)
        display(self.trans_prob_widget)
        display(self.time_alloc_widget)
        display(self.random_time_widget)
        display(self.seq_alloc_widget)
        display(self.update_button)

        #Render overall plot
        iplot(self.plot_fig)

        #Render stat plots
        if not utils:
            return

        avg_fig = gen_vis.plot_util_avg(utils)
        iplot(avg_fig)
        minmax_fig = gen_vis.plot_util_minmax(utils)
        iplot(minmax_fig)
        std_fig = gen_vis.plot_util_std(utils)
        iplot(std_fig)
        opt_fig = gen_vis.plot_util_optimality(utils)
        iplot(opt_fig)
        
    def update_graph(self, button):
        vtx_count = self.vtx_widget.value
        regularity = self.reg_widget.value
        trans_prob = self.trans_prob_widget.value
        time_alloc = self.time_alloc_widget.value
        random_time = self.random_time_widget.value
        seq_alloc = self.seq_alloc_widget.value
        
        #Round to next highest even
        if vtx_count % 2 == 1:
            vtx_count += 1
            
        diameter = graph.calc_diameter(regularity, vtx_count)
            
        #Create new graph
        self.graph = graph.const_kregular(regularity, vtx_count, trans_prob, time_alloc)
        
        #Update plot
        edge_trace = gen_vis.init_edge_traces()
        vtx_trace = gen_vis.init_vertex_traces()
        vtx_trace, edge_trace = gen_vis.init_trace_values(self.graph,
                                                self.radius, vtx_trace, edge_trace)
        
        self.plot_fig.data = []
        self.plot_fig.add_traces([edge_trace, vtx_trace])
        
        #Update graph frames from simulation
        G_s, G_utils = simulation.run_simulation(self.graph, random_time, seq_alloc)
        self.plot_fig = gen_vis.animate_simulation(self.plot_fig, G_utils)

        #Render visuals
        num_steps = len(self.plot_fig['frames'])
        clear_output()
        self.render_visuals(G_utils)
        print("Diameter: {0}".format(diameter))
        print("Step Count: {0}".format(num_steps))

class OverviewSimulator:
    """
    Easy interface for observing aggregated metrics over
    several runs
    agg_method: How to aggregate results of each util set from simulations
                Method should take an 2D array (array of utilities for each run)
    """
    def __init__(self, agg_method):
        self.agg_method = agg_method

        #Graphs created by each simulation set parameters
        self.results = []

        self.range_start_widget = widgets.IntText(value=10,
                                    description='Variable Param Range Start',
                                    disabled=False)
        self.range_end_widget = widgets.IntText(value=100,
                                    description='Variable Param Range End',
                                    disabled=False)
        self.fixed_param_widget = widgets.IntText(value=4,
                                    description='Fixed Param Value',
                                    disabled=False)
        self.select_fixed = widgets.RadioButtons(
                options=['reg', 'size'],
                                    description='Which parameter to fix',
                                    disabled=False)
        self.trans_prob_widget = widgets.FloatSlider(value=0.1,
                                    min=0.0,
                                    max=1.0,
                                    step=0.01,
                                    description='Transmission Probability',
                                    disabled=False,
                                    orientation='horizontal',
                                    continuous_update=False,
                                    readout=True)
        self.time_alloc_widget = widgets.IntText(value=100,
                                    description='Initial Time Allocation',
                                    disabled=False)
        self.random_time_widget = widgets.ToggleButton(value=False,
                                    description='Randomize Time Allocation',
                                    disabled=False,
                                    button_style='info',
                                    tooltip='Random time allocation',
                                    icon='check')
        #Update button
        self.add_set_button = widgets.Button(
                                    description='Add simulation set',
                                    disabled=False,
                                    button_style='',
                                    tooltip='Add a simulation set with current params')
        self.rem_set_button = widgets.Button(
                                    description='Remove simulation set',
                                    disabled=False,
                                    button_style='',
                                    tooltip='Removes most recent set')
        self.run_all_button = widgets.Button(
                                    description='Run all simulations',
                                    disabled=False,
                                    button_style='',
                                    tooltip='Runs all current simulation sets')
        self.add_set_button.on_click(self.add_sim)
        self.rem_set_button.on_click(self.rem_sim)
        self.run_all_button.on_click(self.run_all)

        #Initial visuals
        self.render_visuals()
        
    def render_visuals(self, utils=None):
        """
        Renders visual components
        If utils is None then does not render stats
        Otherwise expects list of lists of utils per iteration
        """

        #Render widgets
        display(self.range_start_widget)
        display(self.range_end_widget)
        display(self.fixed_param_widget)
        display(self.select_fixed)
        display(self.trans_prob_widget)
        display(self.time_alloc_widget)
        display(self.random_time_widget)
        display(self.add_set_button)
        display(self.rem_set_button)
        display(self.run_all_button)

        #Render stat plots
        if not utils:
            return

        agg_data = lambda uts: [ self.agg_method(ut) for ut in uts ]
        aggregated = [ agg_data(uts) for uts in utils ]
        iplot(gen_vis.simple_multiplot(aggregated,
            'Aggregated Utils',
            'Range iteration',
            'Aggregated utility'))

    def add_sim(self, button):
        range_start = self.range_start_widget.value
        range_end = self.range_end_widget.value
        fixed_val = self.fixed_param_widget.value
        fixed = self.select_fixed.value
        trans_rate = self.trans_prob_widget.value
        time_alloc = self.time_alloc_widget.value
        random_time = self.random_time_widget.value

        sim_results = simulation.simple_runs_fix_reg(range_start, range_end,
                fixed_val, trans_rate, time_alloc, fixed, random_time)
        self.results.append(sim_results)
        
    def rem_sim(self, button):
        if len(self.results) > 0:
            self.results.pop(-1)

    def run_all(self, button):
        clear_output()
        self.render_visuals(self.results)
