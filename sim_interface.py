from abc import ABC, abstractmethod
from collections import defaultdict

from plotly.offline import iplot
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np

import gen_vis
import graph_create
import graph_util
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
        
        self.vtx_widget = widgets.IntText(value=100,
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
        self.seq_trans_widget = widgets.ToggleButton(value=False,
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

        #Widgets to render, in order
        self.render_set = [self.vtx_widget, self.reg_widget, self.trans_prob_widget,
                self.time_alloc_widget, self.random_time_widget, self.seq_trans_widget,
                self.update_button]
        
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
        for ritem in self.render_set:
            display(ritem)

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
        seq_trans = self.seq_trans_widget.value
        
        #Round to next highest even
        if vtx_count % 2 == 1:
            vtx_count += 1
            
        diameter = graph_util.calc_diameter(regularity, vtx_count)
            
        #Create new graph
        self.graph = graph_create.const_kregular(regularity, vtx_count, trans_prob, time_alloc)
        
        #Update plot
        edge_trace = gen_vis.init_edge_traces()
        vtx_trace = gen_vis.init_vertex_traces()
        vtx_trace, edge_trace = gen_vis.init_trace_values(self.graph,
                                                self.radius, vtx_trace, edge_trace)
        
        self.plot_fig.data = []
        self.plot_fig.add_traces([edge_trace, vtx_trace])
        
        #Update graph frames from simulation
        G_s, G_utils = simulation.run_simulation(self.graph, random_time, seq_trans)
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
    avg_run_count: If taking average, how many runs to average over
    """
    def __init__(self, agg_method, avg_run_count):
        self.agg_method = agg_method
        self.avg_run_count = avg_run_count

        #Graphs created by each simulation set parameters
        self.results = defaultdict(lambda : [])

        self.range_start_widget = widgets.IntText(value=0,
                                    description='Variable Param Range Start',
                                    disabled=False)
        self.range_end_widget = widgets.IntText(value=100,
                                    description='Variable Param Range End',
                                    disabled=False)
        self.range_num_iter = widgets.IntText(value=100,
                                    description='Number iters',
                                    disabled=False)
        self.fixed_degree_widget = widgets.IntText(value=4,
                                    description='Degree Value',
                                    disabled=False)
        self.fixed_size_widget = widgets.IntText(value=100,
                                    description='Size Value',
                                    disabled=False)
        self.fixed_trate_widget = widgets.FloatSlider(value=0.1,
                                    min=0.0,
                                    max=1.0,
                                    step=0.01,
                                    description='Transmission Probability',
                                    disabled=False,
                                    orientation='horizontal',
                                    continuous_update=False,
                                    readout=True)
        self.select_var = widgets.RadioButtons(
                options=['reg', 'size', 'trate'],
                                    description='Which parameter to vary',
                                    disabled=False)
        self.time_alloc_widget = widgets.IntText(value=100,
                                    description='Initial Time Allocation',
                                    disabled=False)
        self.random_time_widget = widgets.ToggleButton(value=False,
                                    description='Randomize Time Allocation',
                                    disabled=False,
                                    button_style='info',
                                    tooltip='Random time allocation',
                                    icon='check')
        self.seq_trans_widget = widgets.ToggleButton(value=False,
                                    description='Sequential Time Allocation',
                                    disabled=False,
                                    button_style='info',
                                    tooltip='Sequential time allocation',
                                    icon='check')
        self.log_space_widget = widgets.ToggleButton(value=False,
                                    description='Use log space',
                                    disabled=False,
                                    button_style='info',
                                    tooltip='Use log space for var instead of lin space',
                                    icon='check')
        self.avg_res_widget = widgets.ToggleButton(value=False,
                                    description='Average Results',
                                    disabled=False,
                                    button_style='info',
                                    tooltip='Toggle for averaging results vs run once',
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
        self.plot_all_button = widgets.Button(
                                    description='Plot all simulations',
                                    disabled=False,
                                    button_style='',
                                    tooltip='Plots all current simulation sets')
        self.add_set_button.on_click(self.add_sim)
        self.rem_set_button.on_click(self.rem_sim)
        self.plot_all_button.on_click(self.plot_all)

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
        display(self.range_num_iter)
        display(self.fixed_size_widget)
        display(self.fixed_degree_widget)
        display(self.fixed_trate_widget)
        display(self.select_var)
        display(self.time_alloc_widget)
        display(self.seq_trans_widget)
        display(self.random_time_widget)
        display(self.log_space_widget)
        display(self.avg_res_widget)
        display(self.add_set_button)
        display(self.rem_set_button)
        display(self.plot_all_button)

        #Render stat plots
        if not utils:
            return

        #Method for aggregating data
        agg_data = lambda uts: [ self.agg_method(ut) for ut in uts ]

        range_keys = sorted(list(utils.keys()))
        tester = utils[range_keys[1]]
        test = agg_data(tester)
        agg_data_dict = { rk : agg_data(utils[rk]) for rk in range_keys }

        #Should be an array containing a single array
        aggregated = [[ np.mean(agg_data_dict[rk]) for rk in range_keys ]]

        iplot(gen_vis.simple_multiplot(aggregated,
            'Aggregated Utils',
            'Range iteration',
            'Aggregated utility'))

    def add_sim(self, button):
        num_iters = self.range_num_iter.value

        size = self.fixed_size_widget.value
        degree = self.fixed_degree_widget.value
        trate = self.fixed_trate_widget.value

        range_start = self.range_start_widget.value
        range_end = self.range_end_widget.value
        var = self.select_var.value

        time_alloc = self.time_alloc_widget.value
        random_time = self.random_time_widget.value
        seq_update = self.seq_trans_widget.value
        log_space = self.log_space_widget.value

        avg_runs = self.avg_res_widget.value

        var_range = None
        if log_space:
            log_start = 0
            log_end = 0
            if range_start == 0:
                log_start = -100 #This is just to get close to 0, -inf throws errors
            else:
                log_start = np.log(range_start)
            if range_end == 0:
                log_end = -100
            else:
                log_end = np.log(range_end)
            var_range = np.logspace(log_start, log_end, num_iters)
        else:
            var_range = np.linspace(range_start, range_end, num_iters)

        #If runs should be averaged run 10 times
        num_to_run = self.avg_run_count if avg_runs else 1
        for i in range(num_to_run):
            sim_results = simulation.many_runs_fix_vars(size, degree, trate,
                    var_range, time_alloc, var, random_time, seq_update)
            for v_idx, var_val in enumerate(var_range):
                self.results[var_val].append(sim_results[v_idx])
        print("Finished adding simulation")
        
    def rem_sim(self, button):
        if len(self.results) > 0:
            for util_set in self.results.values():
                util_set.pop(-1)

    def plot_all(self, button):
        clear_output()
        self.render_visuals(self.results)
