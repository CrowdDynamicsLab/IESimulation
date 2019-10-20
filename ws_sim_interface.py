import ipywidgets as widgets
from IPython.display import display, clear_output

import gen_vis
import graph_create
import graph_util
import simulation
from sim_interface import SimpleSimulator

class WattsStrogatzSimulator(SimpleSimulator):
    def __init__(self):
        self.beta_widget = widgets.FloatSlider(value=0.1,
                min=0.0,
                max=1.0,
                step=0.01,
                description='Rewire beta value',
                disabled=False,
                orientation='horizontal',
                continuous_update=False,
                readout=True)
        self.far_trans_prob_widget = widgets.FloatSlider(value=0.1,
                min=0.0,
                max=1.0,
                step=0.01,
                description='Rewired Transmission Probability',
                disabled=False,
                orientation='horizontal',
                continuous_update=False,
                readout=True)

        super(WattsStrogatzSimulator, self).__init__()

        self.render_set.insert(3, self.far_trans_prob_widget)
        self.render_set.insert(2, self.beta_widget)
        clear_output()
        self.render_visuals()
        
    def update_graph(self, button):
        vtx_count = self.vtx_widget.value
        regularity = self.reg_widget.value
        beta = self.beta_widget.value
        close_trans_prob = self.trans_prob_widget.value
        far_trans_prob = self.far_trans_prob_widget.value
        time_alloc = self.time_alloc_widget.value
        random_time = self.random_time_widget.value
        seq_trans = self.seq_trans_widget.value
        
        #Round to next highest even
        if vtx_count % 2 == 1:
            vtx_count += 1
            
        diameter = graph_util.calc_diameter(regularity, vtx_count)
            
        #Create new graph
        self.graph = graph_create.watts_strogatz(vtx_count,regularity, beta,
                close_trans_prob, far_trans_prob, time_alloc)
        
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

