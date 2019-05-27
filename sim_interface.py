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
        
        self.vtx_widget = widgets.IntText(value=2,
                                    description='Vertex Count (Rounded to Even)',
                                    disabled=False)
        self.reg_widget = widgets.IntText(value=1,
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
        
    def render_visuals(self):
        display(self.vtx_widget)
        display(self.reg_widget)
        display(self.trans_prob_widget)
        display(self.time_alloc_widget)
        
        display(self.update_button)
        iplot(self.plot_fig)
        
    def update_graph(self, button):
        vtx_count = self.vtx_widget.value
        regularity = self.reg_widget.value
        trans_prob = self.trans_prob_widget.value
        time_alloc = self.time_alloc_widget.value
        
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
        G_s, G_utils = simulation.run_simulation(self.graph)
        self.plot_fig = gen_vis.animate_simulation(self.plot_fig, G_utils)
        num_steps = len(self.plot_fig['frames'])
        clear_output()
        self.render_visuals()
        print("Diameter: {0}".format(diameter))
        print("Step Count: {0}".format(num_steps))

