# Information Elicitation Simulations

# Usage
Use InfoElicitationSim.ipynb to run visualization
If plot does not render, try restarting Jupyter Notebook

# Structure
gen_plot.py contains visualization code

graph.py contains graph implementation and graph analysis functions

simualtion.py contains code to run actual simulations over graph

sim_interface.py contains code to generate IPython interface

Should be noted that simulations may be destructive on graph structure

# TODO
* More interesting simulation parameters
* More interesting graph parameters

# Simulation TODOs
* Add ability to create multiple cliques
* Add ability to vary transmission probabilities
    * Currently probability == given rate
* Add more interesting time allocation schemes. Current:
    * Dirichlet
    * Even
* Different schemes of communication continuation
