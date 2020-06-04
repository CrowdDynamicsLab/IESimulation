# Information Elicitation Simulations

# Running the notebook
* First generate the data using `gen_kgrid_mwu_data.py` which dumps data into a directory called `data` (create this directory if it does not already exist). This takes me about half an hour to run.
* Run the `KGridMWU.ipynb` notebook to see charts from generated data

# Neighbor selection strategies
* mwu : Hedge variant of multiplicative weight update
* se_mwu : MWU with self edges (currently not implemented as Strategy object but will be soon)
* unif : Select from available neighbors with uniform prob
* rr : Round robin selection of neighbors

# Structure
graph.py contains graph implementation

simualtion.py contains code to run actual simulations over graph
