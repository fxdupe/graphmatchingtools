'''
Dirty script

Creation of full affinity matrix with or without dummy.

Requires proper organisation.

'''

import numpy as np
import scipy.io as sio
import networkx as nx
import os
from graph_matching_tools.algorithms.kernels.gaussian import create_gaussian_node_kernel
from graph_matching_tools.algorithms.kernels.utils import create_full_node_affinity_matrix


if __name__ == '__main__':


	path_to_dummy_graphs_folder = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/simu_graph/test_with_dummy/'


	trials = np.sort(os.listdir(path_to_dummy_graphs_folder))

	sigma = 200

	for trial in trials:
		
		if float(trial) >= 0.0:
			
			print('trial: ', trial)

			all_files = os.listdir(path_to_dummy_graphs_folder+trial)

			for folder in all_files:

				if os.path.isdir(path_to_dummy_graphs_folder+trial+'/'+ folder):

					print('Noise folder: ',folder)

					path_to_dummy_graphs = path_to_dummy_graphs_folder + '/' + trial + '/' + folder+'/0/graphs/'

		
					graphs = [nx.read_gpickle(path_to_dummy_graphs + graph) for graph in np.sort(os.listdir(path_to_dummy_graphs))]
					sizes = [nx.number_of_nodes(graph) for graph in graphs]
				
					full_affinity = {}
					node_kernel = create_gaussian_node_kernel(sigma,'coord')
					full_affinity['full_affinity'] = create_full_node_affinity_matrix(graphs, node_kernel)
					
					sio.savemat(path_to_dummy_graphs_folder + '/' + trial + '/' + '/' + folder + '/0/full_affinity.mat',full_affinity)
