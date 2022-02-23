"""
Module for importing graphs
"""

import os

import numpy as np
import networkx as nx


class GraphDataset:
	"""
	Class for a set of graphs
	"""

	def __init__(self, path_to_graphs, path_to_groundtruth_ref, suffix=".gpickle"):
		"""
		Constructor

		:param str path_to_graphs: The path ot the directory with all the graphs
		:param str path_to_groundtruth_ref: The path to the file with the ground truth
		:param str suffix: the suffix of the graph files
		"""
		g_files = []
		with os.scandir(path_to_graphs) as files:
			for file in files:
				if file.name.endswith(suffix):
					g_files.append(file.name)
		g_files.sort()

		self.list_graphs = [nx.read_gpickle(path_to_graphs+"/"+graph) for graph in g_files]
		self.sizes = [nx.number_of_nodes(g) for g in self.list_graphs]
		self.nodes = [list(graph.nodes()) for graph in self.list_graphs]
		self.labels = nx.read_gpickle(path_to_groundtruth_ref)
