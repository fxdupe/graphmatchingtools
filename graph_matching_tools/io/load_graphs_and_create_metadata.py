import os
import sys
sys.path.append("/home/rohit/PhD_Work/GM_my_version/graphmatchingtools/")
import argparse
import numpy as np
import networkx as nx
import slam.plot as splt
import slam.topology as stop
import slam.generate_parametric_surfaces as sgps
import trimesh
import os
import tools.graph_processing as gp
from sphere import *
from tqdm.auto import tqdm,trange
from scipy.stats import betabinom
import random



class dataset_metadata():

	def __init__(self,path_to_graphs,path_to_groundtruth_ref):

		self.list_graphs = [nx.read_gpickle(path_to_graphs+'/'+graph) for graph in np.sort(os.listdir(path_to_graphs))]

		self.sizes = [nx.number_of_nodes(g) for g in self.list_graphs]

		self.nodes = [list(graph.nodes()) for graph in self.list_graphs]

		self.labels = nx.read_gpickle(path_to_groundtruth_ref)
