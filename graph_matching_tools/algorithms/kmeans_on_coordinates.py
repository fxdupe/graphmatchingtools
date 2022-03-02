import os
import sys
from sklearn.cluster import KMeans
import networkx as nx
import numpy as np
from graph_matching_tools.metrics import matching
from graph_matching_tools.io.graph_dataset import GraphDataset
import matplotlib.pyplot as plt




def get_permutation_matrix_from_dictionary(matching, g_sizes):
    """
    Create the full permutation matrix from the matching result
    :param matching: the matching result for each graph (nodes number, assignment)
    :param g_sizes: the list of the size of the different graph
    :return: the full permutation matrix
    """
    f_size = int(np.sum(g_sizes))
    res = np.zeros((f_size, f_size))

    idx1 = 0
    for i_g1 in range(len(g_sizes)):
        idx2 = 0
        for i_g2 in range(len(g_sizes)):
            match = matching["{},{}".format(i_g1, i_g2)]
            for k in match:
                res[idx1 + int(k), idx2 + match[k]] = 1
            idx2 += g_sizes[i_g2]
        idx1 += g_sizes[i_g1]
        
    np.fill_diagonal(res,1)
    return res




def get_all_coords(list_graphs):
    all_coords = []
    for g in list_graphs:
        coords = np.array(list(nx.get_node_attributes(g,'coord').values()))
        all_coords.extend(coords)
    all_coords = np.array(all_coords)
    
    return all_coords




def create_perm_from_labels(labels):
    U = np.zeros((len(labels),len(set(labels))))
    
    for node,label in zip(range(U.shape[0]),labels):
        U[node,label] = 1
        
    return U @ U.T



def get_labels_from_k_means(k, coords):
    
    kmeans = KMeans(n_clusters=k, random_state=0).fit(coords)
    
    return kmeans.labels_



def score_mean_std(scores):
    
    avg_scores = []
    std_scores = []

    for keys,values in scores.items():
        avg_scores.append(np.mean(values))
        std_scores.append(np.std(values))
        
    return np.array(avg_scores), np.array(std_scores)



if __name__ == '__main__':

	path_to_graph_folder = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/simu_graph/simu_test/'


	trials = np.sort(os.listdir(path_to_graph_folder))

	k = 110

	scores = {100:[],400:[],700:[],1000:[],1300:[]}

	for trial in trials:
	    print('trial: ', trial)
	    
	    all_files = os.listdir(path_to_graph_folder+trial)
	    
	    for folder in all_files:
	        
	        if os.path.isdir(path_to_graph_folder+trial+'/'+ folder):
	            
	            path_to_graphs = path_to_graph_folder + '/' + trial + '/' + folder+'/graphs/'
	            path_to_groundtruth_ref = path_to_graph_folder + '/' + trial +'/' + folder + '/permutation_to_ref_graph.gpickle'
	            path_to_groundtruth  = path_to_graph_folder + '/' + trial + '/' + folder + '/ground_truth.gpickle'
	            
	            
	            noise = folder.split(',')[0].split('_')[1]
	            

	            graph_meta = GraphDataset(path_to_graphs, path_to_groundtruth_ref)
	            all_coords = get_all_coords(graph_meta.list_graphs)
	            ground_truth =  nx.read_gpickle(path_to_groundtruth)   
	            res = get_permutation_matrix_from_dictionary(ground_truth, graph_meta.sizes)
	               
	            kmeans_labels = get_labels_from_k_means(k, all_coords)
	                
	            P = create_perm_from_labels(kmeans_labels)
	                
	            f1, prec, rec = matching.compute_f1score(P,res)
	            
	            scores[int(noise)].append(f1)

	nx.write_gpickle(scores,'kmeans_score_k_'+ str(k) +'.gpickle')

	print(scores)