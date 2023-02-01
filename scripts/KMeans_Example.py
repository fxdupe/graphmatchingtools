"""
Example script with KMeans
"""
import os
import argparse

import numpy as np
import networkx as nx

import graph_matching_tools.metrics.matching as matching
from graph_matching_tools.io.graph_dataset import GraphDataset
import graph_matching_tools.algorithms.multiway.kmeans as kmeans
from graph_matching_tools.utils.permutations import (
    get_permutation_matrix_from_dictionary,
)


def get_all_coords(list_graphs):
    """
    Get the coordinates from the graphs
    :param list_graphs: the list of graph
    :return: the array of coordinates
    """
    g_all_coords = []
    for g in list_graphs:
        coords = np.array(list(nx.get_node_attributes(g, "coord").values()))
        g_all_coords.extend(coords)
    g_all_coords = np.array(g_all_coords)
    return g_all_coords


if __name__ == "__main__":
    default_to_graph_folder = (
        "/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/simu_graph/simu_test/"
    )
    parser = argparse.ArgumentParser(description="Segmentation from learning method.")
    parser.add_argument(
        "-g",
        "--graph_dir",
        help="The directory with the graph",
        type=str,
        default=default_to_graph_folder,
    )
    parser.add_argument(
        "-k",
        "--kmeans_cluster",
        help="The number of clusters seeks by KMean",
        type=int,
        default=110,
    )
    args = parser.parse_args()
    path_to_graph_folder = args.graph_dir
    k = args.kmeans_cluster

    trials = np.sort(os.listdir(path_to_graph_folder))

    scores = {100: [], 400: [], 700: [], 1000: [], 1300: []}

    for trial in trials:
        print("trial: ", trial)

        all_files = os.listdir(path_to_graph_folder + trial)

        for folder in all_files:
            if os.path.isdir(path_to_graph_folder + trial + "/" + folder):
                path_to_graphs = (
                    path_to_graph_folder + "/" + trial + "/" + folder + "/graphs/"
                )
                path_to_groundtruth_ref = (
                    path_to_graph_folder
                    + "/"
                    + trial
                    + "/"
                    + folder
                    + "/permutation_to_ref_graph.gpickle"
                )
                path_to_groundtruth = (
                    path_to_graph_folder
                    + "/"
                    + trial
                    + "/"
                    + folder
                    + "/ground_truth.gpickle"
                )

                noise = folder.split(",")[0].split("_")[1]

                graph_meta = GraphDataset(path_to_graphs, path_to_groundtruth_ref)
                all_coords = get_all_coords(graph_meta.list_graphs)
                ground_truth = nx.read_gpickle(path_to_groundtruth)
                res = get_permutation_matrix_from_dictionary(
                    ground_truth, graph_meta.sizes
                )

                P = kmeans.get_permutation_with_kmeans(k, all_coords)

                f1, prec, rec = matching.compute_f1score(P, res)

                scores[int(noise)].append(f1)

    nx.write_gpickle(scores, "kmeans_score_k_" + str(k) + ".gpickle")

    print(scores)
