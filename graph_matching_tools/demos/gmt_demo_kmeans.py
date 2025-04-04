"""
Example script with KMeans to be used with INT graphs.
"""

import os
import argparse
import pickle

import numpy as np
import networkx as nx

import graph_matching_tools.metrics.matching as matching
from graph_matching_tools.io.graph_dataset import GraphDataset
import graph_matching_tools.algorithms.multiway.kmeans as kmeans
from graph_matching_tools.utils.permutations import (
    get_permutation_matrix_from_dictionary,
)


def get_all_coords(list_graphs: list[nx.Graph]) -> np.ndarray:
    """Get the coordinates from the graphs.

    :param list[nx.Graph] list_graphs: the list of graph.
    :return: the array of coordinates.
    :rtype: np.ndarray
    """
    g_all_coords = []
    for g in list_graphs:
        coords = np.array(list(nx.get_node_attributes(g, "coord").values()))
        g_all_coords.extend(coords)
    g_all_coords = np.array(g_all_coords)
    return g_all_coords


def main() -> None:
    parser = argparse.ArgumentParser(
        description="K-Mean node matching applied to different sets of graphs."
    )
    parser.add_argument(
        "directory",
        help="The directory with the different sets of graphs",
        type=str,
    )
    parser.add_argument(
        "-k",
        "--kmeans_cluster",
        help="The number of clusters seeks by KMean",
        type=int,
        default=110,
    )
    parser.add_argument(
        "--ground_truth_name",
        help="The name of the ground truth file",
        type=str,
        default="ground_truth.pkl",
    )
    parser.add_argument(
        "--graph_directory_name",
        help="The name of the directory with the graphs",
        type=str,
        default="graphs",
    )
    parser.add_argument(
        "--save_results",
        help="Save the results into a file",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    folders = np.sort(os.listdir(args.directory))
    scores = dict()

    for folder in folders:
        print("folder: ", folder)

        path_to_graphs = (
            args.directory + "/" + folder + "/" + args.graph_directory_name + "/"
        )
        path_to_groundtruth = (
            args.directory + "/" + folder + "/" + args.ground_truth_name
        )

        graph_meta = GraphDataset(path_to_graphs, path_to_groundtruth)
        all_coords = get_all_coords(graph_meta.list_graphs)
        res = get_permutation_matrix_from_dictionary(
            graph_meta.node_references, graph_meta.sizes
        )

        p = kmeans.get_permutation_with_kmeans(args.kmeans_cluster, all_coords)
        f1, prec, rec = matching.compute_f1score(p, res)
        scores[f"{folder}"] = f1

    if args.save_results:
        with open("kmeans_score_k_" + str(args.kmeans_cluster) + ".pkl", "wb") as f:
            pickle.dump(scores, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(scores)


if __name__ == "__main__":
    main()
