"""Example of using graph generation tools.

.. moduleauthor:: François-Xavier Dupé
"""

import argparse
import pickle
import json
import os

import networkx as nx
import matplotlib.pyplot as plt

import graph_matching_tools.generators.graph_family as graph_family
import graph_matching_tools.generators.reference_graph as reference_graph


def visualisation_info(
    ref_graph: nx.Graph, noisy_graphs: list[nx.Graph], graph_index: int = 0
) -> None:
    """Display some information about the generated graphs.

    :param nx.Graph ref_graph: the reference graph.
    :param list[nx.Graph] noisy_graphs: the list of generated graphs.
    :param int graph_index: the index of the generated graph (used to display some nodes/edges information).
    """
    plt.figure(1)
    nx.draw(ref_graph, with_labels=True)
    plt.figure(2)
    nx.draw(noisy_graphs[graph_index], with_labels=True)
    plt.show()

    for node, node_data in noisy_graphs[graph_index].nodes.items():
        print(node_data["coord"])
        print(node_data["label"])
        print(node_data["is_outlier"])

    for edge, edge_data in noisy_graphs[graph_index].edges.items():
        print(edge_data["geodesic_distance"])
        print(edge_data["id"])


def save_graphs(
    output_dir: str, ref_graph: nx.Graph, noisy_graphs: list[nx.Graph], parameters: dict
) -> None:
    """Save the generated graphs into directories.

    :param str output_dir: the output directory.
    :param nx.Graph ref_graph: the reference graph.
    :param list[nx.Graph] noisy_graphs: the list of generated noisy graphs.
    :param dict parameters: the parameters of the generator.
    """
    # Check the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save parameters in JSON format
    with open(os.path.join(output_dir, "parameters.json"), "w") as f:
        json.dump(parameters, f)

    # Check the repertory for the graphs
    if not os.path.exists(output_dir + "/generated_graphs"):
        os.makedirs(output_dir + "/generated_graphs")

    # Save reference graph
    with open(
        os.path.join(output_dir + "/generated_graphs", "graph_{:05d}.pkl".format(0)),
        "wb",
    ) as f:
        pickle.dump(ref_graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the noisy graphs
    for noisy_graph_idx in range(len(noisy_graphs)):
        file_name = "graph_{:05d}.pkl".format(noisy_graph_idx + 1)
        with open(os.path.join(output_dir + "/generated_graphs", file_name), "wb") as f:
            pickle.dump(
                noisy_graphs[noisy_graph_idx], f, protocol=pickle.HIGHEST_PROTOCOL
            )

    # Compute and save the permutation
    permutation_dictionary = build_permutation_dictionary(
        [
            ref_graph,
        ]
        + noisy_graphs
    )
    with open(os.path.join(output_dir, "ground_truth.pkl"), "wb") as f:
        pickle.dump(permutation_dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_permutation_dictionary(source_graph: nx.Graph, target_graph: nx.Graph) -> dict:
    """Get the permutation dictionary between two graphs.

    :param nx.Graph source_graph: the source graph.
    :param nx.Graph target_graph: the target graph.
    :return: the dictionary.
    """
    permutation_dictionary = dict()
    for node, node_data in source_graph.nodes.items():
        if node_data["is_outlier"]:
            continue

        for target_node, target_node_data in target_graph.nodes.items():
            if target_node_data["is_outlier"]:
                continue
            if node_data["label"] == target_node_data["label"]:
                permutation_dictionary[node] = target_node
                break

    return permutation_dictionary


def build_permutation_dictionary(graphs: list[nx.Graph]) -> dict:
    """Build the permutation information

    :param list[nx.Graph] graphs: the list of graphs.
    :return: the dictionary with the permutation information.
    :rtype: dict
    """
    full_permutation_dictionary = dict()

    for idx_s in range(len(graphs)):
        for idx_t in range(len(graphs)):
            permutation_dictionary = get_permutation_dictionary(
                graphs[idx_s], graphs[idx_t]
            )
            full_permutation_dictionary[f"{idx_s},{idx_t}"] = permutation_dictionary

    return full_permutation_dictionary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulated sucal pits graph generation."
    )
    parser.add_argument(
        "--sample_number", help="The number of generated graphs", type=int, default=10
    )
    parser.add_argument(
        "--node_number", help="The number of nodes", type=int, default=100
    )
    parser.add_argument(
        "--outliers_mu",
        help="The mean number of outliers/suppressed nodes",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--outliers_sigma",
        help="The variance number of outliers/suppressed nodes",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--percent_remove_edges",
        help="The percentage of removed edges",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--coord_noise_kappa",
        help="The variance of the noise distribution on coordinates",
        type=float,
        default=200.0,
    )
    parser.add_argument(
        "--add_outliers",
        help="Add outliers in the generated graphs",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--suppress_nodes",
        help="Suppress nodes when generating graphs",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--display",
        help="Display and plot some information",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save",
        help="Save the generated graphs into files",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--output_dir",
        help="The output directory for saving",
        type=str,
        default="generated_outputs",
    )
    p_args = parser.parse_args()

    graph_reference = reference_graph.generate_reference_graph(p_args.node_number, 1.0)
    generated_noisy_graph = graph_family.generation_graph_family(
        p_args.sample_number,
        graph_reference,
        kappa_noise_node=p_args.coord_noise_kappa,
        outlier_mu=p_args.outliers_mu,
        outlier_sigma=p_args.outliers_sigma,
        edge_delete_percent=p_args.percent_remove_edges,
        add_outliers=p_args.add_outliers,
        suppress_nodes=p_args.suppress_nodes,
    )

    if p_args.display:
        visualisation_info(graph_reference, generated_noisy_graph)

    if p_args.save:
        current_parameters = dict()
        current_parameters["node_number"] = p_args.node_number
        current_parameters["kappa_noise_node"] = p_args.coord_noise_kappa
        current_parameters["outlier_mu"] = p_args.outliers_mu
        current_parameters["outlier_sigma"] = p_args.outliers_sigma
        current_parameters["edge_delete_percent"] = p_args.percent_remove_edges
        current_parameters["add_outliers"] = p_args.add_outliers
        current_parameters["suppress_nodes"] = p_args.suppress_nodes

        save_graphs(
            p_args.output_dir,
            graph_reference,
            generated_noisy_graph,
            current_parameters,
        )


if __name__ == "__main__":
    main()
