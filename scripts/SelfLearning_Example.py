"""
Example of matching with self learning (for improving the features)

.. moduleauthor: François-Xavier Dupé
"""
import argparse
import copy

import numpy as np
import networkx as nx
import torch
import torch_geometric as tgeo
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as func

import graph_matching_tools.algorithms.kernels.utils as kutils
import graph_matching_tools.algorithms.kernels.gaussian as gaussian
import graph_matching_tools.algorithms.kernels.rff as rff
import graph_matching_tools.algorithms.multiway.mkergm as mkergm
import graph_matching_tools.algorithms.multiway.matcheig as matcheig
import graph_matching_tools.utils.utils as utils
import graph_matching_tools.metrics.matching as measures
import graph_matching_tools.io.pygeo_graphs as pyg
import graph_matching_tools.utils.permutations as perm


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, output_dim):
        super().__init__()
        # torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_dim)

    def forward(self, graph):
        noise = torch.randn(graph.x.shape) * 0.1
        x = self.conv1(graph.x + noise, graph.edge_index)
        x = x.relu()
        x = func.dropout(x, p=0.1)
        x = self.conv2(x, graph.edge_index)
        x = func.softmax(x, dim=1)
        return x


def test_model(model, dataloader):
    model.eval()
    loss = 0
    for batch, sample in enumerate(dataloader):
        out = model(sample)  # Perform a single forward pass.
        adj_tensor = torch.squeeze(
            tgeo.utils.to_dense_adj(sample.edge_index, batch=sample.batch)
        )
        for graph in range(sample.batch[-1]):
            embedding = out[sample.batch == graph, :]
            loss += torch.nn.functional.mse_loss(
                embedding @ embedding.T,
                adj_tensor[graph, :, :],
            )  # Compute the loss solely based on the training nodes.

    return loss / len(dataloader)


def update_graphs(graphs, model):
    new_graphs = copy.deepcopy(graphs)

    for graph in new_graphs:
        data = tgeo.utils.from_networkx(graph, ["x"], None)
        new_representation = model(data)
        for i_n in range(nx.number_of_nodes(graph)):
            graph.nodes[i_n]["x"] = np.squeeze(
                new_representation[i_n, :].detach().numpy()
            )

    return new_graphs


def run_learning(dataset):
    device_d = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_d)

    data_t = list()
    for graph in dataset:
        data = tgeo.utils.from_networkx(graph, ["x"], None)
        data.to(device_d)
        data_t.append(data)

    train_loader = DataLoader(data_t[0:30], batch_size=10, shuffle=True)
    test_loader = DataLoader(data_t[30:], batch_size=5)

    # Define model and optimization environment
    model = GCN(1024, 256, 64)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)  # , weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(500):
        # noinspection PyTypeChecker
        for batch, sample in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()  # Clear gradients.
            out = model(sample)  # Perform a single forward pass.
            adj_tensor = torch.squeeze(
                tgeo.utils.to_dense_adj(sample.edge_index, batch=sample.batch)
            )
            loss = None
            for graph in range(sample.batch[-1]):
                embedding = out[sample.batch == graph, :]
                loss = criterion(
                    embedding @ embedding.T,
                    adj_tensor[graph, :, :],
                )  # Compute the loss solely based on the training nodes.

            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.

        # noinspection PyTypeChecker
        print(
            "Epoch {}, mse train = {}, mse test = {}".format(
                epoch, test_model(model, train_loader), test_model(model, test_loader)
            )
        )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--isotropic", help="Build isotropic graphs", action="store_true", default=False
    )
    parser.add_argument(
        "--sigma", help="The sigma parameter (nodes)", type=float, default=200.0
    )
    parser.add_argument(
        "--gamma", help="The gamma parameter (edges)", type=float, default=0.1
    )
    parser.add_argument(
        "--rff",
        help="The number of Random Fourier Features (edges)",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--rank", help="The maximal rank of the results", type=int, default=10
    )
    parser.add_argument(
        "--node_factor",
        help="The multiplication factor for node affinity.",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--tolerance", help="The tolerance for convergence", type=float, default=1e-3
    )
    parser.add_argument(
        "--iterations", help="The maximal number of iterations", type=int, default=20
    )
    parser.add_argument(
        "--database",
        help="The graph database",
        type=str,
        choices=["Willow", "PascalVOC", "PascalPF"],
        default="Willow",
    )
    parser.add_argument(
        "--category", help="The category inside the database", type=str, default="car"
    )
    parser.add_argument(
        "--repo",
        help="The repository for downloaded data",
        type=str,
        default="/home/fx/Projets/Data/Graphes/pytorch",
    )
    parser.add_argument(
        "--random",
        help="Do a random sampling on the of graphs",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--shuffle", help="Shuffle the graphs", action="store_true", default=False
    )
    parser.add_argument(
        "--entropy",
        help="The entropy for S-T method",
        default=2.0,
        type=float,
    )
    parser.add_argument(
        "--add_dummy", help="Add dummy nodes", action="store_true", default=False
    )
    parser.add_argument(
        "--proj_method",
        help="Projection method",
        default="matcheig",
        type=str,
        choices=["matcheig", "msync", "irgcl", "gpow"],
    )
    parser.add_argument(
        "--load_model", help="The learned model to load", default=None, type=str
    )
    parser.add_argument(
        "--model_name",
        help="The filename for the model weights",
        default=None,
        type=str,
    )
    args = parser.parse_args()

    all_graphs = pyg.get_graph_database(
        args.database, args.isotropic, args.category, args.repo + "/" + args.database
    )

    print("Size dataset: {} graphs".format(len(all_graphs)))
    print([nx.number_of_nodes(g) for g in all_graphs])
    all_graphs = [g for g in all_graphs]

    dummy_index = []
    graph_index = []

    if args.shuffle:
        all_graphs, graph_index = utils.randomize_nodes_position(all_graphs)

    if not args.add_dummy and not args.shuffle:
        graph_index = [list(range(nx.number_of_nodes(g))) for g in all_graphs]

    g_sizes = [nx.number_of_nodes(g) for g in all_graphs]
    full_size = int(np.sum(g_sizes))
    print("Full-size: {} nodes".format(full_size))

    gnn_model = None
    if args.load_model is None:
        gnn_model = run_learning(all_graphs)
        if args.model_name is not None:
            torch.save(gnn_model.state_dict(), args.model_name)
    elif args.model_name is not None:
        gnn_model = GCN(1024, 256, 64)
        # noinspection PyArgumentList
        gnn_model.state_dict(torch.load(args.model_name, map_location="cpu"))

    new_graphs = all_graphs
    if gnn_model is not None:
        gnn_model.eval()
        new_graphs = update_graphs(all_graphs, gnn_model)

    if args.add_dummy:
        dim = new_graphs[0].nodes[0]["x"].shape[0]
        all_graphs, graph_index, dummy_index = pyg.add_dummy_nodes(
            all_graphs, args.rank, dimension=dim
        )

    node_kernel = gaussian.create_gaussian_node_kernel(
        args.sigma, "pos" if args.database == "PascalPF" else "x"
    )
    knode = kutils.create_full_node_affinity_matrix(new_graphs, node_kernel)

    import matplotlib.pyplot as plt

    plt.imshow(knode)
    plt.colorbar()
    plt.show()

    # Compute the big phi matrix
    vectors, offsets = rff.create_random_vectors(1, args.rff, args.gamma)
    phi = np.zeros((args.rff, full_size, full_size))
    index = 0
    for i in range(len(new_graphs)):
        g_phi = rff.compute_phi(new_graphs[i], "weight", vectors, offsets)
        phi[:, index : index + g_sizes[i], index : index + g_sizes[i]] = g_phi
        index += g_sizes[i]

    x_init = np.ones(knode.shape) / knode.shape[0]

    # Set the tradeoff between nodes and edges
    if args.node_factor is None:
        knode *= args.rank
    else:
        knode *= args.node_factor

    gradient = mkergm.create_gradient(phi, knode)

    m_res = mkergm.mkergm(
        gradient,
        g_sizes,
        args.rank,
        iterations=args.iterations,
        init=x_init,
        tolerance=args.tolerance,
        projection_method=args.proj_method,
    )
    # m_res = matcheig.matcheig(knode, args.rank, g_sizes)

    # Compare with groundtruth
    truth = pyg.generate_groundtruth(g_sizes, full_size, graph_index)
    a_truth = perm.get_permutation_matrix_from_matching(truth, g_sizes)

    # Remove dummy nodes before computing scores
    if dummy_index:
        index = 0
        for i_g in range(len(all_graphs)):
            for i_d in dummy_index[i_g]:
                t_i_d = graph_index[i_g][i_d]
                m_res[index + t_i_d, :] = 0
                m_res[:, index + t_i_d] = 0
                a_truth[:, index + t_i_d] = 0
                a_truth[index + t_i_d, :] = 0
            index += g_sizes[i_g]

    f1_score, precision, recall = measures.compute_f1score(m_res, a_truth)
    print("Precision = {:.3f}".format(precision))
    print("Recall = {:.3f}".format(recall))
    print("F1-Score = {:.3f}".format(f1_score))
