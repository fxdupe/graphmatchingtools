"""
Graph Auto-Encoder as proposed by Zipf et Welling in "Variational Graph Auto-Encoders", Bayesian Deep Learning
 Workshop (NIPS 2016), https://arxiv.org/abs/1611.07308

.. moduleauthor:: François-Xavier Dupé
"""

import copy

import numpy as np
import networkx as nx
import torch
import torch_geometric as tgeo
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as func


class GAE(torch.nn.Module):
    """
    GAE model
    """

    def __init__(self, num_features: int, hidden_channels: int, output_dim: int):
        """Constructor.

        :param int num_features: the dimension of the data on nodes.
        :param int hidden_channels: the size of the intermediate layer.
        :param int output_dim: the output dimension.
        """
        super().__init__()
        # torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_dim)

    def forward(self, graph: tgeo.data.Data) -> torch.Tensor:
        """Forward operation.

        :param graph: the input graph.
        :return: the output data from the model application.
        """
        x = self.conv1(graph.x, graph.edge_index)
        x = x.relu()
        x = func.dropout(x, p=0.2)
        x = self.conv2(x, graph.edge_index)
        x = func.softmax(x, dim=1)
        return x


def test_model(model: torch.nn.Module, dataloader: tgeo.data.DataLoader) -> float:
    """Test a model on data

    :param torch.nn.Module model: the model to test.
    :param tgeo.data.DataLoader dataloader: the loader of the data.
    :return: the value of the loss function on the set of data.
    :rtype: float
    """
    model.eval()
    loss = 0

    # noinspection PyTypeChecker
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

    # noinspection PyTypeChecker
    return loss / len(dataloader)


def update_graphs(graph_list: list[nx.Graph], model: torch.nn.Module) -> list[nx.Graph]:
    """Update graph value on nodes.

    :param list[nx.Graph] graph_list: list of graphs.
    :param torch.nn.Module model: the model.
    :return: the graphs with the new data on nodes.
    :rtype: list[nx.Graph]
    """
    new_g = copy.deepcopy(graph_list)

    for graph in new_g:
        data = tgeo.utils.from_networkx(graph, ["x"], None)
        new_representation = model(data)
        for i_n in range(nx.number_of_nodes(graph)):
            graph.nodes[i_n]["x"] = np.squeeze(
                new_representation[i_n, :].detach().numpy()
            )

    return new_g


def gae_learning(
    dataset: list[nx.Graph],
    input_dim: int,
    data_name: str,
    epochs: int = 1000,
    batch_size: int = 10,
    learning_rate: float = 1e-5,
) -> torch.nn.Module:
    """GAE learning procedure.

    :param list[nx.Graph] dataset: the data.
    :param int input_dim: the size of the node data.
    :param str data_name: the name of the data vector.
    :param int epochs: the number of epochs.
    :param int batch_size: the size of one batch.
    :param float learning_rate: the learning rate.
    :return: the learned model.
    :rtype: torch.nn.Module
    """
    device_d = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_d)

    data_t = list()
    for graph in dataset:
        data = tgeo.utils.from_networkx(graph, [data_name], None)
        data.to(device_d)
        data_t.append(data)

    split_len = int(len(data_t) * 0.85)
    train_loader = DataLoader(data_t[0:split_len], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_t[split_len:], batch_size=batch_size)

    # Define model and optimization environment
    model = GAE(input_dim, input_dim // 4, input_dim // 8)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
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
