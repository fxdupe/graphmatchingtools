from unittest import TestCase

import graph_matching_tools.generators.noisy_graph as noisy_graph
import graph_matching_tools.generators.reference_graph as reference_graph


class Test(TestCase):
    def test_noisy_graph_generation(self):
        graph = reference_graph.generate_reference_graph(25, 100.0)
        perturbed_graph = noisy_graph.noisy_graph_generation(
            graph, nb_vertices=25, sigma_noise_nodes=0.1
        )
        self.assertEqual(perturbed_graph.number_of_nodes(), 25)
