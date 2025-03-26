from unittest import TestCase

import graph_matching_tools.generators.reference_graph as reference_graph


class Test(TestCase):
    def test_generate_reference_graph(self):
        graph = reference_graph.generate_reference_graph(10, 1.0)
        self.assertEqual(graph.number_of_nodes(), 10)

    def test_generate_reference_graph_with_biggest_minimal_geodesic_distance(self):
        graph = reference_graph.generated_max_geodesic_reference_graph(10, 1.0)
        self.assertEqual(graph.number_of_nodes(), 10)
