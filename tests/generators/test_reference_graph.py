from unittest import TestCase

import graph_matching_tools.generators.reference_graph as reference_graph


class Test(TestCase):
    def test_generate_reference_graph(self):
        graph = reference_graph.generate_reference_graph(10, 1.0)
        self.assertEqual(graph.number_of_nodes(), 10)
