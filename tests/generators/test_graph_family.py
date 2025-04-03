from unittest import TestCase

import graph_matching_tools.generators.reference_graph as reference_graph
import graph_matching_tools.generators.graph_family as graph_family


class Test(TestCase):
    def test_generation_graph_family(self):
        graph = reference_graph.generate_reference_graph(25, 100.0)
        all_graph = graph_family.generation_graph_family(10, graph, 1.0, 10.0)
        self.assertEqual(len(all_graph), 10)
        self.assertGreater(all_graph[0].number_of_nodes(), 0)
