from unittest import TestCase

import graph_matching_tools.io.graph_dataset as gd


class TestGraphDataset(TestCase):
    def test_init(self):
        data = gd.GraphDataset(
            "data/cortex_simulation/graphs",
            "data/cortex_simulation/ground_truth.gpickle",
        )
        self.assertEqual(len(data.sizes), 10)
