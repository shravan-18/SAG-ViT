import unittest
import torch
import networkx as nx
from graph_construction import extract_patches, build_graph_from_patches, build_graph_data_from_patches

class TestGraphConstruction(unittest.TestCase):
    def test_extract_patches_shape(self):
        # Create a dummy feature map: B=2, C=16, H=32, W=32
        feature_map = torch.randn(2, 16, 32, 32)
        patches = extract_patches(feature_map, patch_size=(4,4))
        # Check dimensions: after extraction, 
        # number_of_patches = (H/4)*(W/4) = 8*8=64 per image, total 2*64=128
        self.assertEqual(patches.shape, (2, 64, 16, 4, 4))

    def test_build_graph_from_patches_graph_structure(self):
        feature_map = torch.randn(1, 16, 32, 32)
        G_batch, patches = build_graph_from_patches(feature_map, patch_size=(4,4))
        # 1 image => G_batch[0] is the graph
        G = G_batch[0]
        # We have 64 patches
        self.assertEqual(len(G.nodes), 64)
        # Check if edges exist (8-neighborhood). 
        # Interior nodes should have edges to neighbors.
        # Just check a random node in the middle
        node_index = 9 # assuming row=1, col=1 in an 8x8 grid
        self.assertTrue(len(list(G.neighbors(node_index))) > 0)

    def test_build_graph_data_from_patches_conversion(self):
        feature_map = torch.randn(2, 16, 32, 32)
        G_batch, patches = build_graph_from_patches(feature_map, patch_size=(4,4))
        data_list = build_graph_data_from_patches(G_batch, patches)
        self.assertEqual(len(data_list), 2)
        # Check node feature shape
        self.assertEqual(data_list[0].x.shape[1], 16*4*4)  # C * patch_h * patch_w = 16*4*4=256
        # Check edges are present
        self.assertTrue(data_list[0].edge_index.shape[1] > 0)

if __name__ == '__main__':
    unittest.main()
