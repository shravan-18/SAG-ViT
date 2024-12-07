import unittest
import torch
from model_components import InceptionV3FeatureExtractor, GATGNN, TransformerEncoder, MLPBlock
from torch_geometric.data import Data

class TestModelComponents(unittest.TestCase):
    def test_inception_extractor_output_shape(self):
        model = InceptionV3FeatureExtractor()
        model.eval()
        x = torch.randn(2, 3, 299, 299)
        with torch.no_grad():
            features = model(x)
        # Check output shape - depends on inception intermediate layer
        # Example: shape could be (2, 768, 8, 8) depending on the chosen layer
        self.assertEqual(features.size(0), 2)
        self.assertTrue(features.size(1) > 0)
        self.assertTrue(features.size(2) > 0)
        self.assertTrue(features.size(3) > 0)

    def test_gatgnn_forward(self):
        # Graph with 4 nodes, each node feature dim=256
        x = torch.randn(4, 256)
        edge_index = torch.tensor([[0,1,1,2],[1,0,2,3]], dtype=torch.long)
        batch = torch.tensor([0,0,0,0])
        data = Data(x=x, edge_index=edge_index, batch=batch)
        
        gnn = GATGNN(in_channels=256, hidden_channels=64, out_channels=32)
        output = gnn(data)
        # After pooling: should be (batch_size, out_channels) = (1,32)
        self.assertEqual(output.shape, (1, 32))

    def test_transformer_encoder(self):
        # (B, N, D) = (2, 10, 64)
        x = torch.randn(2, 10, 64)
        encoder = TransformerEncoder(d_model=64, nhead=8, num_layers=2, dim_feedforward=128)
        out = encoder(x)
        # same shape as input
        self.assertEqual(out.shape, (2, 10, 64))

    def test_mlp_block(self):
        mlp = MLPBlock(in_features=64, hidden_features=128, out_features=10)
        x = torch.randn(2, 64)
        out = mlp(x)
        self.assertEqual(out.shape, (2,10))

    def test_inception_freeze(self):
        # Ensure params are frozen
        model = InceptionV3FeatureExtractor()
        for param in model.parameters():
            self.assertFalse(param.requires_grad)

if __name__ == '__main__':
    unittest.main()
