import unittest
import torch
from sag_vit_model import SAGViTClassifier


class TestSAGViTModel(unittest.TestCase):
    def test_forward_pass(self):
        model = SAGViTClassifier(
            patch_size=(4, 4),
            num_classes=10,  # smaller num classes for test
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            hidden_mlp_features=64,
            in_channels=2560,  # from patch dimension example
            gcn_hidden=128,
            gcn_out=64,
        )
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        # Check output shape: (B, num_classes) = (2,10)
        self.assertEqual(out.shape, (2, 10))

    def test_empty_input(self):
        model = SAGViTClassifier()
        # Passing an empty tensor should fail gracefully
        with self.assertRaises(Exception):
            model(torch.empty(0, 3, 224, 224))

    def test_invalid_input_dimensions(self):
        model = SAGViTClassifier()
        # Incorrect dimension (e.g., missing channel)
        with self.assertRaises(RuntimeError):
            model(torch.randn(2, 224, 224))  # no channel dimension


if __name__ == "__main__":
    unittest.main()
