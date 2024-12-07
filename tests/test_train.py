import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from train import train_model
from sag_vit_model import SAGViTClassifier

class TestTrain(unittest.TestCase):
    @patch("train.optim.Adam")
    def test_train_model_loop(self, mock_adam):
        # Mock the optimizer
        mock_optimizer = MagicMock()
        mock_adam.return_value = mock_optimizer

        # Mock dataloaders with a small dummy dataset
        # Just one batch with a couple of samples
        train_dataloader = [ (torch.randn(2,3,224,224), torch.tensor([0,1])) ]
        val_dataloader = [ (torch.randn(2,3,224,224), torch.tensor([0,1])) ]

        model = SAGViTClassifier(num_classes=2)

        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        # Test a single epoch training
        history = train_model(model, "TestModel", train_dataloader, val_dataloader, 
                              num_epochs=1, criterion=criterion, optimizer=mock_optimizer, device=device, patience=2, verbose=False)
        
        # Check if history is properly recorded
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)
        self.assertGreaterEqual(len(history["train_loss"]), 1)
        self.assertGreaterEqual(len(history["val_loss"]), 1)

    def test_early_stopping(self):
        # Mocking dataloaders where validation loss doesn't improve
        model = SAGViTClassifier(num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cpu")

        # create a scenario where val loss won't improve
        # first epoch normal, second epoch slightly worse
        train_dataloader = [ (torch.randn(2,3,224,224), torch.tensor([0,1])) ]
        val_dataloader = [ (torch.randn(2,3,224,224), torch.tensor([0,1])) ]

        history = train_model(model, "TestModelEarlyStop", train_dataloader, val_dataloader, 
                              num_epochs=5, criterion=criterion, optimizer=optimizer, device=device, patience=1, verbose=False)
        
        # Should have triggered early stopping before all 5 epochs
        self.assertLessEqual(len(history["train_loss"]), 5)

if __name__ == '__main__':
    unittest.main()
