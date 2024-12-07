import unittest
import os
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader
import torch

# Assuming get_dataloaders is defined in data_loader.py
from data_loader import get_dataloaders

class TestDataLoader(unittest.TestCase):
    @patch("data_loader.datasets.ImageFolder")
    @patch("data_loader.random_split")
    def test_get_dataloaders_structure(self, mock_random_split, mock_imagefolder):
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 1000
        mock_imagefolder.return_value = mock_dataset

        # mock splits
        mock_train = MagicMock()
        mock_val = MagicMock()
        mock_random_split.return_value = (mock_train, mock_val)

        train_dl, val_dl = get_dataloaders(data_dir="path/to/data/dir", batch_size=32)

        # Check that returned objects are DataLoaders
        self.assertIsInstance(train_dl, DataLoader)
        self.assertIsInstance(val_dl, DataLoader)

    @patch("data_loader.datasets.ImageFolder")
    @patch("data_loader.random_split")
    def test_data_split_ratios(self, mock_random_split, mock_imagefolder):
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 1000
        mock_imagefolder.return_value = mock_dataset

        get_dataloaders(data_dir="path/to/data/dir", batch_size=32, train_split=0.8)
        mock_random_split.assert_called_once()
        args, _ = mock_random_split.call_args
        train_subset, val_subset = args[0], args[1]
        self.assertEqual(train_subset, 1000)
        self.assertEqual(val_subset[0], 800)
        self.assertEqual(val_subset[1], 200)

    def test_invalid_data_dir(self):
        # If directory doesn't exist, ImageFolder should fail.
        # We'll let it raise an exception and catch it.
        with self.assertRaises(Exception):
            get_dataloaders(data_dir="invalid/path", batch_size=32)

if __name__ == '__main__':
    unittest.main()
