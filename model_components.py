import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

from torchvision import models

###############################################################
# These modules correspond to core building blocks of SAG-ViT:
# 1. A CNN feature extractor for high-fidelity multi-scale feature maps.
# 2. A Graph Attention Network (GAT) to refine patch embeddings.
# 3. A Transformer Encoder to capture global long-range dependencies.
# 4. An MLP classifier head.
###############################################################

class InceptionV3FeatureExtractor(nn.Module):
    """
    Extracts multi-scale, high-fidelity feature maps from images using
    a pre-trained InceptionV3 network. This corresponds to Section 3.1,
    where a CNN backbone (here Inception) is used to produce rich feature maps
    that preserve semantic information at multiple scales.
    """
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        inception = models.inception_v3(pretrained=True)
        inception.eval()
        # Extract features up to a chosen layer (Mixed_7c) for a rich feature map
        self.extractor = nn.Sequential(*list(inception.children())[:13])
        # Freezing the extractor parameters (if desired)
        for param in self.extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the CNN backbone.

        Input:
        - x (Tensor): Input images of shape (B, 3, H, W)

        Output:
        - features (Tensor): Extracted feature map of shape (B, C, H', W'),
          where H' and W' are reduced spatial dimensions.
        """
        features = self.extractor(x)
        return features

class GATGNN(nn.Module):
    """
    A Graph Attention Network (GAT) that processes patch-graph embeddings.
    This module corresponds to the Graph Attention stage (Section 3.3),
    refining local relationships between patches in a learned manner.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(GATGNN, self).__init__()
        # GAT layers: 
        # First layer maps raw patch embeddings to a higher-level representation.
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        # Second layer produces final node embeddings with a single head.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)
        self.pool = global_mean_pool

    def forward(self, data):
        """
        Input:
        - data (PyG Data): Contains x (node features), edge_index (graph edges), and batch indexing.

        Output:
        - x (Tensor): Aggregated graph-level embedding after mean pooling.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = self.pool(x, batch)
        return x

class TransformerEncoder(nn.Module):
    """
    A Transformer encoder to capture long-range dependencies among patch embeddings.
    Integrates global dependencies after GAT processing, as per Section 3.3.
    """
    def __init__(self, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Input:
        - x (Tensor): Sequence of patch embeddings with shape (B, N, D).

        Output:
        - (Tensor): Transformed embeddings with global relationships integrated (B, N, D).
        """
        # The Transformer expects (N, B, D), so transpose first
        x = x.transpose(0, 1)  # (N, B, D)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # (B, N, D)
        return x

class MLPBlock(nn.Module):
    """
    An MLP classification head to map final global embeddings to classification logits.
    """
    def __init__(self, in_features, hidden_features, out_features):
        super(MLPBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        return self.mlp(x)
