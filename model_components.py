import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

from torchvision import models

###############################################################
# These modules correspond to core building blocks of SAG-ViT:
# 1. A CNN feature extractor for high-fidelity multi-scale feature maps.
# 2. A Graph Attention Network (GAT) to refine patch embeddings.
# 3. A Transformer Encoder to capture global long-range dependencies.
# 4. An MLP classifier head.
###############################################################


class EfficientNetV2FeatureExtractor(nn.Module):
    """
    Extracts multi-scale, spatially-rich, and semantically-meaningful feature maps
    from images using a pre-trained EfficientNetV2-S model. This corresponds
    to Section 3.1, where a CNN backbone (EfficientNetV2-S) is used to produce rich
    feature maps that preserve semantic information at multiple scales.

    Feature Extraction Details:
    - The model extracts features from all layers except the last downsampling block.
    - A pre-trained model on ImageNet (IMAGENET1K_V1) can be used if `pretrained=True`.

    Unfreezing Mechanism:
    - The `unfreeze_blocks` parameter specifies how many of the last feature extraction
      blocks should be unfrozen, allowing them to be trained.
    - If `unfreeze_blocks` exceeds the total number of available blocks, all layers
      are unfrozen.

    Parameters:
    - `pretrained` (bool): If True, loads pretrained EfficientNetV2-S weights. Default is False.
    - `unfreeze_blocks` (int): Number of last feature extraction blocks to unfreeze. Default is 6 (all).

    Inputs:
    - `x` (Tensor): Input images of shape (B, C, H, W).

    Outputs:
    - Feature map (Tensor): Extracted features of shape (B, C', H', W').
    """

    def __init__(self, pretrained=False, unfreeze_blocks=6) -> None:
        super(EfficientNetV2FeatureExtractor, self).__init__()

        # Load EfficientNetV2-S with pretrained weights
        efficientnet = models.efficientnet_v2_s(
            weights="IMAGENET1K_V1" if pretrained else None
        )

        # Extract relevant features (all layers except the last downsampling)
        self.extractor = nn.Sequential(*list(efficientnet.features.children())[:-2])

        # Total number of available blocks
        total_blocks = len(list(self.extractor.children()))
        print("Total blocks: ", total_blocks)

        # Ensure unfreeze_blocks doesn't exceed the max number of blocks
        unfreeze_blocks = min(unfreeze_blocks, total_blocks)

        # Freeze all layers initially
        for param in self.extractor.parameters():
            param.requires_grad = False

        # Unfreeze last `unfreeze_blocks` blocks
        if unfreeze_blocks > 0:
            for block in list(self.extractor.children())[-unfreeze_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, x):
        return self.extractor(x)

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

    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(GATGNN, self).__init__()
        # GAT layers:
        # First layer maps raw patch embeddings to a higher-level representation.
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        # Final GCN layer for refined representation
        self.conv2 = GCNConv(hidden_channels * heads, out_channels)
        self.pool = global_mean_pool

    def forward(self, data):
        """
        Input:
        - data (PyG Data): Contains x (node features), edge_index (graph edges), and batch indexing.

        Output:
        - x (Tensor): Aggregated graph-level embedding after mean pooling.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GAT layer with ReLU activation
        x = F.relu(self.conv1(x, edge_index))

        # GCN layer for further aggregation
        x = self.conv2(x, edge_index)

        # Global mean pooling to obtain graph-level representation
        out = self.pool(x, batch)

        return out


class TransformerEncoder(nn.Module):
    """
    A Transformer encoder to capture long-range dependencies among patch embeddings.
    Integrates global dependencies after GAT processing, as per Section 3.3.
    """

    def __init__(self, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

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
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.mlp(x)
