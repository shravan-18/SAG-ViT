import torch
from torch import nn

from torch_geometric.data import Batch
from model_components import EfficientNetV2FeatureExtractor, GATGNN, TransformerEncoder, MLPBlock
from graph_construction import build_graph_from_patches, build_graph_data_from_patches

###############################################################################
# SAG-ViT Model:
# This class combines:
# 1) CNN backbone to produce high-fidelity feature maps (Section 3.1),
# 2) Graph construction and GAT to refine local patch embeddings (Section 3.2 and 3.3),
# 3) A Transformer encoder to capture global relationships (Section 3.3),
# 4) A final MLP classifier.
###############################################################################

class SAGViTClassifier(nn.Module):
    """
    SAG-ViT: Scale-Aware Graph Attention Vision Transformer

    This model integrates the following steps:
    - Extract multi-scale features from images using a CNN backbone (InceptionV3 here).
    - Partition the feature map into patches and build a graph where each node is a patch.
    - Use a Graph Attention Network (GAT) to refine patch embeddings based on local spatial relationships.
    - Utilize a Transformer encoder to model long-range dependencies and integrate multi-scale information.
    - Finally, classify the resulting representation into desired classes.

    Inputs:
    - x (Tensor): Input images (B, 3, H, W)

    Outputs:
    - out (Tensor): Classification logits (B, num_classes)
    """
    def __init__(
        self,
        patch_size=(4,4),
        num_classes=10,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        hidden_mlp_features=64,
        in_channels=2560,  # Derived from patch dimensions and CNN output channels
        gcn_hidden=128,
        gcn_out=64
    ):
        super(SAGViTClassifier, self).__init__()

        # CNN feature extractor (frozen pre-trained InceptionV3)
        self.cnn = EfficientNetV2FeatureExtractor()

        # Graph Attention Network to process patch embeddings
        self.gcn = GATGNN(in_channels=in_channels, hidden_channels=gcn_hidden, out_channels=gcn_out)

        # Learnable positional embedding for Transformer input
        self.positional_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        # Extra embedding token (similar to class token) to summarize global info
        self.extra_embedding = nn.Parameter(torch.randn(1, d_model))

        # Transformer encoder to capture long-range global dependencies
        self.transformer_encoder = TransformerEncoder(d_model, nhead, num_layers, dim_feedforward)

        # MLP classification head
        self.mlp = MLPBlock(d_model, hidden_mlp_features, num_classes)

        self.patch_size = patch_size

    def forward(self, x):
        # Step 1: High-fidelity feature extraction from CNN
        feature_map = self.cnn(x)

        # Step 2: Build graphs from patches
        G_global_batch, patches = build_graph_from_patches(feature_map, self.patch_size)

        # Step 3: Convert to PyG Data format and batch
        data_list = build_graph_data_from_patches(G_global_batch, patches)
        device = x.device
        batch = Batch.from_data_list(data_list).to(device)

        # Step 4: GAT stage
        x_gcn = self.gcn(batch)

        # Step 5: Reshape GCN output back to (B, N, D)
        # The number of patches per image is determined by patch size and feature map dimensions.
        B = x.size(0)
        D = x_gcn.size(-1)
        # N is automatically inferred
        # Thus x_gcn is (B, D) now. We need a sequence dimension for the Transformer.
        # Let's treat each image-level embedding as one "patch token" plus an extra token:
        patch_embeddings = x_gcn.unsqueeze(1)  # (B, 1, D)

        # Add positional embedding
        patch_embeddings = patch_embeddings + self.positional_embedding  # (B, 1, D)

        # Add an extra learnable embedding (like a CLS token)
        patch_embeddings = torch.cat([patch_embeddings, self.extra_embedding.unsqueeze(0).expand(B, -1, -1)], dim=1)  # (B, 2, D)

        # Step 6: Transformer encoder
        x_trans = self.transformer_encoder(patch_embeddings)

        # Step 7: Global pooling (here we just take the mean)
        x_pooled = x_trans.mean(dim=1)  # (B, D)

        # Classification
        out = self.mlp(x_pooled)
        return out
