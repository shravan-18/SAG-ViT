dependencies = ["torch", "torch-geometric", "networkx"]

from sag_vit_model import SAGViTClassifier
import torch


def SAGViT(pretrained=False, **kwargs):
    """
    SAG-ViT model endpoint.
    Args:
        pretrained (bool): If True, loads pretrained weights.
        **kwargs: Additional arguments for the model.
    Returns:
        model (nn.Module): The SAG-ViT model as proposed in the
        paper: SAG-ViT: A Scale-Aware, High-Fidelity Patching
        Approach with Graph Attention for Vision Transformers.
        https://doi.org/10.48550/arXiv.2411.09420
    """
    model = SAGViTClassifier(**kwargs)
    if pretrained:
        checkpoint = "https://github.com/shravan-18/SAG-ViT/blob/main/weights/SAG-ViT_CIFAR10.pth"
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=True)
        model.load_state_dict(state_dict)
    return model
