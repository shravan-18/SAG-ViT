import argparse
import torch
from PIL import Image
from torchvision import transforms

from sag_vit_model import SAGViTClassifier
from opts import *


def load_model(checkpoint_path, device):
    """
    Loads the SAG-ViT model with given checkpoint weights.

    Parameters:
    - checkpoint_path (str): Path to the model weights file.
    - device (torch.device): Device to load the model on.

    Returns:
    - model (SAGViTClassifier): The SAG-ViT model with weights loaded.
    """
    # Initialize the model with the same configuration used during training
    model = SAGViTClassifier(
        patch_size=(4, 4),
        num_classes=10,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        hidden_mlp_features=64,
        in_channels=2560,
        gcn_hidden=128,
        gcn_out=64,
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    """
    Preprocesses the image to match the model input.

    Adjust the transforms if you used different normalization
    or size during training.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img)
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def predict_image(model, img_tensor, device):
    """
    Predicts the class label for the given image tensor.
    """
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)
        return preds.item(), probs[0, preds.item()].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAG-ViT Inference")
    parser.add_argument(
        "--image_path",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--cpu", action="store_true", default=DEFAULT_CPU, help="Run on CPU"
    )
    args = parser.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    # Load model
    model = load_model(args.checkpoint_path, device)

    # Preprocess image
    img_tensor = preprocess_image(args.image_path)

    # Predict
    pred_class, confidence = predict_image(model, img_tensor, device)

    # CIFAR-10 label mapping
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    predicted_label = class_names[pred_class]
    print(f"Predicted class: {predicted_label} with confidence: {confidence:.4f}")
