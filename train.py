import os
import torch
from torch import nn, optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
)

from sag_vit_model import SAGViTClassifier
from data_loader import get_dataloaders

#####################################################################
# This file provides the training loop and metric computation. It uses
# the SAG-ViT model defined in sag_vit_model.py, and the data from data_loader.py.
# The training loop is adapted to implement early stopping and track various metrics.
#####################################################################


def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    num_epochs,
    criterion,
    optimizer,
    device,
    patience=8,
    verbose=True,
):
    """
    Trains the SAG-ViT model and evaluates it on the validation set.
    Implements early stopping based on validation loss.

    Parameters:
    - model (nn.Module): The SAG-ViT model.
    - model_name (str): A name to identify the model (used for saving checkpoints).
    - train_loader, val_loader: DataLoaders for training and validation.
    - num_epochs (int): Maximum number of epochs.
    - criterion (nn.Module): Loss function.
    - optimizer (torch.optim.Optimizer): Optimization algorithm.
    - device (torch.device): Device to run the computations on (CPU/GPU).
    - patience (int): Early stopping patience.

    Returns:
    - history (dict): Dictionary containing training and validation metrics per epoch.
    """

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_prec": [],
        "train_rec": [],
        "train_f1": [],
        "train_auc": [],
        "train_mcc": [],
        "train_cohen_kappa": [],
        "train_confusion_matrix": [],
        "val_loss": [],
        "val_acc": [],
        "val_prec": [],
        "val_rec": [],
        "val_f1": [],
        "val_auc": [],
        "val_mcc": [],
        "val_cohen_kappa": [],
        "val_confusion_matrix": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()

        train_loss_total, correct, total = 0, 0, 0
        all_preds, all_labels, all_probs = [], [], []

        # Training loop
        for batch_idx, (X, y) in enumerate(tqdm(train_loader)):
            inputs, labels = X.to(device), y.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

        # Compute training metrics
        train_acc = correct / total
        train_prec = precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        )
        train_rec = recall_score(all_labels, all_preds, average="macro")
        train_f1 = f1_score(all_labels, all_preds, average="macro")
        train_cohen_kappa = cohen_kappa_score(all_labels, all_preds)
        train_mcc = matthews_corrcoef(all_labels, all_preds)
        train_confusion = confusion_matrix(all_labels, all_preds)

        history["train_loss"].append(train_loss_total / len(train_loader))
        history["train_acc"].append(train_acc)
        history["train_prec"].append(train_prec)
        history["train_rec"].append(train_rec)
        history["train_f1"].append(train_f1)
        history["train_cohen_kappa"].append(train_cohen_kappa)
        history["train_mcc"].append(train_mcc)
        history["train_confusion_matrix"].append(train_confusion)

        # Validation
        model.eval()
        val_loss_total, correct, total = 0, 0, 0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(tqdm(val_loader)):
                inputs, labels = X.to(device), y.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss_total += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())

        # Compute validation metrics
        val_acc = correct / total
        val_prec = precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        )
        val_rec = recall_score(all_labels, all_preds, average="macro")
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        val_cohen_kappa = cohen_kappa_score(all_labels, all_preds)
        val_mcc = matthews_corrcoef(all_labels, all_preds)
        val_confusion = confusion_matrix(all_labels, all_preds)

        history["val_loss"].append(val_loss_total / len(val_loader))
        history["val_acc"].append(val_acc)
        history["val_prec"].append(val_prec)
        history["val_rec"].append(val_rec)
        history["val_f1"].append(val_f1)
        history["val_cohen_kappa"].append(val_cohen_kappa)
        history["val_mcc"].append(val_mcc)
        history["val_confusion_matrix"].append(val_confusion)

        # Print epoch summary
        if verbose:
            print(
                f"Train Loss: {history['train_loss'][-1]:.4f}, Train Acc: {history['train_acc'][-1]:.4f}, "
                f"Val Loss: {history['val_loss'][-1]:.4f}, Val Acc: {history['val_acc'][-1]:.4f}"
            )

        # Early stopping
        current_val_loss = history["val_loss"][-1]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                model.load_state_dict(best_model_state)
                torch.save(model.state_dict(), f"{model_name}-best.pth")
                return history

    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), f"{model_name}-{num_epochs}_epochs.pth")

    return history


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    data_dir = "data/PlantVillage"  # "path/to/data/dir"
    num_classes = len(os.listdir(data_dir))
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir, img_size=224, batch_size=32
    )  # Minimum image size should be atleast (49, 49)

    model = SAGViTClassifier(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 100

    history = train_model(
        model,
        "SAGViT",
        train_loader,
        val_loader,
        num_epochs,
        criterion,
        optimizer,
        device,
    )

    # You may save history to a CSV or analyze it further as needed.
    # Example:
    # import pandas as pd
    # history_df = pd.DataFrame(history)
    # history_df.to_csv("training_history.csv", index=False)
