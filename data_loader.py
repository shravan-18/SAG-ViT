import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_dataloaders(data_dir="path/to/data/dir", batch_size=512, train_split=0.8, img_size=224, num_workers=4):
    """
    Returns training and validation dataloaders for an image classification dataset.

    Parameters:
    - data_dir (str): Path to the directory containing image data in a folder structure compatible with ImageFolder.
    - batch_size (int): Number of samples per batch.
    - train_split (float): Fraction of data to use for training. Remaining is for validation.
    - img_size (int): Target size to which all images are resized after validation.
    - num_workers (int): Number of worker processes for data loading.

    Image Size Validation:
    - Minimum allowed image size: 49x49 pixels.
    - If an image has either width or height less than 49 pixels, a ValueError is raised.

    Returns:
    - train_dataloader (DataLoader): DataLoader for the training split.
    - val_dataloader (DataLoader): DataLoader for the validation split.
    """

    # Check if the provided image size is valid
    if img_size < 49:
        raise ValueError(f"Image size must be at least 49x49 pixels, but got {img_size}x{img_size}.")
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load full dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Split into training and validation sets
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataloader, val_dataloader
