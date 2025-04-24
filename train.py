import os
import torch
from torch.utils.data import DataLoader
from model import ThermalSR, IRSuperResolutionDataset, get_transforms, train_model

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data directories if they don't exist
    os.makedirs('data/train/hr', exist_ok=True)
    os.makedirs('data/train/lr', exist_ok=True)
    os.makedirs('data/val/hr', exist_ok=True)
    os.makedirs('data/val/lr', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Get transforms
    train_transform, val_transform = get_transforms()

    # Create datasets
    train_dataset = IRSuperResolutionDataset(
        hr_dir='data/train/hr',
        lr_dir='data/train/lr',
        transform=train_transform
    )

    val_dataset = IRSuperResolutionDataset(
        hr_dir='data/val/hr',
        lr_dir='data/val/lr',
        transform=val_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )

    # Initialize model
    model = ThermalSR()

    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,  # Reduced number of epochs for faster training
        lr=1e-4
    )

if __name__ == '__main__':
    main() 