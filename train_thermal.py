import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from tqdm import tqdm
from app import SimpleSuperResolution

class ThermalDataset(Dataset):
    def __init__(self, data_dir, size=256):
        self.hr_dir = os.path.join(data_dir, 'hr')
        self.lr_dir = os.path.join(data_dir, 'lr')
        self.filenames = [f for f in os.listdir(self.hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.size = size
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.filenames[idx])
        lr_path = os.path.join(self.lr_dir, self.filenames[idx])
        
        # Read images in grayscale to preserve thermal information
        hr_img = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE)
        lr_img = cv2.imread(lr_path, cv2.IMREAD_GRAYSCALE)
        
        if hr_img is None or lr_img is None:
            raise ValueError(f"Could not read image {self.filenames[idx]}")
        
        # Resize images to consistent sizes
        hr_img = cv2.resize(hr_img, (self.size, self.size))
        lr_img = cv2.resize(lr_img, (self.size, self.size))
        
        # Normalize while preserving relative temperature values
        hr_img = hr_img.astype(np.float32)
        lr_img = lr_img.astype(np.float32)
        
        hr_img = (hr_img - np.min(hr_img)) / (np.max(hr_img) - np.min(hr_img))
        lr_img = (lr_img - np.min(lr_img)) / (np.max(lr_img) - np.min(lr_img))
        
        # Convert to tensors
        hr_img = torch.from_numpy(hr_img).unsqueeze(0)
        lr_img = torch.from_numpy(lr_img).unsqueeze(0)
        
        return lr_img, hr_img

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for lr_imgs, hr_imgs in pbar:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            
            # Calculate loss
            loss = criterion(outputs, hr_imgs)
            
            # Add L1 loss to preserve thermal information
            l1_loss = nn.L1Loss()(outputs, hr_imgs)
            total_loss = loss + 0.5 * l1_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            pbar.set_postfix({'loss': total_loss.item()})
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                
                outputs = model(lr_imgs)
                loss = criterion(outputs, hr_imgs)
                l1_loss = nn.L1Loss()(outputs, hr_imgs)
                total_loss = loss + 0.5 * l1_loss
                
                val_loss += total_loss.item()
        
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_model.pth')
            print('Model saved!')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Create datasets with specified size
    train_dataset = ThermalDataset('data/processed/train', size=128)
    val_dataset = ThermalDataset('data/processed/val', size=128)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    model = SimpleSuperResolution().to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    
    # Train model
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device=device)

if __name__ == '__main__':
    main() 