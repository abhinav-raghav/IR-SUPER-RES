import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import cv2

class SimpleSuperResolution(nn.Module):
    def __init__(self):
        super(SimpleSuperResolution, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class ThermalDataset(Dataset):
    def __init__(self, data_dir):
        self.hr_dir = os.path.join(data_dir, 'hr')
        self.lr_dir = os.path.join(data_dir, 'lr')
        self.filenames = [f for f in os.listdir(self.hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.filenames[idx])
        lr_path = os.path.join(self.lr_dir, self.filenames[idx])
        
        # Read images
        hr_img = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE)
        lr_img = cv2.imread(lr_path, cv2.IMREAD_GRAYSCALE)
        
        if hr_img is None or lr_img is None:
            raise ValueError(f"Could not read image {self.filenames[idx]}")
        
        # Resize images to a fixed size
        lr_img = cv2.resize(lr_img, (128, 128))
        hr_img = cv2.resize(hr_img, (128, 128))
        
        # Convert to float32 and normalize
        hr_img = hr_img.astype(np.float32) / 255.0
        lr_img = lr_img.astype(np.float32) / 255.0
        
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
        for lr_imgs, hr_imgs in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
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
                val_loss += loss.item()
                
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
    
    # Create datasets
    train_dataset = ThermalDataset('data/processed/train')
    val_dataset = ThermalDataset('data/processed/val')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = SimpleSuperResolution().to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)

if __name__ == '__main__':
    main() 