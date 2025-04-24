import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from tqdm import tqdm

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(residual + self.gamma * out)

class ThermalSR(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initial feature extraction
        self.conv_input = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Main processing blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Final output
        self.conv_output = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Store input for residual connection
        input_img = x
        
        # Initial features
        x = self.relu(self.conv_input(x))
        
        # Main processing
        x = self.block1(x)
        x = self.block2(x)
        
        # Output with residual connection
        x = self.conv_output(x)
        out = x + input_img
        
        return out

class IRSuperResolutionDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform = transform
        self.files = [f for f in os.listdir(hr_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.files[idx])
        lr_path = os.path.join(self.lr_dir, self.files[idx])
        
        hr_img = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE)
        lr_img = cv2.imread(lr_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            augmented = self.transform(image=hr_img, image0=lr_img)
            hr_img = augmented['image']
            lr_img = augmented['image0']
        
        # Convert to float32 and normalize to [-1, 1]
        hr_img = torch.from_numpy(hr_img).float() / 127.5 - 1
        lr_img = torch.from_numpy(lr_img).float() / 127.5 - 1
        
        # Add channel dimension
        hr_img = hr_img.unsqueeze(0)
        lr_img = lr_img.unsqueeze(0)
            
        return lr_img, hr_img

def get_transforms():
    train_transform = A.Compose([
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ], additional_targets={'image0': 'image'})
    
    val_transform = A.Compose([
        A.CenterCrop(224, 224),
    ], additional_targets={'image0': 'image'})
    
    return train_transform, val_transform

def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.L1Loss()  # L1 loss for better preservation of thermal values
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for lr_imgs, hr_imgs in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                
                outputs = model(lr_imgs)
                loss = criterion(outputs, hr_imgs)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_model.pth')
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}') 