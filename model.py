import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from tqdm import tqdm
import wandb

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
            
        return lr_img, hr_img

class ViTSuperResolution(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Feature extraction
        self.feature_extractor = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # Thermal-specific feature extraction
        self.thermal_features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(1) for _ in range(8)]
        )
        
    def forward(self, x):
        # Extract features using ViT
        features = self.feature_extractor(x).last_hidden_state
        
        # Extract thermal-specific features
        thermal_features = self.thermal_features(x)
        
        # Reshape features for upsampling
        batch_size = features.shape[0]
        features = features[:, 1:].transpose(1, 2).view(batch_size, 768, 14, 14)
        
        # Concatenate features
        combined_features = torch.cat([features, thermal_features], dim=1)
        
        # Upsample
        x = self.upsample(combined_features)
        
        # Apply residual blocks
        x = self.residual_blocks(x)
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

def train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # Initialize wandb
    wandb.init(project="ir-super-resolution")
    
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
        
        # Log metrics
        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        
    wandb.finish()

def get_transforms():
    train_transform = A.Compose([
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.CenterCrop(224, 224),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform 