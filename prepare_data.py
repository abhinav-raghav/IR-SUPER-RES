import os
import cv2
import numpy as np
from tqdm import tqdm

def create_synthetic_thermal_data(num_samples=100):
    """Create synthetic thermal images for training."""
    os.makedirs('data/train/hr', exist_ok=True)
    os.makedirs('data/train/lr', exist_ok=True)
    os.makedirs('data/val/hr', exist_ok=True)
    os.makedirs('data/val/lr', exist_ok=True)

    # Create training data
    print("Creating training data...")
    for i in tqdm(range(num_samples)):
        # Create a high-resolution thermal image
        hr_img = np.zeros((224, 224), dtype=np.uint8)
        
        # Add random thermal sources
        num_sources = np.random.randint(1, 5)
        for _ in range(num_sources):
            x = np.random.randint(0, 224)
            y = np.random.randint(0, 224)
            radius = np.random.randint(10, 50)
            intensity = np.random.randint(150, 255)
            cv2.circle(hr_img, (x, y), radius, intensity, -1)
        
        # Add random noise
        noise = np.random.normal(0, 10, hr_img.shape).astype(np.uint8)
        hr_img = cv2.add(hr_img, noise)
        
        # Apply Gaussian blur for more realistic thermal effect
        hr_img = cv2.GaussianBlur(hr_img, (7, 7), 1.5)
        
        # Create low-resolution version
        lr_img = cv2.resize(hr_img, (56, 56))  # 1/4 resolution
        lr_img = cv2.resize(lr_img, (224, 224))  # upscale back to original size
        
        # Save images
        cv2.imwrite(f'data/train/hr/{i:04d}.png', hr_img)
        cv2.imwrite(f'data/train/lr/{i:04d}.png', lr_img)

    # Create validation data
    print("Creating validation data...")
    for i in tqdm(range(num_samples // 5)):  # 20% of training size
        # Create a high-resolution thermal image
        hr_img = np.zeros((224, 224), dtype=np.uint8)
        
        # Add random thermal sources
        num_sources = np.random.randint(1, 5)
        for _ in range(num_sources):
            x = np.random.randint(0, 224)
            y = np.random.randint(0, 224)
            radius = np.random.randint(10, 50)
            intensity = np.random.randint(150, 255)
            cv2.circle(hr_img, (x, y), radius, intensity, -1)
        
        # Add random noise
        noise = np.random.normal(0, 10, hr_img.shape).astype(np.uint8)
        hr_img = cv2.add(hr_img, noise)
        
        # Apply Gaussian blur for more realistic thermal effect
        hr_img = cv2.GaussianBlur(hr_img, (7, 7), 1.5)
        
        # Create low-resolution version
        lr_img = cv2.resize(hr_img, (56, 56))  # 1/4 resolution
        lr_img = cv2.resize(lr_img, (224, 224))  # upscale back to original size
        
        # Save images
        cv2.imwrite(f'data/val/hr/{i:04d}.png', hr_img)
        cv2.imwrite(f'data/val/lr/{i:04d}.png', lr_img)

if __name__ == '__main__':
    create_synthetic_thermal_data() 