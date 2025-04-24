import os
import cv2
import numpy as np
from tqdm import tqdm

def create_low_resolution(img, scale_factor=4):
    """Create a low-resolution version of the image."""
    h, w = img.shape[:2]
    lr_h, lr_w = h // scale_factor, w // scale_factor
    lr_img = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    return lr_img

def process_images(input_dir, output_dir, scale_factor=4):
    """Process images and create low-resolution versions."""
    # Create output directories
    hr_dir = os.path.join(output_dir, 'hr')
    lr_dir = os.path.join(output_dir, 'lr')
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)
    
    # Process images
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Read image
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Create low resolution version
                lr_img = create_low_resolution(img, scale_factor)
                
                # Save images
                hr_path = os.path.join(hr_dir, filename)
                lr_path = os.path.join(lr_dir, filename)
                
                cv2.imwrite(hr_path, img)
                cv2.imwrite(lr_path, lr_img)

def main():
    # Create directories
    os.makedirs('data/raw/thermal_sr', exist_ok=True)
    os.makedirs('data/processed/train', exist_ok=True)
    os.makedirs('data/processed/val', exist_ok=True)
    
    # Process training data
    print("Processing training data...")
    process_images('data/raw/thermal_sr', 'data/processed/train')
    
    # Process validation data
    print("Processing validation data...")
    process_images('data/raw/thermal_sr', 'data/processed/val')

if __name__ == '__main__':
    main() 