import os
import numpy as np
import cv2
from tqdm import tqdm

def create_synthetic_thermal_image(size=(256, 256)):
    """Create a synthetic thermal image with random hot spots."""
    # Create base image with random temperature distribution
    base = np.random.normal(128, 20, size).astype(np.uint8)
    
    # Add random hot spots
    num_spots = np.random.randint(3, 8)
    for _ in range(num_spots):
        x = np.random.randint(0, size[1])
        y = np.random.randint(0, size[0])
        radius = np.random.randint(10, 50)
        intensity = np.random.randint(180, 255)
        cv2.circle(base, (x, y), radius, intensity, -1)
    
    # Apply Gaussian blur to make it look more natural
    return cv2.GaussianBlur(base, (15, 15), 0)

def main():
    # Create directories
    os.makedirs('data/raw/thermal_sr', exist_ok=True)
    
    # Generate synthetic thermal images
    print("Generating synthetic thermal images...")
    num_images = 10
    
    for i in tqdm(range(num_images)):
        # Create synthetic thermal image
        img = create_synthetic_thermal_image()
        
        # Save the image
        output_path = f'data/raw/thermal_sr/synthetic_{i+1}.png'
        cv2.imwrite(output_path, img)
    
    print("Synthetic data generation complete!")

if __name__ == '__main__':
    main() 