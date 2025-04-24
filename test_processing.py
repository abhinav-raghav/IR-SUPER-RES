import cv2
import torch
import numpy as np
from app import SimpleSuperResolution, preprocess_image, postprocess_image

def test_image_processing():
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleSuperResolution().to(device)
    
    # Load trained model
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.eval()
    
    # Load test image
    test_image_path = 'uploads/test_image.png'
    image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Could not load test image")
        return
    
    print(f"Original image size: {image.shape}")
    
    # Preprocess image
    input_tensor = preprocess_image(image)
    input_tensor = input_tensor.to(device)
    
    # Process image
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Postprocess output
    output_image = postprocess_image(output_tensor)
    print(f"Processed image size: {output_image.shape}")
    
    # Save processed image
    output_path = 'uploads/test_processed.png'
    cv2.imwrite(output_path, output_image)
    print(f"Processed image saved to {output_path}")
    
    # Resize original image to match processed image size for comparison
    original_resized = cv2.resize(image, (output_image.shape[1], output_image.shape[0]))
    
    # Apply same colormap to original image for fair comparison
    original_colored = cv2.applyColorMap(original_resized, cv2.COLORMAP_JET)
    
    # Calculate PSNR on colored images
    mse = np.mean((original_colored.astype(float) - output_image.astype(float)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    print(f"PSNR: {psnr:.2f} dB")
    
    # Save comparison image
    comparison = np.hstack((original_colored, output_image))
    cv2.imwrite('uploads/comparison.png', comparison)
    print("Comparison image saved to uploads/comparison.png")

if __name__ == '__main__':
    test_image_processing() 