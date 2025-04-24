from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import torch
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class SimpleSuperResolution(torch.nn.Module):
    def __init__(self):
        super(SimpleSuperResolution, self).__init__()
        # Initial feature extraction
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        
        # Residual blocks
        self.res_blocks = torch.nn.ModuleList([
            self.make_res_block(64) for _ in range(3)
        ])
        
        # Final reconstruction
        self.conv_final = torch.nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
    def make_res_block(self, channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # Initial feature extraction
        out = self.relu(self.conv1(x))
        
        # Store the input features for residual connection
        identity = out
        
        # Residual blocks
        for res_block in self.res_blocks:
            out = res_block(out) + identity
            identity = out
            
        # Final reconstruction
        out = self.conv_final(out)
        
        # Add global residual connection
        return out + x

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleSuperResolution().to(device)

# Try to load trained model, if it exists
model_path = 'models/best_model.pth'
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using untrained model...")
else:
    print("No trained model found. Using untrained model...")

def preprocess_image(image, size=128):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Store original size for later
    original_size = image.shape[:2]
    
    # Resize to model input size
    image = cv2.resize(image, (size, size))
    
    # Normalize to [0, 1] range while preserving relative temperature values
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    # Add batch and channel dimensions
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    
    return image, original_size

def postprocess_image(tensor, original_size):
    # Convert to numpy and remove batch and channel dimensions
    image = tensor.squeeze().cpu().numpy()
    
    # Rescale to [0, 255] while preserving relative temperature values
    image = np.clip(image, 0, 1)
    image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
    
    # Resize back to original size
    image = cv2.resize(image, (original_size[1], original_size[0]))
    
    # Apply thermal colormap (COLORMAP_JET is commonly used for thermal images)
    colored = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    
    # Enhance contrast for better visualization while preserving relative values
    colored = cv2.convertScaleAbs(colored, alpha=1.2, beta=0)
    
    return colored

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read image in grayscale to preserve thermal information
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return jsonify({'error': 'Failed to read image'}), 400
            
            # Preprocess image while keeping track of original size
            input_tensor, original_size = preprocess_image(image)
            input_tensor = input_tensor.to(device)
            
            # Process image
            with torch.no_grad():
                output_tensor = model(input_tensor)
            
            # Postprocess while preserving temperature range and restoring original size
            output_image = postprocess_image(output_tensor, original_size)
            
            # Save processed image
            processed_filename = f"processed_{filename}"
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            cv2.imwrite(processed_filepath, output_image)
            
            return jsonify({
                'message': 'Image processed successfully',
                'original_url': f'/uploads/{filename}',
                'processed_url': f'/uploads/{processed_filename}'
            })
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({'error': 'Error processing image. Please try again.'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Check if this is a download request
    if request.args.get('download'):
        return send_file(
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            as_attachment=True,
            download_name=filename,
            mimetype='image/png'
        )
    # Otherwise, display the image
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

# Add CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response

if __name__ == '__main__':
    app.run(debug=True) 