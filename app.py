from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from model import ThermalSR
import tempfile

app = Flask(__name__)
CORS(app)

# Create a temporary directory for uploads
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize model
device = torch.device('cpu')
model = None

def load_model():
    global model
    if model is None:
        try:
            model = ThermalSR().to(device)
            model.eval()
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return None
    return model

# HTML template for the upload form
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Infrared Image Super Resolution</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            min-height: 100vh;
            background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
            color: #ffffff;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            padding: 40px 0;
            background: linear-gradient(90deg, #32CD32 0%, #98FB98 100%);
            margin-bottom: 40px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .header h1 {
            margin: 0;
            color: #000000;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .upload-section {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 40px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        .button {
            background: linear-gradient(90deg, #32CD32 0%, #98FB98 100%);
            color: #000000;
            padding: 12px 25px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: transform 0.2s;
            margin: 10px;
        }
        .button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(50,205,50,0.3);
        }
        .image-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
            margin-bottom: 40px;
        }
        .image-box {
            flex: 1;
            min-width: 300px;
            max-width: 500px;
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }
        .image-box h3 {
            color: #32CD32;
            margin-bottom: 15px;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .footer {
            background: rgba(0,0,0,0.8);
            padding: 20px;
            text-align: center;
            position: fixed;
            bottom: 0;
            width: 100%;
            backdrop-filter: blur(10px);
        }
        .footer a {
            color: #32CD32;
            text-decoration: none;
            margin: 0 15px;
            transition: color 0.3s;
        }
        .footer a:hover {
            color: #98FB98;
        }
        .fa-linkedin {
            margin-right: 5px;
        }
        .download-btn {
            background: linear-gradient(90deg, #2ecc71 0%, #27ae60 100%);
            color: white;
            padding: 8px 15px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 10px;
            display: none;
        }
        .download-btn:hover {
            background: linear-gradient(90deg, #27ae60 0%, #219a52 100%);
        }
        @media (max-width: 768px) {
            .image-container {
                flex-direction: column;
            }
            .image-box {
                min-width: unset;
            }
            .header h1 {
                font-size: 1.8em;
                padding: 0 10px;
            }
            .footer {
                position: relative;
                padding: 20px 10px;
            }
            .footer a {
                display: block;
                margin: 10px 0;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Infrared Image Super Resolution Model</h1>
    </div>
    
    <div class="container">
        <div class="upload-section">
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required style="display: none" id="fileInput">
                <button type="button" class="button" onclick="document.getElementById('fileInput').click()">Choose Image</button>
                <button type="submit" class="button">Process Image</button>
            </form>
        </div>
        
        <div class="image-container">
            <div class="image-box">
                <h3>Original Image</h3>
                <img id="originalImage" style="display: none;">
                <button class="download-btn" id="downloadOriginal" style="display: none;">
                    <i class="fas fa-download"></i> Download
                </button>
            </div>
            <div class="image-box">
                <h3>Enhanced Image</h3>
                <img id="processedImage" style="display: none;">
                <button class="download-btn" id="downloadProcessed" style="display: none;">
                    <i class="fas fa-download"></i> Download
                </button>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <a href="https://www.linkedin.com/in/raghav-abhinav/" target="_blank">
            <i class="fab fa-linkedin"></i>Abhinav Raghav
        </a>
        <a href="https://www.linkedin.com/in/harshvardhan-sanguri-90248b281/" target="_blank">
            <i class="fab fa-linkedin"></i>Harshvardhan Sanguri
        </a>
    </div>

    <script>
        document.getElementById('uploadForm').onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            
            // Display and enable download for original image
            const file = formData.get('file');
            const originalImage = document.getElementById('originalImage');
            originalImage.src = URL.createObjectURL(file);
            originalImage.style.display = 'block';
            
            const downloadOriginal = document.getElementById('downloadOriginal');
            downloadOriginal.style.display = 'inline-block';
            downloadOriginal.onclick = () => {
                const link = document.createElement('a');
                link.href = originalImage.src;
                link.download = 'original_' + file.name;
                link.click();
            };
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.error) {
                    alert(data.error);
                    return;
                }
                
                // Display and enable download for processed image
                const processedImage = document.getElementById('processedImage');
                processedImage.src = data.processed_url;
                processedImage.style.display = 'block';
                
                const downloadProcessed = document.getElementById('downloadProcessed');
                downloadProcessed.style.display = 'inline-block';
                downloadProcessed.onclick = () => {
                    window.location.href = data.processed_url + '?download=true';
                };
            } catch (error) {
                alert('Error processing image: ' + error.message);
            }
        };
        
        // Update file input label with selected filename
        document.getElementById('fileInput').onchange = (e) => {
            const fileName = e.target.files[0]?.name || 'No file chosen';
            e.target.nextElementSibling.textContent = fileName;
        };
    </script>
</body>
</html>
'''

def preprocess_image(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Resize to model input size
    image = cv2.resize(image, (224, 224))
    
    # Add batch and channel dimensions
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    return image

def postprocess_image(tensor, original_size=None):
    # Remove batch and channel dimensions
    image = tensor.squeeze().cpu().numpy()
    
    # Scale back to [0, 255]
    image = (image * 255).clip(0, 255).astype(np.uint8)
    
    # Resize back to original size if specified
    if original_size is not None:
        image = cv2.resize(image, (original_size[1], original_size[0]))
    
    return image

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read and process the image
        img_array = np.frombuffer(file.read(), np.uint8)
        original_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if original_image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Save original image
        original_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'original_' + file.filename)
        cv2.imwrite(original_filename, original_image)

        # Load model
        model = load_model()
        if model is None:
            return jsonify({'error': 'Model initialization failed'}), 500

        # Process image
        with torch.no_grad():
            input_tensor = preprocess_image(original_image)
            output_tensor = model(input_tensor)
            processed_image = postprocess_image(output_tensor, original_image.shape[:2])

        # Save processed image
        processed_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + file.filename)
        cv2.imwrite(processed_filename, processed_image)

        return jsonify({
            'message': 'Success',
            'original_image': f'/uploads/original_{file.filename}',
            'processed_image': f'/uploads/processed_{file.filename}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(
        os.path.join(app.config['UPLOAD_FOLDER'], filename),
        mimetype='image/jpeg'
    )

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response

if __name__ == '__main__':
    app.run(debug=True) 