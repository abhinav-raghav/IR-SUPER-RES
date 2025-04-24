# Infrared Image Super Resolution Model

A deep learning model for enhancing the resolution and quality of infrared/thermal images. The project uses PyTorch for the model implementation and Flask for the web interface.

## Features

- Web-based interface for easy image upload and processing
- Real-time image enhancement
- Download capability for both original and enhanced images
- Modern UI with responsive design
- Support for various image formats

## Tech Stack

- Python 3.x
- PyTorch
- Flask
- OpenCV
- Material Design

## Setup Instructions

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Super-Res
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Project Structure

- `app.py`: Flask web application
- `model.py`: Neural network model architecture
- `train.py`: Training script for the model
- `prepare_data.py`: Data preparation utilities
- `models/`: Directory for saved model weights
- `data/`: Training and validation data
- `uploads/`: Temporary storage for processed images

## Model Architecture

The model uses a custom CNN architecture optimized for thermal image enhancement:
- Residual learning for better feature preservation
- Skip connections to maintain thermal information
- Optimized for real-time processing

## Authors

- [Abhinav Raghav](https://www.linkedin.com/in/raghav-abhinav/)
- Harshvardhan Sanguri

## License

This project is licensed under the MIT License - see the LICENSE file for details. 