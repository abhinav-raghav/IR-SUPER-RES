# Thermal Image Super Resolution

A web application for enhancing the resolution of thermal images using deep learning. Built with PyTorch, Flask, and React.

## Features

- Upload thermal images for super-resolution processing
- Real-time image enhancement using deep learning
- Modern, responsive UI with Material-UI components
- Download enhanced images

## Tech Stack

### Backend
- Python 3.8+
- Flask
- PyTorch
- OpenCV
- NumPy

### Frontend
- React
- Material-UI
- Axios

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd thermal-super-resolution
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
```

4. Start the backend server:
```bash
# From the root directory
python app.py
```

5. Start the frontend development server:
```bash
# From the frontend directory
npm start
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

## Project Structure

```
.
├── app.py              # Flask backend server
├── model.py           # PyTorch model definition
├── train_thermal.py   # Training script
├── process_sample_data.py  # Data preprocessing
├── requirements.txt   # Python dependencies
├── uploads/          # Directory for uploaded images
└── frontend/         # React frontend application
```

## Authors

- [Abhinav Raghav](https://www.linkedin.com/in/raghav-abhinav/)
- [Harshvardhan Sanguri](https://www.linkedin.com/in/harshvardhan-sanguri-90248b281/)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 