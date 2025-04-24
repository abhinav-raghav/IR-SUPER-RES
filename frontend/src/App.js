import React, { useState } from 'react';
import { Button, CircularProgress, Box, Typography, Container } from '@mui/material';
import axios from 'axios';
import './App.css';
import LinkedInIcon from '@mui/icons-material/LinkedIn';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    setSelectedFile(event.target.files[0]);
    setOriginalImage(URL.createObjectURL(event.target.files[0]));
    setProcessedImage(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData);
      setProcessedImage(`http://localhost:5000${response.data.processed_url}`);
    } catch (error) {
      setError(error.response?.data?.error || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (processedImage) {
      // Add download parameter to trigger download instead of display
      const downloadUrl = `${processedImage}?download=true`;
      window.location.href = downloadUrl;
    }
  };

  return (
    <div className="App">
      <Container maxWidth="md">
        <Typography variant="h4" gutterBottom>
          Thermal Image Super Resolution
        </Typography>
        
        <Box className="upload-container">
          <input
            type="file"
            accept=".png,.jpg,.jpeg"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
            id="file-input"
          />
          <label htmlFor="file-input">
            <Button variant="contained" component="span">
              Select Image
            </Button>
          </label>
          
          {selectedFile && (
            <Button
              variant="contained"
              onClick={handleUpload}
              disabled={loading}
              style={{ marginLeft: '10px' }}
            >
              Process Image
            </Button>
          )}
        </Box>

        {error && <Typography color="error" className="error-message">{error}</Typography>}
        
        {loading && (
          <Box className="loading-container">
            <CircularProgress />
            <Typography>Processing image...</Typography>
          </Box>
        )}

        <Box className="images-container">
          {originalImage && (
            <Box className="image-box">
              <Typography variant="h6">Original Image</Typography>
              <img src={originalImage} alt="Original" className="image" />
            </Box>
          )}
          
          {processedImage && (
            <Box className="image-box">
              <Typography variant="h6">Enhanced Image</Typography>
              <img src={processedImage} alt="Enhanced" className="image" />
              <Button
                variant="contained"
                onClick={handleDownload}
                className="download-button"
              >
                Download Enhanced Image
              </Button>
            </Box>
          )}
        </Box>
      </Container>
      <footer className="footer">
        <div className="footer-content">
          <div>
            Created by{' '}
            <a href="https://www.linkedin.com/in/raghav-abhinav/" target="_blank" rel="noopener noreferrer">
              Abhinav Raghav
              <LinkedInIcon className="linkedin-icon" />
            </a>
            {' '}&{' '}
            <a href="https://www.linkedin.com/in/harshvardhan-sanguri-90248b281/" target="_blank" rel="noopener noreferrer">
              Harshvardhan Sanguri
              <LinkedInIcon className="linkedin-icon" />
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App; 