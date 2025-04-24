import os
import requests
from tqdm import tqdm
import zipfile

def download_file(url, output):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(output, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def main():
    # Create directories
    os.makedirs('data/raw/thermal_sr', exist_ok=True)
    
    # Download sample thermal images from FLIR dataset
    print("Downloading sample thermal images...")
    urls = [
        "https://raw.githubusercontent.com/FLIR/thermal-dataset/master/FLIR_ADAS_1_3/train/PreviewData/FLIR_00001.jpg",
        "https://raw.githubusercontent.com/FLIR/thermal-dataset/master/FLIR_ADAS_1_3/train/PreviewData/FLIR_00002.jpg",
        "https://raw.githubusercontent.com/FLIR/thermal-dataset/master/FLIR_ADAS_1_3/train/PreviewData/FLIR_00003.jpg",
        "https://raw.githubusercontent.com/FLIR/thermal-dataset/master/FLIR_ADAS_1_3/train/PreviewData/FLIR_00004.jpg",
        "https://raw.githubusercontent.com/FLIR/thermal-dataset/master/FLIR_ADAS_1_3/train/PreviewData/FLIR_00005.jpg"
    ]
    
    for i, url in enumerate(urls, 1):
        output = f'data/raw/thermal_sr/sample{i}.jpg'
        try:
            download_file(url, output)
            print(f"Downloaded {output}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    
    print("Sample data download complete!")

if __name__ == '__main__':
    main() 