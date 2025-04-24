import os
import requests
import zipfile
from tqdm import tqdm

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
    
    # Download sample thermal images
    print("Downloading sample thermal images...")
    urls = [
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/thermal1.jpg",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/thermal2.jpg",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/thermal3.jpg"
    ]
    
    for i, url in enumerate(urls):
        output = f'data/raw/thermal_sr/thermal_{i+1}.jpg'
        download_file(url, output)
        print(f"Downloaded {output}")

if __name__ == '__main__':
    main() 