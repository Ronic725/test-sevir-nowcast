"""
Downloads pretrained models for nowcast and synrad
"""
import pandas as pd
import urllib.request
import os
import subprocess

def main():
    model_info = pd.read_csv('model_urls.csv')
    for i,r in model_info.iterrows():
        print(f'Downloading {r.model}...')
        # Create directory if it doesn't exist
        app_dir = r.application
        if not os.path.exists(app_dir):
            os.makedirs(app_dir)
        success = download_file(r.url,f'{r.application}/{r.model}')
        if not success:
            print(f'Failed to download {r.model}')

def download_file(url, filename):
    # Convert Dropbox share URLs to direct download URLs
    if 'dropbox.com' in url and '?dl=0' in url:
        direct_url = url.replace('?dl=0', '?dl=1')
    else:
        direct_url = url
    
    print(f'Downloading from: {direct_url}')
    print(f'Saving to: {filename}')
    
    # Use curl for more reliable downloads with Dropbox
    try:
        cmd = ['curl', '-L', '-o', filename, direct_url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Check file size and type
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f'Downloaded {filename} - Size: {file_size} bytes')
                
                # Check if it's actually an H5 file by looking at file header
                try:
                    with open(filename, 'rb') as f:
                        header = f.read(8)
                        # HDF5 files start with specific signature
                        if header.startswith(b'\x89HDF\r\n\x1a\n'):
                            print(f'✓ {filename} appears to be a valid HDF5 file')
                            return True
                        elif b'<!DOCTYPE' in header or b'<html' in header:
                            print(f'✗ {filename} is HTML, not an HDF5 file - download failed')
                            os.remove(filename)
                            return False
                        else:
                            print(f'? {filename} has unknown format - may still be valid')
                            return True
                except Exception as e:
                    print(f'Could not verify file format: {e}')
                    return True
            else:
                print(f'File {filename} was not created')
                return False
        else:
            print(f'curl failed with return code {result.returncode}')
            print(f'Error: {result.stderr}')
            return False
            
    except Exception as e:
        print(f'Download failed for {filename}: {e}')
        return False

if __name__=='__main__':
    main()



