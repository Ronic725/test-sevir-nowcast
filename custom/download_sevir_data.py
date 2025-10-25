#!/usr/bin/env python3
"""
SEVIR Data Downloader
Downloads real SEVIR data using AWS CLI for training and testing weather nowcasting models
Fixed to use correct AWS S3 paths and proper download methods
"""

import os
import sys
import subprocess
import h5py
import numpy as np
from datetime import datetime
import shutil
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import centralized configuration
from config.project_paths import get_paths
paths = get_paths()
paths.setup_python_path()

# ============================================================================
# CONFIGURATION - Modify these variables to control what data to download
# ============================================================================

# Image type: 'vil', 'vis', 'ir069', 'ir107', 'lght'
IMAGE_TYPE = 'vil'

# Sampling strategy: 'RANDOMEVENTS' or 'STORMEVENTS'
SAMPLING_STRATEGY = 'STORMEVENTS'

# Year to download: 2017, 2018, 2019
YEAR = 2019

# Date range (MMDD format): Start and end dates
# For RANDOMEVENTS: typically split into quarters (0101-0430, 0501-0831, 0901-1231)
# For STORMEVENTS: typically half-year splits (0101-0630, 0701-1231)
# Set to None to download all files matching the criteria
DATE_START = '0101'  # MMDD format, e.g., '0101' for January 1
DATE_END = '0630'    # MMDD format, e.g., '0630' for June 30

# Maximum file size to download (in GB). Set to None for no limit.
MAX_FILE_SIZE_GB = 10.0

# Auto-confirm downloads (skip confirmation prompt)
AUTO_CONFIRM = False

# ============================================================================

def print_banner():
    """Print banner with system info"""
    print("=" * 70)
    print("SEVIR Data Downloader (AWS S3)")
    print("=" * 70)
    print("Downloading real SEVIR weather radar data")
    print(f"Configuration: {IMAGE_TYPE} | {SAMPLING_STRATEGY} | {YEAR}")
    if DATE_START and DATE_END:
        print(f"Date Range: {DATE_START}-{DATE_END}")
    print("=" * 70)

def check_aws_cli():
    """Check if AWS CLI is available and working"""
    print("\n[INFO] Checking AWS CLI...")
    
    try:
        # Test AWS CLI with SEVIR bucket
        result = subprocess.run([
            'aws', 's3', 'ls', '--no-sign-request', 's3://sevir/'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("[SUCCESS] AWS CLI working and can access SEVIR bucket")
            return True
        else:
            print(f"[ERROR] AWS CLI error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("[ERROR] AWS CLI timeout - check internet connection")
        return False
    except FileNotFoundError:
        print("[ERROR] AWS CLI not installed")
        print("[INFO] Install with: brew install awscli")
        return False
    except Exception as e:
        print(f"[ERROR] AWS CLI check failed: {e}")
        return False

def list_available_sevir_data():
    """List available SEVIR data files"""
    print(f"\n[INFO] Checking available SEVIR data for {IMAGE_TYPE}/{YEAR}...")
    
    try:
        # Construct S3 path based on configuration
        s3_path = f's3://sevir/data/{IMAGE_TYPE}/{YEAR}/'
        
        result = subprocess.run([
            'aws', 's3', 'ls', '--no-sign-request', '--human-readable',
            s3_path
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(f"[SUCCESS] Available SEVIR {IMAGE_TYPE.upper()} files ({YEAR}):")
            
            # Parse for structured data
            lines = result.stdout.strip().split('\n')
            files = []
            for line in lines:
                if line.strip() and '.h5' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        # With --human-readable, size is like "15.2 GiB"
                        size_str = f"{parts[2]} {parts[3]}"
                        filename = parts[4]
                        
                        # Filter by sampling strategy
                        if SAMPLING_STRATEGY and SAMPLING_STRATEGY not in filename:
                            continue
                        
                        # Filter by date range if specified
                        if DATE_START and DATE_END:
                            if not (DATE_START in filename and DATE_END in filename):
                                continue
                        
                        # Convert size to GB for filtering
                        size_gb = float(parts[2])
                        if parts[3] == 'MiB':
                            size_gb = size_gb / 1024
                        elif parts[3] == 'KiB':
                            size_gb = size_gb / (1024 * 1024)
                        
                        files.append((filename, size_str, size_gb))
                        print(f"   - {filename} ({size_str})")
            
            if not files:
                print(f"[WARNING] No files match the criteria: {SAMPLING_STRATEGY}, {DATE_START}-{DATE_END}")
            
            return files
            
    except Exception as e:
        print(f"[ERROR] Error listing files: {e}")
        return []

def download_sevir_file(s3_path, local_path, description="SEVIR file"):
    """Download SEVIR file using AWS CLI"""
    print(f"\n[INFO] Downloading {description}...")
    print(f"   S3 path: {s3_path}")
    print(f"   Local path: {local_path}")
    
    # Ensure local directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    try:
        # Use AWS CLI to download
        cmd = [
            'aws', 's3', 'cp', '--no-sign-request',
            s3_path, local_path
        ]
        
        print(f"   Running: {' '.join(cmd)}")
        
        # Run with real-time output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print progress in real-time
        if process.stdout:
            for line in process.stdout:
                if line.strip():
                    print(f"   {line.strip()}")
        
        process.wait()
        
        if process.returncode == 0:
            if os.path.exists(local_path):
                size_mb = os.path.getsize(local_path) / (1024**2)
                print(f"   [SUCCESS] Download successful! ({size_mb:.1f}MB)")
                return True
            else:
                print(f"   [ERROR] Download completed but file not found")
                return False
        else:
            print(f"   [ERROR] Download failed with exit code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"   [ERROR] Download error: {e}")
        return False

def verify_sevir_file(filepath):
    """Verify SEVIR HDF5 file"""
    print(f"\n[INFO] Verifying {os.path.basename(filepath)}...")
    
    try:
        with h5py.File(filepath, 'r') as f:
            print("   [SUCCESS] HDF5 file is valid")
            print(f"   Available datasets:")
            
            def print_item(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"      {name}: {obj.shape} {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"      {name}/: Group with {len(obj)} items")
            
            f.visititems(print_item)
            
            # Check for data (vil, vis, ir069, ir107, lght)
            if IMAGE_TYPE in f:
                data = f[IMAGE_TYPE]
                if isinstance(data, h5py.Dataset):
                    print(f"   {IMAGE_TYPE.upper()} data shape: {data.shape}")
                    sample = data[0:min(10, data.shape[0])]
                    print(f"   {IMAGE_TYPE.upper()} data range: {sample.min():.2f} to {sample.max():.2f}")
                
        return True
        
    except Exception as e:
        print(f"   [ERROR] Verification failed: {e}")
        return False

def download_catalog():
    """Download SEVIR catalog using AWS CLI"""
    print("\n[INFO] Downloading SEVIR catalog...")
    
    catalog_s3 = "s3://sevir/CATALOG.csv"
    catalog_local = os.path.join(str(paths.data), 'CATALOG.csv')
    
    if os.path.exists(catalog_local):
        size_mb = os.path.getsize(catalog_local) / (1024**2)
        print(f"   [INFO] Catalog already exists ({size_mb:.1f}MB)")
        return catalog_local
    
    success = download_sevir_file(catalog_s3, catalog_local, "SEVIR catalog")
    return catalog_local if success else None

def create_data_directories():
    """Create necessary data directories"""
    print("\n[INFO] Creating data directories...")
    
    directories = [
        os.path.join(str(paths.data), 'sevir'),
        os.path.join(str(paths.data), 'sevir', IMAGE_TYPE),
        os.path.join(str(paths.data), 'sample'),
        os.path.join(str(paths.data), 'interim')
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"   [SUCCESS] {dir_path}")

def main():
    """Main download function"""
    try:
        print_banner()
        
        # Check AWS CLI
        if not check_aws_cli():
            print("\n[ERROR] Cannot proceed without AWS CLI access to SEVIR")
            print("[INFO] Please install AWS CLI: brew install awscli")
            return 1
        
        # Create directories
        create_data_directories()
        
        # List available data
        available_files = list_available_sevir_data()
        
        if not available_files:
            print("[ERROR] No SEVIR files found matching criteria")
            print(f"[INFO] Try different configuration: IMAGE_TYPE={IMAGE_TYPE}, SAMPLING_STRATEGY={SAMPLING_STRATEGY}, YEAR={YEAR}")
            return 1
        
        # Download catalog first
        catalog_path = download_catalog()
        
        # Select file based on criteria
        target_file = None
        
        # Filter by max file size if specified
        if MAX_FILE_SIZE_GB:
            eligible_files = [f for f in available_files if f[2] <= MAX_FILE_SIZE_GB]
            if eligible_files:
                target_file = eligible_files[0]  # Take first eligible file
            else:
                print(f"[WARNING] All files exceed MAX_FILE_SIZE_GB={MAX_FILE_SIZE_GB}GB")
                target_file = min(available_files, key=lambda x: x[2])  # Take smallest
        else:
            target_file = available_files[0]  # Take first available
        
        if target_file is None:
            print("[ERROR] No suitable file found")
            return 1
        
        filename, size_str, size_gb = target_file
        
        print(f"\n[INFO] Selected file for download: {filename} ({size_str})")
        
        # Confirmation prompt unless auto-confirm is enabled
        if not AUTO_CONFIRM and size_gb > 1.0:
            response = input(f"   This file is {size_str}. Continue? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("   [INFO] Download cancelled")
                return 1
        
        # Download the selected file
        s3_path = f"s3://sevir/data/{IMAGE_TYPE}/{YEAR}/{filename}"
        local_path = os.path.join(str(paths.data), 'sevir', IMAGE_TYPE, filename)
        
        success = download_sevir_file(s3_path, local_path, f"SEVIR {IMAGE_TYPE.upper()} data ({filename})")
        
        if success:
            # Verify the downloaded file
            if verify_sevir_file(local_path):
                print("\n" + "=" * 70)
                print("SEVIR data download completed successfully!")
                print("=" * 70)
                print("\nSummary:")
                print(f"   Data directory: {paths.data}")
                if catalog_path:
                    print(f"   Catalog: {os.path.relpath(catalog_path)}")
                print(f"   {IMAGE_TYPE.upper()} data: {os.path.relpath(local_path)} ({size_str})")
                
                print("\nNext steps:")
                print("   1. Run training: python custom/training/train_limited_data.py --data_source sevir")
                print("   2. Run tests: python custom/testing/test_realistic_sevir.py")
                print(f"   3. Explore data: python -c \"import h5py; f=h5py.File('{local_path}', 'r'); print(list(f.keys()))\"")
                
                return 0
            else:
                print("[ERROR] Downloaded file verification failed")
                return 1
        else:
            print("[ERROR] Download failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n[INFO] Download cancelled by user")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Download failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)