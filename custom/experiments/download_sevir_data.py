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

# Add parent directories to path
sys.path.append('../../')
sys.path.append('../../src/')

def print_banner():
    """Print banner with system info"""
    print("🌩️  SEVIR Data Downloader (AWS S3)")
    print("=" * 60)
    print("Downloading real SEVIR weather radar data")
    print("=" * 60)

def check_aws_cli():
    """Check if AWS CLI is available and working"""
    print("🔍 Checking AWS CLI...")
    
    try:
        # Test AWS CLI with SEVIR bucket
        result = subprocess.run([
            'aws', 's3', 'ls', '--no-sign-request', 's3://sevir/'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ AWS CLI working and can access SEVIR bucket")
            return True
        else:
            print(f"❌ AWS CLI error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ AWS CLI timeout - check internet connection")
        return False
    except FileNotFoundError:
        print("❌ AWS CLI not installed")
        print("Install with: brew install awscli")
        return False
    except Exception as e:
        print(f"❌ AWS CLI check failed: {e}")
        return False

def list_available_sevir_data():
    """List available SEVIR data files"""
    print("\n📋 Checking available SEVIR data...")
    
    try:
        # List VIL data for 2019
        result = subprocess.run([
            'aws', 's3', 'ls', '--no-sign-request', 's3://sevir/data/vil/2019/'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Available SEVIR VIL files (2019):")
            lines = result.stdout.strip().split('\n')
            files = []
            for line in lines:
                if line.strip() and '.h5' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        size_bytes = int(parts[2])
                        filename = parts[3]
                        size_gb = size_bytes / (1024**3)
                        files.append((filename, size_gb, size_bytes))
                        print(f"   📁 {filename} ({size_gb:.1f}GB)")
            
            return files
        else:
            print(f"❌ Could not list files: {result.stderr}")
            return []
            
    except Exception as e:
        print(f"❌ Error listing files: {e}")
        return []

def download_sevir_file(s3_path, local_path, description="SEVIR file"):
    """Download SEVIR file using AWS CLI"""
    print(f"\n📥 Downloading {description}...")
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
        
        print(f"   🚀 Running: {' '.join(cmd)}")
        
        # Run with real-time output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print progress in real-time
        for line in process.stdout:
            if line.strip():
                print(f"   {line.strip()}")
        
        process.wait()
        
        if process.returncode == 0:
            if os.path.exists(local_path):
                size_mb = os.path.getsize(local_path) / (1024**2)
                print(f"   ✅ Download successful! ({size_mb:.1f}MB)")
                return True
            else:
                print(f"   ❌ Download completed but file not found")
                return False
        else:
            print(f"   ❌ Download failed with exit code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"   ❌ Download error: {e}")
        return False

def verify_sevir_file(filepath):
    """Verify SEVIR HDF5 file"""
    print(f"\n🔍 Verifying {os.path.basename(filepath)}...")
    
    try:
        with h5py.File(filepath, 'r') as f:
            print("   ✅ HDF5 file is valid")
            print(f"   📊 Available datasets:")
            
            def print_item(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"      {name}: {obj.shape} {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"      {name}/: Group with {len(obj)} items")
            
            f.visititems(print_item)
            
            # Check for VIL data specifically
            if 'vil' in f:
                vil_data = f['vil']
                print(f"   🌩️  VIL data shape: {vil_data.shape}")
                print(f"   📈 VIL data range: {vil_data[:].min():.2f} to {vil_data[:].max():.2f}")
                
        return True
        
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        return False

def download_catalog():
    """Download SEVIR catalog using AWS CLI"""
    print("\n📋 Downloading SEVIR catalog...")
    
    catalog_s3 = "s3://sevir/CATALOG.csv"
    catalog_local = os.path.abspath('../../data/CATALOG.csv')
    
    if os.path.exists(catalog_local):
        size_mb = os.path.getsize(catalog_local) / (1024**2)
        print(f"   ✅ Catalog already exists ({size_mb:.1f}MB)")
        return catalog_local
    
    success = download_sevir_file(catalog_s3, catalog_local, "SEVIR catalog")
    return catalog_local if success else None

def create_data_directories():
    """Create necessary data directories"""
    print("\n📁 Creating data directories...")
    
    directories = [
        '../../data',
        '../../data/sevir', 
        '../../data/sevir/vil',
        '../../data/sample',
        '../../data/interim'
    ]
    
    for dir_path in directories:
        abs_path = os.path.abspath(dir_path)
        os.makedirs(abs_path, exist_ok=True)
        print(f"   ✅ {abs_path}")

def main():
    """Main download function"""
    try:
        print_banner()
        
        # Check AWS CLI
        if not check_aws_cli():
            print("\n❌ Cannot proceed without AWS CLI access to SEVIR")
            print("Please install AWS CLI: brew install awscli")
            return 1
        
        # Create directories
        create_data_directories()
        
        # List available data
        available_files = list_available_sevir_data()
        
        if not available_files:
            print("❌ No SEVIR files found")
            return 1
        
        # Download catalog first
        catalog_path = download_catalog()
        
        # Choose optimal file for M1 MacBook Pro (smallest storm events file)
        target_file = None
        for filename, size_gb, size_bytes in available_files:
            if 'STORMEVENTS' in filename and size_gb < 10:  # Under 10GB for M1 Mac
                target_file = (filename, size_gb, size_bytes)
                break
        
        if target_file is None:
            # Fallback to smallest available file
            target_file = min(available_files, key=lambda x: x[2])
        
        filename, size_gb, size_bytes = target_file
        
        print(f"\n🎯 Selected file for download: {filename} ({size_gb:.1f}GB)")
        
        if size_gb > 8:
            response = input(f"   ⚠️  This file is {size_gb:.1f}GB. Continue? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("   ⏹️  Download cancelled")
                return 1
        
        # Download the selected file
        s3_path = f"s3://sevir/data/vil/2019/{filename}"
        local_path = os.path.abspath(f"../../data/sevir/vil/{filename}")
        
        success = download_sevir_file(s3_path, local_path, f"SEVIR VIL data ({filename})")
        
        if success:
            # Verify the downloaded file
            if verify_sevir_file(local_path):
                print("\n🎉 SEVIR data download completed successfully!")
                print("\n📊 Summary:")
                print(f"   📁 Data directory: {os.path.abspath('../../data')}")
                if catalog_path:
                    print(f"   📋 Catalog: {os.path.relpath(catalog_path)}")
                print(f"   🌩️  VIL data: {os.path.relpath(local_path)} ({size_gb:.1f}GB)")
                
                print("\n🚀 Next steps:")
                print("   1. Run training: python custom/training/train_limited_data.py --data_source sevir")
                print("   2. Run tests: python custom/testing/test_realistic_sevir.py")
                print(f"   3. Explore data: python -c \"import h5py; f=h5py.File('{local_path}', 'r'); print(list(f.keys()))\"")
                
                return 0
            else:
                print("❌ Downloaded file verification failed")
                return 1
        else:
            print("❌ Download failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Download cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Download failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
