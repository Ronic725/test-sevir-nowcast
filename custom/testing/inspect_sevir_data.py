#!/usr/bin/env python3
"""
SEVIR Data Inspector
Inspect the downloaded SEVIR data without loading everything into memory
"""

import os
import sys
import h5py
import numpy as np
from datetime import datetime

# Add parent directories to path
sys.path.append('../../')
sys.path.append('../../src/')

def print_banner():
    """Print banner"""
    print("🔍 SEVIR Data Inspector")
    print("=" * 60)
    print("Analyzing downloaded SEVIR data structure and content")
    print("=" * 60)

def find_sevir_files():
    """Find all SEVIR data files"""
    print("\n📁 Searching for SEVIR data files...")
    
    data_dir = os.path.abspath('../../data')
    sevir_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.h5') and ('SEVIR' in file or 'sevir' in file.lower()):
                filepath = os.path.join(root, file)
                size_bytes = os.path.getsize(filepath)
                size_gb = size_bytes / (1024**3)
                sevir_files.append((filepath, size_gb, size_bytes))
    
    if sevir_files:
        print("✅ Found SEVIR data files:")
        for filepath, size_gb, size_bytes in sevir_files:
            rel_path = os.path.relpath(filepath, data_dir)
            print(f"   📊 {rel_path} ({size_gb:.2f}GB)")
    else:
        print("❌ No SEVIR data files found")
    
    return sevir_files

def inspect_sevir_structure(filepath):
    """Inspect SEVIR file structure without loading data"""
    print(f"\n🔍 Inspecting: {os.path.basename(filepath)}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"   📊 File size: {os.path.getsize(filepath) / (1024**3):.2f}GB")
            
            def print_structure(name, obj, level=0):
                indent = "   " * (level + 1)
                if isinstance(obj, h5py.Group):
                    print(f"{indent}📁 {name}/ (Group with {len(obj)} items)")
                    if level < 3:  # Limit depth to avoid too much output
                        for key in list(obj.keys())[:5]:  # Show first 5 items
                            print_structure(key, obj[key], level + 1)
                        if len(obj) > 5:
                            print(f"{indent}   ... and {len(obj) - 5} more items")
                elif isinstance(obj, h5py.Dataset):
                    print(f"{indent}📈 {name}: {obj.shape} {obj.dtype} ({obj.nbytes / (1024**2):.1f}MB)")
                    
                    # Show data range for small datasets or samples
                    if obj.size < 1000000:  # Less than 1M elements
                        try:
                            data_min, data_max = obj[:].min(), obj[:].max()
                            print(f"{indent}   Range: {data_min:.2f} to {data_max:.2f}")
                        except:
                            pass
                    else:
                        # Sample a small portion for large datasets
                        try:
                            sample = obj[0:1] if len(obj.shape) > 0 else obj[()]
                            if hasattr(sample, 'min'):
                                print(f"{indent}   Sample shape: {sample.shape}")
                        except:
                            pass
            
            print("   🗂️  File structure:")
            print_structure("root", f, -1)
            
            # Look for common SEVIR datasets
            common_keys = ['vil', 'ir069', 'ir107', 'lght']
            print(f"\n   🎯 Common SEVIR data types found:")
            for key in common_keys:
                if key in f:
                    dataset = f[key]
                    print(f"      ✅ {key}: {dataset.shape} {dataset.dtype}")
                else:
                    print(f"      ❌ {key}: Not found")
            
            # Check attributes
            if f.attrs:
                print(f"\n   📋 File attributes:")
                for key, value in f.attrs.items():
                    print(f"      {key}: {value}")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Error inspecting file: {e}")
        return False

def analyze_data_loading_strategy(filepath):
    """Analyze optimal data loading strategy"""
    print(f"\n⚡ Analyzing loading strategy for: {os.path.basename(filepath)}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Look for VIL data (most common for nowcasting)
            if 'vil' in f:
                vil_data = f['vil']
                print(f"   📊 VIL dataset: {vil_data.shape} {vil_data.dtype}")
                
                total_size_gb = vil_data.nbytes / (1024**3)
                print(f"   💾 Total VIL data size: {total_size_gb:.2f}GB")
                
                # Calculate memory-efficient loading strategies
                if len(vil_data.shape) >= 4:  # Assuming (events, height, width, time)
                    num_events = vil_data.shape[0]
                    event_size_mb = (vil_data.nbytes / num_events) / (1024**2)
                    
                    print(f"   🌩️  Number of storm events: {num_events:,}")
                    print(f"   📏 Size per event: {event_size_mb:.1f}MB")
                    
                    # Recommendations based on system memory (16GB M1 Mac)
                    available_memory_gb = 8  # Conservative estimate for data processing
                    max_events_in_memory = int((available_memory_gb * 1024) / event_size_mb)
                    
                    print(f"\n   💡 Loading recommendations for 16GB M1 Mac:")
                    print(f"      🚀 Safe batch size: {min(max_events_in_memory, 50)} events")
                    print(f"      ⚡ Quick test: {min(10, num_events)} events")
                    print(f"      🎯 Training subset: {min(500, num_events)} events")
                    print(f"      🏆 Full dataset: {num_events} events (use data generator)")
                    
                    # Sample a small portion to check data quality
                    print(f"\n   🔬 Data quality check (first event):")
                    try:
                        sample_event = vil_data[0]
                        print(f"      Shape: {sample_event.shape}")
                        print(f"      Range: {sample_event.min():.2f} to {sample_event.max():.2f}")
                        print(f"      Non-zero pixels: {np.count_nonzero(sample_event):,} / {sample_event.size:,}")
                        
                        # Check for realistic radar values
                        if sample_event.max() > 0:
                            high_values = np.sum(sample_event > 40)  # Strong reflectivity
                            moderate_values = np.sum((sample_event > 10) & (sample_event <= 40))
                            print(f"      Strong reflectivity (>40 dBZ): {high_values:,} pixels")
                            print(f"      Moderate reflectivity (10-40 dBZ): {moderate_values:,} pixels")
                    except Exception as e:
                        print(f"      ❌ Could not sample data: {e}")
                
            else:
                print("   ❌ No 'vil' dataset found - checking other data types...")
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        dataset = f[key]
                        size_gb = dataset.nbytes / (1024**3)
                        print(f"      📈 {key}: {dataset.shape} ({size_gb:.2f}GB)")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Error analyzing loading strategy: {e}")
        return False

def generate_sample_code(filepath):
    """Generate sample code for loading data efficiently"""
    print(f"\n💻 Sample code for efficient data loading:")
    
    rel_path = os.path.relpath(filepath, '../../')
    
    print(f"""
# Load small subset for testing (recommended for M1 Mac)
import h5py
import numpy as np

def load_sevir_subset(filepath='{rel_path}', max_events=10):
    with h5py.File(filepath, 'r') as f:
        if 'vil' in f:
            vil_data = f['vil']
            # Load only first max_events
            subset = vil_data[:max_events]
            print(f"Loaded {{subset.shape}} subset from {{vil_data.shape}}")
            return subset
        else:
            print("Available datasets:", list(f.keys()))
            return None

# Usage
data = load_sevir_subset()
if data is not None:
    print(f"Data range: {{data.min():.2f}} to {{data.max():.2f}}")
    print(f"Memory usage: {{data.nbytes / (1024**2):.1f}}MB")
""")

def create_data_loader_script():
    """Create a dedicated data loader script"""
    print(f"\n📝 Creating efficient SEVIR data loader...")
    
    loader_script = """#!/usr/bin/env python3
'''
Efficient SEVIR Data Loader for M1 MacBook Pro
Loads data in batches to avoid memory issues
'''

import h5py
import numpy as np
import os

class SEVIRDataLoader:
    def __init__(self, filepath, batch_size=10):
        self.filepath = filepath
        self.batch_size = batch_size
        self._file = None
        self._dataset = None
        self._num_events = 0
        
    def __enter__(self):
        self._file = h5py.File(self.filepath, 'r')
        if 'vil' in self._file:
            self._dataset = self._file['vil']
            self._num_events = self._dataset.shape[0]
            print(f"Opened SEVIR file with {self._num_events:,} events")
        else:
            print(f"Available datasets: {list(self._file.keys())}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
    
    def get_batch(self, start_idx=0, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            
        end_idx = min(start_idx + batch_size, self._num_events)
        
        if self._dataset is not None:
            batch = self._dataset[start_idx:end_idx]
            print(f"Loaded batch {start_idx}:{end_idx} - Shape: {batch.shape}")
            return batch
        return None
    
    def get_sample(self, num_events=5):
        return self.get_batch(0, num_events)
        
    @property
    def num_events(self):
        return self._num_events

# Usage example:
# with SEVIRDataLoader('data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5') as loader:
#     sample = loader.get_sample(5)  # Load 5 events
#     print(f"Sample shape: {sample.shape}")
"""
    
    loader_path = "../../custom/testing/sevir_data_loader.py"
    os.makedirs(os.path.dirname(loader_path), exist_ok=True)
    
    with open(loader_path, 'w') as f:
        f.write(loader_script)
    
    print(f"   ✅ Created: {os.path.relpath(loader_path)}")

def main():
    """Main inspection function"""
    print_banner()
    
    # Find SEVIR files
    sevir_files = find_sevir_files()
    
    if not sevir_files:
        print("\n❌ No SEVIR data files found. Please run download script first.")
        return 1
    
    # Inspect each file
    for filepath, size_gb, size_bytes in sevir_files:
        print(f"\n{'='*60}")
        inspect_sevir_structure(filepath)
        analyze_data_loading_strategy(filepath)
        
        # Only generate code for the largest/main file
        if size_gb > 1.0:  # Only for substantial files
            generate_sample_code(filepath)
    
    # Create efficient data loader
    create_data_loader_script()
    
    print(f"\n🎯 SUMMARY:")
    print(f"   📊 Found {len(sevir_files)} SEVIR data file(s)")
    total_size = sum(size_gb for _, size_gb, _ in sevir_files)
    print(f"   💾 Total size: {total_size:.2f}GB")
    print(f"   ⚡ Recommended approach: Load small batches (10-50 events)")
    print(f"   🧪 Next step: Test with sample data first")
    
    print(f"\n🚀 Quick test commands:")
    print(f"   python custom/testing/sevir_data_loader.py")
    print(f"   python -c \"exec(open('custom/testing/sevir_data_loader.py').read())\"")

if __name__ == "__main__":
    main()
