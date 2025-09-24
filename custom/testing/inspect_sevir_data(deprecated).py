#!/usr/bin/env python3
"""
SEVIR Data Inspector
Inspect the downloaded SEVIR data without loading everything into memory
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
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
    
    # data_dir = os.path.abspath('../../data') # this is wrong as it assume the code is run from cwd
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data')) 
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
                        
                        # Check for synthetic radar values
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

def plot_sevir_vil_data(filepath, num_events=3, frames_per_event=4, seed=42):
    """Plot sample SEVIR VIL data for visualization with controlled seeding"""
    print(f"\n📊 Plotting SEVIR VIL data from: {os.path.basename(filepath)}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            if 'vil' not in f:
                print("   ❌ No 'vil' dataset found in file")
                return False
            
            vil_data = f['vil']
            print(f"   📈 VIL dataset shape: {vil_data.shape}")
            
            # Use seeded random sampling for consistency
            max_events = min(num_events, vil_data.shape[0])
            
            # Set random seed for reproducible sampling
            import random
            random.seed(seed)
            np.random.seed(seed)
            
            available_indices = list(range(vil_data.shape[0]))
            selected_indices = sorted(random.sample(available_indices, max_events))
            sample_data = vil_data[selected_indices]
            print(f"   🎲 Randomly selected events (seed={seed}): {selected_indices}")
            
            print(f"   📊 Loaded {max_events} events for visualization")
            print(f"   🌩️  Data range: {sample_data.min():.2f} to {sample_data.max():.2f} dBZ")
            
            # Create visualization with proper spacing for colorbar
            fig, axes = plt.subplots(max_events, frames_per_event, figsize=(22, 4*max_events))
            if max_events == 1:
                axes = axes.reshape(1, -1)
            
            last_im = None  # Keep track of last image for colorbar
            
            for event_idx in range(max_events):
                event_data = sample_data[event_idx]
                total_frames = event_data.shape[-1] if len(event_data.shape) > 2 else 1
                
                # Select frames to display (evenly spaced)
                if total_frames > 1:
                    frame_indices = np.linspace(0, total_frames-1, frames_per_event, dtype=int)
                else:
                    frame_indices = [0] * frames_per_event
                
                for i, frame_idx in enumerate(frame_indices):
                    ax = axes[event_idx, i]
                    
                    if len(event_data.shape) > 2:  # Time series data
                        frame = event_data[:, :, frame_idx]
                        title = f'Event {event_idx+1}, Frame {frame_idx+1}/{total_frames}'
                    else:  # Single frame
                        frame = event_data
                        title = f'Event {event_idx+1}'
                    
                    # Plot radar reflectivity
                    im = ax.imshow(frame, cmap='viridis', vmin=0, vmax=70)
                    ax.set_title(title, fontsize=10)
                    ax.axis('off')
                    last_im = im  # Keep reference for colorbar
            
            # Manually adjust subplot layout to leave space for colorbar
            plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05, 
                               wspace=0.1, hspace=0.2)
            
            # Add colorbar in the reserved space
            if last_im is not None:
                cbar = fig.colorbar(last_im, ax=axes, label='Reflectivity (dBZ)', 
                                   shrink=0.9, aspect=25)
            
            # Determine data type from filename and path for output naming
            filename_base = os.path.basename(filepath).lower()
            
            # Check if it's synthetic data
            if 'synthetic' in filename_base or 'synthetic' in filename_base or 'test' in filename_base:
                data_type = "synthetic"
            elif 'sevir_vil_stormevents' in filename_base or 'vil' in os.path.dirname(filepath).lower():
                data_type = "true_sevir"
            else:
                data_type = "unknown"
            
            # Save plot with descriptive filename including seed notation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sevir_vil_plot_{data_type}_s{seed}_{timestamp}.png"
            
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"   ✅ Plot saved to: {filename}")
            
            # Show statistics
            print(f"\n   📈 Data Statistics (seed={seed}):")
            print(f"      Non-zero pixels: {np.count_nonzero(sample_data):,} / {sample_data.size:,}")
            print(f"      Mean reflectivity: {sample_data.mean():.2f} dBZ")
            print(f"      Max reflectivity: {sample_data.max():.2f} dBZ")
            
            # Threshold analysis
            high_refl = np.sum(sample_data > 40)
            moderate_refl = np.sum((sample_data > 20) & (sample_data <= 40))
            low_refl = np.sum((sample_data > 5) & (sample_data <= 20))
            
            print(f"      High reflectivity (>40 dBZ): {high_refl:,} pixels")
            print(f"      Moderate reflectivity (20-40 dBZ): {moderate_refl:,} pixels") 
            print(f"      Light reflectivity (5-20 dBZ): {low_refl:,} pixels")
            
            plt.show()  # Display the plot
            return True
            
    except Exception as e:
        print(f"   ❌ Error plotting data: {e}")
        return False

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
    """Main inspection function with controlled seeding"""
    print_banner()
    
    # Find SEVIR files
    sevir_files = find_sevir_files()
    
    if not sevir_files:
        print("\n❌ No SEVIR data files found. Please run download script first.")
        return 1
    
    # Configuration for reproducible analysis
    seed = 42
    num_events = 3
    frames_per_event = 4
    output_dir = "../../results/sevir_dataset"
    
    print(f"\n🎲 Analysis Configuration:")
    print(f"   Random seed: {seed} (notation: s{seed})")
    print(f"   Events to analyze: {num_events}")
    print(f"   Frames per event: {frames_per_event}")
    
    # Inspect each file
    for filepath, size_gb, size_bytes in sevir_files:
        print(f"\n{'='*60}")
        inspect_sevir_structure(filepath)
        analyze_data_loading_strategy(filepath)
        
        # Plot VIL data if available with controlled seeding
        plot_sevir_vil_data(filepath, num_events=num_events, frames_per_event=frames_per_event, seed=seed)
        
        # Only generate code for the largest/main file
        if size_gb > 1.0:  # Only for substantial files
            generate_sample_code(filepath)
    
    # Create efficient data loader
    create_data_loader_script()
    
    print(f"\n🎯 SUMMARY:")
    print(f"   📊 Found {len(sevir_files)} SEVIR data file(s)")
    total_size = sum(size_gb for _, size_gb, _ in sevir_files)
    print(f"   💾 Total size: {total_size:.2f}GB")
    print(f"   🎲 Seed used: {seed} (notation: s{seed})")
    print(f"   ⚡ Recommended approach: Load small batches (10-50 events)")
    print(f"   🧪 Next step: Test with sample data first")
    print(f"   📊 Visualization: VIL plots saved with seed notation")
    
    print(f"\n🚀 Quick test commands:")
    print(f"   python custom/testing/sevir_data_loader.py")
    print(f"   python -c \"exec(open('custom/testing/sevir_data_loader.py').read())\"")
    print(f"\n📊 To plot specific data with seed:")
    print(f"   # In Python:")
    print(f"   from custom.testing.inspect_sevir_data import plot_sevir_vil_data")
    print(f"   plot_sevir_vil_data('path/to/sevir/file.h5', num_events=5, seed={seed})")
    print(f"\n🔍 For consistency with test_actual_sevir.py, use seed={seed}")

if __name__ == "__main__":
    main()
