#!/usr/bin/env python3
"""
SEVIR Analysis Module
Functions for analyzing SEVIR data structure and characteristics
"""

import os
import sys
import h5py
import numpy as np
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import centralized configuration
from config.project_paths import get_paths
paths = get_paths()
paths.setup_python_path()

# Add the core directory to sys.path to import sevir_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from sevir_utils import convert_to_physical_units_with_thresholds


def find_sevir_files(base_dir=None):
    """Find all SEVIR data files"""
    print("\n📁 Searching for SEVIR data files...")
    
    if base_dir is None:
        base_dir = str(paths.data)
    
    sevir_files = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.h5') and ('SEVIR' in file or 'sevir' in file.lower()):
                filepath = os.path.join(root, file)
                size_bytes = os.path.getsize(filepath)
                size_gb = size_bytes / (1024**3)
                sevir_files.append((filepath, size_gb, size_bytes))
    
    if sevir_files:
        print("✅ Found SEVIR data files:")
        for filepath, size_gb, size_bytes in sevir_files:
            rel_path = os.path.relpath(filepath, base_dir)
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
            
            print("   🗂️  File structure:")
            _print_h5_structure("root", f, -1)
            
            # Look for common SEVIR datasets
            _check_common_sevir_datasets(f)
            
            # Check attributes
            _print_file_attributes(f)
            
            return True
            
    except Exception as e:
        print(f"   ❌ Error inspecting file: {e}")
        return False


def analyze_data_loading_strategy(filepath):
    """Analyze optimal data loading strategy"""
    print(f"\n⚡ Analyzing loading strategy for: {os.path.basename(filepath)}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            if 'vil' in f:
                _analyze_vil_dataset(f['vil'], filepath)
            else:
                _analyze_other_datasets(f)
            
            return True
            
    except Exception as e:
        print(f"   ❌ Error analyzing loading strategy: {e}")
        return False


def analyze_vil_units(filepath):
    """Analyze VIL data to determine if it's actually dBZ or kg/m²"""
    print(f"\n🔬 Analyzing VIL units in: {os.path.basename(filepath)}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            if 'vil' in f:
                vil_data = f['vil']
                
                # Sample some data for analysis
                sample = vil_data[0:10] if vil_data.shape[0] >= 10 else vil_data[0:vil_data.shape[0]]
                sample_array = np.array(sample)
                print(f"   📊 Analyzing {sample_array.shape[0]} events for unit detection")
                
                _print_basic_statistics(sample_array)
                unit_type = _determine_unit_type(sample_array)
                _check_zero_percentage(sample_array)
                _print_dataset_attributes(vil_data, f)
                _print_conclusion(sample_array, unit_type)
                
                return sample_array, unit_type
                
            else:
                print("   ❌ No 'vil' dataset found in file")
                return None, None
                
    except Exception as e:
        print(f"   ❌ Error analyzing VIL units: {e}")
        return None, None


def _print_h5_structure(name, obj, level=0):
    """Helper function to print HDF5 structure"""
    indent = "   " * (level + 1)
    if isinstance(obj, h5py.Group):
        print(f"{indent}📁 {name}/ (Group with {len(obj)} items)")
        if level < 3:  # Limit depth to avoid too much output
            for key in list(obj.keys())[:5]:  # Show first 5 items
                _print_h5_structure(key, obj[key], level + 1)
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


def _check_common_sevir_datasets(f):
    """Helper function to check for common SEVIR datasets"""
    common_keys = ['vil', 'ir069', 'ir107', 'lght']
    print(f"\n   🎯 Common SEVIR data types found:")
    for key in common_keys:
        if key in f:
            dataset = f[key]
            print(f"      ✅ {key}: {dataset.shape} {dataset.dtype}")
        else:
            print(f"      ❌ {key}: Not found")


def _print_file_attributes(f):
    """Helper function to print file attributes"""
    if f.attrs:
        print(f"\n   📋 File attributes:")
        for key, value in f.attrs.items():
            print(f"      {key}: {value}")


def _analyze_vil_dataset(vil_data, filepath):
    """Helper function to analyze VIL dataset"""
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
        _sample_data_quality_check(vil_data, filepath)


def _sample_data_quality_check(vil_data, filepath):
    """Helper function to perform data quality check on sample"""
    print(f"\n   🔬 Data quality check (first event):")
    try:
        sample_event = np.array(vil_data[0])
        print(f"      Shape: {sample_event.shape}")
        print(f"      Raw range: {sample_event.min():.2f} to {sample_event.max():.2f}")
        print(f"      Non-zero pixels: {np.count_nonzero(sample_event):,} / {sample_event.size:,}")
        
        # Convert to physical units for quality assessment
        conv_event, _, _, units, thresholds = convert_to_physical_units_with_thresholds(
            sample_event, data_type="auto", filepath=filepath
        )
        
        # Apply appropriate thresholds based on units
        from sevir_utils import print_intensity_breakdown
        print_intensity_breakdown(conv_event.reshape(1, *conv_event.shape), units)
        
    except Exception as e:
        print(f"      ❌ Could not sample data: {e}")


def _analyze_other_datasets(f):
    """Helper function to analyze non-VIL datasets"""
    print("   ❌ No 'vil' dataset found - checking other data types...")
    for key in f.keys():
        if isinstance(f[key], h5py.Dataset):
            dataset = f[key]
            size_gb = dataset.nbytes / (1024**3)
            print(f"      📈 {key}: {dataset.shape} ({size_gb:.2f}GB)")


def _print_basic_statistics(sample_array):
    """Helper function to print basic statistics"""
    print(f"   📈 Data range: {sample_array.min():.2f} to {sample_array.max():.2f}")
    print(f"   📊 Mean value: {sample_array.mean():.2f}")
    print(f"   📉 Standard deviation: {sample_array.std():.2f}")


def _determine_unit_type(sample_array):
    """Helper function to determine unit type from data characteristics"""
    max_val = sample_array.max()
    min_val = sample_array.min()
    
    print(f"\n   🎯 Unit Analysis:")
    
    # Check for dBZ characteristics
    if max_val > 80:
        print(f"      ✅ Likely dBZ: Max value {max_val:.1f} > 80 (typical radar reflectivity)")
        unit_type = "dBZ (radar reflectivity)"
    elif max_val <= 80 and max_val > 50:
        print(f"      🤔 Could be either:")
        print(f"         - dBZ: Max {max_val:.1f} in heavy storm range")
        print(f"         - kg/m²: Max {max_val:.1f} in severe VIL range")
        
        # Additional checks for VIL vs dBZ
        moderate_count = np.sum((sample_array > 20) & (sample_array <= 40))
        high_count = np.sum(sample_array > 40)
        
        if high_count > moderate_count:
            print(f"      🌩️  High values dominate → Likely dBZ")
            unit_type = "dBZ (radar reflectivity)"
        else:
            print(f"      🌧️  Moderate values dominate → Possibly kg/m²")
            unit_type = "possibly kg/m² (VIL)"
    else:
        print(f"      ❓ Unclear: Max {max_val:.1f} could be either unit")
        unit_type = "unclear"
    
    return unit_type


def _check_zero_percentage(sample_array):
    """Helper function to check zero value percentage"""
    zero_count = np.sum(sample_array == 0)
    total_count = sample_array.size
    zero_percentage = (zero_count / total_count) * 100
    print(f"   🌫️  Zero values: {zero_percentage:.1f}% of total pixels")
    
    if zero_percentage > 50:
        print(f"      → High zero percentage typical of radar data")


def _print_dataset_attributes(vil_data, f):
    """Helper function to print dataset and file attributes"""
    print(f"\n   📋 Dataset attributes:")
    if hasattr(vil_data, 'attrs') and len(vil_data.attrs) > 0:
        for key, value in vil_data.attrs.items():
            print(f"      {key}: {value}")
    else:
        print(f"      No attributes found")
    
    print(f"\n   📋 File-level attributes:")
    if len(f.attrs) > 0:
        for key, value in f.attrs.items():
            print(f"      {key}: {value}")
    else:
        print(f"      No file-level attributes found")


def _print_conclusion(sample_array, unit_type):
    """Helper function to print analysis conclusion"""
    min_val = sample_array.min()
    max_val = sample_array.max()
    
    print(f"\n   🏁 CONCLUSION:")
    print(f"      📊 Data range: {min_val:.1f} to {max_val:.1f}")
    print(f"      🎯 Most likely units: {unit_type}")
    
    if "dBZ" in unit_type:
        print(f"      💡 Recommendation: Treat as radar reflectivity (dBZ)")
        print(f"         - Use for precipitation intensity mapping")
        print(f"         - 0-20: light, 20-40: moderate, 40+: heavy")
    elif "kg/m²" in unit_type:
        print(f"      💡 Recommendation: Treat as VIL (kg/m²)")
        print(f"         - Use for total liquid water content")
        print(f"         - 0-20: light, 20-50: moderate, 50+: heavy")
    else:
        print(f"      💡 Recommendation: Check SEVIR documentation")
