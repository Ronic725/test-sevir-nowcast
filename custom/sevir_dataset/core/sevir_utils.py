#!/usr/bin/env python3
"""
SEVIR Utilities Module
Utility functions for SEVIR data analysis and conversion
Based on NOAA NEXRAD Level III VIL product specification
"""

import os
import numpy as np

# VIL Conversion Constants (from NOAA NEXRAD product 134 specification)
# Data levels 2-254 represent VIL values 0-80 kg/m²
# Data level 254 corresponds to VIL >= 80 kg/m²
# Data levels 0 and 1 are "Below threshold" and "flagged data"
VIL_SCALE = 80.0 / 254.0  # Approximately 0.315 kg/m² per data level


def convert_raw_vil_to_physical(raw_data):
    """
    Convert raw VIL data (uint8, 0-255) to physical units (kg/m²)
    Based on NOAA NEXRAD Level III product 134 specification
    
    Data Level Mapping:
    - 0: Below threshold → 0.0 kg/m²
    - 1: Flagged data → 0.0 kg/m²
    - 2-253: Linear mapping → (level * 80.0/254.0) kg/m²
    - 254: VIL >= 80 kg/m² (saturation) → 80.0 kg/m²
    - 255: Reserved/missing → 0.0 kg/m² (or NaN for strict handling)
    
    Args:
        raw_data: numpy array of raw VIL data (0-255)
    
    Returns:
        numpy array of VIL in kg/m²
    """
    # Convert to float for calculation
    physical_data = raw_data.astype(np.float32) * VIL_SCALE
    
    # Handle special values according to NOAA specification
    physical_data[raw_data < 2] = 0.0          # Below threshold and flagged data
    physical_data[raw_data == 254] = 80.0      # Explicitly set saturation to 80.0
    physical_data[raw_data == 255] = 0.0       # Reserved/missing value → treat as zero
    
    return physical_data


def convert_to_physical_units_with_thresholds(data, data_type="vil", filepath=None):
    """
    Convert SEVIR data to physical units and apply thresholds for visualization
    
    Args:
        data: numpy array of SEVIR data (raw uint8 or already converted)
        data_type: "vil" (raw VIL data from SEVIR) or "vil_physical" (already converted)
        filepath: optional filepath for context
    
    Returns:
        tuple: (converted_data, vmin, vmax, units, thresholds_info)
    """
    
    if data_type == "vil":
        # Raw VIL data - convert to physical units
        converted_data = convert_raw_vil_to_physical(data)
        units = "kg/m²"
        
        # VIL thresholds for weather intensity (based on meteorological standards)
        light_threshold = 20      # Light precipitation
        moderate_threshold = 45   # Moderate precipitation  
        heavy_threshold = 65      # Heavy precipitation / possible hail
        
        # Set visualization range (0-80 kg/m² is the standard VIL range)
        vmin = 0
        vmax = 80
        
        thresholds_info = {
            "light": f"0-{light_threshold} {units} (light)",
            "moderate": f"{light_threshold}-{moderate_threshold} {units} (moderate)", 
            "heavy": f"{moderate_threshold}-{heavy_threshold} {units} (heavy)",
            "severe": f"{heavy_threshold}+ {units} (severe/hail likely)"
        }
        
    elif data_type == "vil_physical":
        # Already in physical units
        converted_data = data.copy()
        units = "kg/m²"
        
        # VIL thresholds
        light_threshold = 20
        moderate_threshold = 45
        heavy_threshold = 65
        
        vmin = 0
        vmax = 80
        
        thresholds_info = {
            "light": f"0-{light_threshold} {units} (light)",
            "moderate": f"{light_threshold}-{moderate_threshold} {units} (moderate)", 
            "heavy": f"{moderate_threshold}-{heavy_threshold} {units} (heavy)",
            "severe": f"{heavy_threshold}+ {units} (severe/hail likely)"
        }
    
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Use 'vil' or 'vil_physical'")
    
    print(f"   [INFO] Converted to {units}: range [{converted_data.min():.1f}, {converted_data.max():.1f}]")
    print(f"   [INFO] Visualization range: [{vmin:.1f}, {vmax:.1f}] {units}")
    print(f"   [INFO] Intensity thresholds:")
    for category, info in thresholds_info.items():
        print(f"      - {info}")
    
    return converted_data, vmin, vmax, units, thresholds_info


def get_weather_category_counts(data, units):
    """
    Get pixel counts for different weather intensity categories
    
    Args:
        data: numpy array of converted data (in kg/m²)
        units: "kg/m²"
    
    Returns:
        tuple: (categories, counts, colors, category_labels)
    """
    flat_data = data.flatten()
    
    # VIL categories (data should be in kg/m²)
    categories = ['0', '0-5', '5-20', '20-45', '45-65', '>65']
    counts = [
        np.sum(flat_data == 0),
        np.sum((flat_data > 0) & (flat_data <= 5)),
        np.sum((flat_data > 5) & (flat_data <= 20)),
        np.sum((flat_data > 20) & (flat_data <= 45)),
        np.sum((flat_data > 45) & (flat_data <= 65)),
        np.sum(flat_data > 65)
    ]
    colors = ['lightgray', 'lightblue', 'yellow', 'orange', 'red', 'darkred']
    category_labels = ['None', 'Trace', 'Light', 'Moderate', 'Heavy', 'Severe']
    
    return categories, counts, colors, category_labels


def calculate_statistics(data, units):
    """
    Calculate key statistics for the data
    
    Args:
        data: numpy array of data
        units: unit string for display
    
    Returns:
        dict: dictionary with statistics
    """
    flat_data = data.flatten()
    non_zero_data = flat_data[flat_data > 0]
    
    stats = {
        'total_pixels': len(flat_data),
        'non_zero_pixels': len(non_zero_data),
        'zero_pixels': len(flat_data) - len(non_zero_data),
        'zero_percentage': (len(flat_data) - len(non_zero_data)) / len(flat_data) * 100,
        'non_zero_percentage': len(non_zero_data) / len(flat_data) * 100,
        'units': units
    }
    
    if len(non_zero_data) > 0:
        stats.update({
            'mean': non_zero_data.mean(),
            'median': np.median(non_zero_data),
            'std': non_zero_data.std(),
            'min': non_zero_data.min(),
            'max': non_zero_data.max(),
            'p50': np.percentile(non_zero_data, 50),
            'p75': np.percentile(non_zero_data, 75),
            'p90': np.percentile(non_zero_data, 90),
            'p95': np.percentile(non_zero_data, 95),
            'p99': np.percentile(non_zero_data, 99)
        })
    
    return stats


def print_intensity_breakdown(data, units):
    """
    Print breakdown of intensity categories
    
    Args:
        data: numpy array of converted data (in kg/m²)
        units: "kg/m²"
    """
    # VIL thresholds
    light_count = np.sum((data > 5) & (data <= 20))
    moderate_count = np.sum((data > 20) & (data <= 45))
    heavy_count = np.sum((data > 45) & (data <= 65))
    severe_count = np.sum(data > 65)
    
    print(f"      Light VIL (5-20 kg/m²): {light_count:,} pixels")
    print(f"      Moderate VIL (20-45 kg/m²): {moderate_count:,} pixels")
    print(f"      Heavy VIL (45-65 kg/m²): {heavy_count:,} pixels")
    print(f"      Severe VIL (>65 kg/m²): {severe_count:,} pixels")


def determine_data_type_from_filepath(filepath):
    """
    Determine data type from filepath for output naming
    
    Args:
        filepath: path to the data file
        
    Returns:
        str: data type identifier
    """
    filename_base = os.path.basename(filepath).lower()
    
    if 'synthetic' in filename_base or 'test' in filename_base:
        return "synthetic"
    elif 'sevir_vil_stormevents' in filename_base or 'vil' in os.path.dirname(filepath).lower():
        return "true_sevir"
    else:
        return "unknown"
