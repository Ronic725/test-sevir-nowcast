#!/usr/bin/env python3
"""
SEVIR Utilities Module
Utility functions for SEVIR data analysis and conversion
"""

import os
import numpy as np


def convert_to_physical_units_with_thresholds(data, data_type="auto", filepath=None):
    """
    Convert SEVIR data to appropriate physical units and apply smart thresholds for visualization
    
    Args:
        data: numpy array of SEVIR data
        data_type: "auto", "vil_kg", "dBZ", or "synthetic"
        filepath: optional filepath to auto-detect data type
    
    Returns:
        tuple: (converted_data, vmin, vmax, units, thresholds_info)
    """
    
    # Auto-detect data type if not specified
    if data_type == "auto":
        if filepath:
            filename = os.path.basename(filepath).lower()
            if 'synthetic' in filename:
                data_type = "vil_kg"
            elif 'sevir_vil_stormevents' in filename:
                data_type = "dBZ"
            else:
                # Use data characteristics to guess
                max_val = float(data.max())
                if max_val > 100:
                    data_type = "dBZ"
                else:
                    data_type = "vil_kg"
        else:
            # Use data range to guess
            max_val = float(data.max())
            data_type = "dBZ" if max_val > 100 else "vil_kg"
    
    # Convert based on detected/specified type
    if data_type in ["vil_kg", "synthetic"]:
        # VIL data in kg/m² (synthetic data)
        converted_data = data.copy()  # Already in kg/m²
        units = "kg/m²"
        
        # VIL thresholds for weather intensity
        light_threshold = 20      # Light precipitation
        moderate_threshold = 45   # Moderate precipitation  
        heavy_threshold = 65      # Heavy precipitation
        
        # Set visualization range
        vmin = 0
        vmax = min(80, max(data.max(), 70))  # Cap at reasonable VIL maximum
        
        thresholds_info = {
            "light": f"0-{light_threshold} {units} (light)",
            "moderate": f"{light_threshold}-{moderate_threshold} {units} (moderate)", 
            "heavy": f"{moderate_threshold}-{heavy_threshold} {units} (heavy)",
            "severe": f"{heavy_threshold}+ {units} (severe)"
        }
        
    elif data_type == "dBZ":
        # Radar reflectivity in dBZ (real SEVIR data)
        converted_data = data.copy()  # Already in dBZ
        units = "dBZ"
        
        # dBZ thresholds for precipitation intensity
        light_threshold = 20      # Light rain
        moderate_threshold = 35   # Moderate rain
        heavy_threshold = 50      # Heavy rain/storms
        severe_threshold = 65     # Severe storms
        
        # Set visualization range - cap very high values for better contrast
        vmin = 0
        data_max = float(data.max())
        if data_max > 80:
            vmax = min(80, data_max)  # Cap at 80 dBZ for better visualization of most weather
        else:
            vmax = data_max
            
        thresholds_info = {
            "light": f"0-{light_threshold} {units} (light rain)",
            "moderate": f"{light_threshold}-{moderate_threshold} {units} (moderate rain)", 
            "heavy": f"{moderate_threshold}-{heavy_threshold} {units} (heavy rain)",
            "severe": f"{heavy_threshold}+ {units} (severe storms)"
        }
    
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Use 'auto', 'vil_kg', or 'dBZ'")
    
    print(f"   🔄 Converted to {units}: range [{converted_data.min():.1f}, {converted_data.max():.1f}]")
    print(f"   📊 Visualization range: [{vmin:.1f}, {vmax:.1f}] {units}")
    print(f"   🌡️  Intensity thresholds:")
    for category, info in thresholds_info.items():
        print(f"      • {info}")
    
    return converted_data, vmin, vmax, units, thresholds_info


def get_weather_category_counts(data, units):
    """
    Get pixel counts for different weather intensity categories
    
    Args:
        data: numpy array of converted data
        units: "kg/m²" or "dBZ"
    
    Returns:
        tuple: (categories, counts, colors, category_labels)
    """
    flat_data = data.flatten()
    
    if units == "kg/m²":
        # VIL categories
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
    else:
        # dBZ categories
        categories = ['0', '0-5', '5-20', '20-35', '35-50', '>50']
        counts = [
            np.sum(flat_data == 0),
            np.sum((flat_data > 0) & (flat_data <= 5)),
            np.sum((flat_data > 5) & (flat_data <= 20)),
            np.sum((flat_data > 20) & (flat_data <= 35)),
            np.sum((flat_data > 35) & (flat_data <= 50)),
            np.sum(flat_data > 50)
        ]
        colors = ['lightgray', 'lightblue', 'yellow', 'orange', 'red', 'darkred']
        category_labels = ['None', 'Trace', 'Light Rain', 'Moderate Rain', 'Heavy Rain', 'Severe Storms']
    
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
        data: numpy array of converted data
        units: "kg/m²" or "dBZ"
    """
    if units == "kg/m²":
        # VIL thresholds
        light_count = np.sum((data > 5) & (data <= 20))
        moderate_count = np.sum((data > 20) & (data <= 45))
        heavy_count = np.sum((data > 45) & (data <= 65))
        severe_count = np.sum(data > 65)
        
        print(f"      Light VIL (5-20 kg/m²): {light_count:,} pixels")
        print(f"      Moderate VIL (20-45 kg/m²): {moderate_count:,} pixels")
        print(f"      Heavy VIL (45-65 kg/m²): {heavy_count:,} pixels")
        print(f"      Severe VIL (>65 kg/m²): {severe_count:,} pixels")
    else:
        # dBZ thresholds  
        light_count = np.sum((data > 5) & (data <= 20))
        moderate_count = np.sum((data > 20) & (data <= 35))
        heavy_count = np.sum((data > 35) & (data <= 50))
        severe_count = np.sum(data > 50)
        
        print(f"      Light rain (5-20 dBZ): {light_count:,} pixels")
        print(f"      Moderate rain (20-35 dBZ): {moderate_count:,} pixels")
        print(f"      Heavy rain (35-50 dBZ): {heavy_count:,} pixels")
        print(f"      Severe storms (>50 dBZ): {severe_count:,} pixels")


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
