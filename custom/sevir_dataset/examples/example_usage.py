#!/usr/bin/env python3
"""
Example: Using Modular SEVIR Analysis Components

This script demonstrates how to use the refactored SEVIR analysis modules
for various weather radar data analysis tasks.
"""

# Option 1: Import individual functions as needed
from sevir_utils import convert_to_physical_units_with_thresholds, calculate_statistics
from sevir_visualization import plot_sevir_vil_data, plot_vil_distribution
from sevir_analysis import find_sevir_files, inspect_sevir_structure

# Option 2: Import everything from the package (alternative)
# from sevir_package import *

import os

def example_basic_analysis():
    """Example: Basic file analysis and statistics"""
    print("=== Basic SEVIR File Analysis ===")
    
    # Find SEVIR files
    data_dir = "/Users/ronaldleung/Code/fyp-testing/neurips-2020-sevir/data/sevir"
    file_info = find_sevir_files(data_dir)
    
    if file_info:
        # Analyze first file
        file_path, size_gb, size_bytes = file_info[0]
        print(f"Analyzing: {os.path.basename(file_path)} ({size_gb:.2f}GB)")
        inspect_sevir_structure(file_path)
    else:
        print("No SEVIR files found")

def example_visualization():
    """Example: Create visualizations"""
    print("\n=== SEVIR Visualization Example ===")
    
    # This would be called with actual data
    # plot_sevir_vil_data(vil_data, timestamps, "Synthetic SEVIR Data", "example_plot.png")
    # plot_vil_distribution(vil_data, "VIL Distribution Example", "distribution.png")
    print("Visualization functions available:")
    print("- plot_sevir_vil_data(): For time series weather data")
    print("- plot_vil_distribution(): For statistical distributions")

def example_unit_conversion():
    """Example: Unit conversion and statistics"""
    print("\n=== Unit Conversion Example ===")
    
    # Example with dummy data (replace with real data)
    import numpy as np
    dummy_data = np.random.rand(4, 192, 192) * 50  # Dummy VIL data
    
    # Convert units
    result = convert_to_physical_units_with_thresholds(
        dummy_data, 
        data_type="synthetic"
    )
    converted_data, min_val, max_val, units, category_info = result
    
    # Calculate statistics  
    stats = calculate_statistics(converted_data, units)
    
    print(f"Original shape: {dummy_data.shape}")
    print(f"Converted units: {units}")
    print(f"Value range: {min_val:.2f} - {max_val:.2f} {units}")
    print(f"Statistics: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")

def main():
    """Run all examples"""
    print("SEVIR Modular Analysis Examples")
    print("=" * 40)
    
    example_basic_analysis()
    example_visualization() 
    example_unit_conversion()
    
    print("\n=== Available Modules ===")
    print("sevir_utils.py     - Unit conversion, statistics, weather categorization")
    print("sevir_analysis.py  - File inspection, data structure analysis")
    print("sevir_visualization.py - Plotting and visualization functions")
    print("sevir_package.py   - Convenient imports")
    print("\nMain script: inspect_sevir_data_refactored.py")

if __name__ == "__main__":
    main()
