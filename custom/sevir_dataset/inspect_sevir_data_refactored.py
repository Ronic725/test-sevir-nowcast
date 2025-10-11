#!/usr/bin/env python3
"""
SEVIR Data Inspector - Refactored Main Script
Inspect the downloaded SEVIR data without loading everything into memory
"""

import os
import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import centralized configuration
from config.project_paths import get_paths
paths = get_paths()
paths.setup_python_path()

# Import the modular components
from analysis.sevir_analysis import find_sevir_files, inspect_sevir_structure, analyze_data_loading_strategy, analyze_vil_units
from visualization.sevir_visualization import plot_sevir_vil_data, plot_vil_distribution


def print_banner():
    """Print banner"""
    print("🔍 SEVIR Data Inspector")
    print("=" * 60)
    print("Analyzing downloaded SEVIR data structure and content")
    print("=" * 60)


def print_summary(sevir_files, seed):
    """Print analysis summary"""
    print(f"\n🎯 SUMMARY:")
    print(f"   📊 Found {len(sevir_files)} SEVIR data file(s)")
    total_size = sum(size_gb for _, size_gb, _ in sevir_files)
    print(f"   💾 Total size: {total_size:.2f}GB")
    print(f"   🎲 Seed used: {seed} (notation: s{seed})")
    print(f"   ⚡ Recommended approach: Load small batches (10-50 events)")
    print(f"   🧪 Next step: Test with sample data first")
    print(f"   📊 Visualization: VIL plots saved with seed notation")
    
    print(f"\n🚀 Quick test commands:")
    print(f"   # Use existing data loader:")
    print(f"   python custom/sevir_dataset/core/sevir_data_loader.py")
    print(f"   # Or import in Python:")
    print(f"   from custom.sevir_dataset.core.sevir_data_loader import SEVIRDataLoader")
    print(f"   # Example usage:")
    print(f"   with SEVIRDataLoader('data/path/file.h5') as loader:")
    print(f"       sample = loader.get_sample(10)  # Load 10 events")
    print(f"\n📊 To plot specific data with seed:")
    print(f"   # In Python:")
    print(f"   from custom.sevir_dataset.visualization.sevir_visualization import plot_sevir_vil_data")
    print(f"   plot_sevir_vil_data('path/to/sevir/file.h5', num_events=5, seed={seed})")
    print(f"\n🔍 For consistency with test_actual_sevir.py, use seed={seed}")
    print(f"\n💡 The existing SEVIRDataLoader class is ready to use and provides:")
    print(f"   ✅ Memory-efficient batch loading")
    print(f"   ✅ Context manager support (with/as)")
    print(f"   ✅ Automatic file handling")
    print(f"   ✅ Configurable batch sizes")


def analyze_single_file(filepath, size_gb, config):
    """Analyze a single SEVIR file"""
    print(f"\n{'='*60}")
    
    # Structural analysis
    inspect_sevir_structure(filepath)
    analyze_data_loading_strategy(filepath)
    
    # Unit analysis
    analyze_vil_units(filepath)
    
    # Visualization
    plot_sevir_vil_data(
        filepath, 
        num_events=config['num_events'], 
        frames_per_event=config['frames_per_event'], 
        seed=config['seed'], 
        output_dir=config['output_dir']
    )
    
    # Distribution analysis
    plot_vil_distribution(
        filepath, 
        max_events=config['distribution_events'], 
        seed=config['seed'], 
        output_dir=config['output_dir']
    )
    
    # Warning for large files
    if size_gb > 1.0:
        print("CAUTION: This is a large file with size >1GB. We might want to batch load the data with sevir_data_loader.py")


def main():
    """Main inspection function with controlled seeding and modular structure"""
    print_banner()
    
    # Find SEVIR files
    sevir_files = find_sevir_files()
    
    if not sevir_files:
        print("\n❌ No SEVIR data files found. Please run download script first.")
        return 1
    
    # Configuration for reproducible analysis
    config = {
        'seed': 42,
        'num_events': 3,
        'frames_per_event': 4,
        'distribution_events': 50,
        'output_dir': str(paths.root / 'custom' / 'results' / 'sevir_dataset')
    }
    
    print(f"\n🎲 Analysis Configuration:")
    print(f"   Random seed: {config['seed']} (notation: s{config['seed']})")
    print(f"   Events to analyze: {config['num_events']}")
    print(f"   Frames per event: {config['frames_per_event']}")
    print(f"   Distribution events: {config['distribution_events']}")
    
    # Analyze each file
    for filepath, size_gb, size_bytes in sevir_files:
        analyze_single_file(filepath, size_gb, config)
    
    # Print summary
    print_summary(sevir_files, config['seed'])
    
    return 0


if __name__ == "__main__":
    exit(main())
