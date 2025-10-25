#!/usr/bin/env python3
"""
SEVIR Visualization Module
Functions for plotting and visualizing SEVIR data
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
import random
from datetime import datetime
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from core utilities
from custom.sevir_dataset.core.sevir_utils import (
    convert_raw_vil_to_physical,
    VIL_SCALE,
    get_weather_category_counts,
    calculate_statistics,
    determine_data_type_from_filepath
)


def plot_sevir_vil_data(filepath, num_events=3, frames_per_event=4, seed=42, output_dir='.'):
    """Plot sample SEVIR VIL data for visualization with controlled seeding"""
    print(f"\n[INFO] Plotting SEVIR VIL data from: {os.path.basename(filepath)}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            if 'vil' not in f:
                print("   [ERROR] No 'vil' dataset found in file")
                return False
            
            vil_data = f['vil']
            print(f"   [INFO] VIL dataset shape: {vil_data.shape}")
            
            # Use seeded random sampling for consistency
            max_events = min(num_events, vil_data.shape[0])
            
            # Set random seed for reproducible sampling
            random.seed(seed)
            np.random.seed(seed)
            
            available_indices = list(range(vil_data.shape[0]))
            selected_indices = sorted(random.sample(available_indices, max_events))
            raw_sample_data = np.array(vil_data[selected_indices])
            print(f"   [INFO] Randomly selected events (seed={seed}): {selected_indices}")
            
            # Convert raw data to physical units
            sample_data = convert_raw_vil_to_physical(raw_sample_data)
            
            print(f"   [INFO] Loaded {max_events} events for visualization")
            print(f"   [INFO] Physical data range: {sample_data.min():.2f} to {sample_data.max():.2f} kg/m²")
            
            units = "kg/m²"
            vmin, vmax = 0, 80  # Standard VIL range
            
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
                    
                    # Plot with physical units and proper thresholds
                    im = ax.imshow(frame, cmap='viridis', vmin=vmin, vmax=vmax)
                    ax.set_title(title, fontsize=10)
                    ax.axis('off')
                    last_im = im  # Keep reference for colorbar
            
            # Manually adjust subplot layout to leave space for colorbar
            plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05, 
                               wspace=0.1, hspace=0.2)
            
            # Add colorbar in the reserved space with correct units
            if last_im is not None:
                cbar = fig.colorbar(last_im, ax=axes, label=f'Intensity ({units})', 
                                   shrink=0.9, aspect=25)
            
            # Save plot
            data_type = determine_data_type_from_filepath(filepath)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sevir_vil_plot_{data_type}_s{seed}_{timestamp}.png"
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                full_path = os.path.join(output_dir, filename)
            else:
                full_path = filename

            plt.savefig(full_path, dpi=150, bbox_inches='tight')
            print(f"   [SUCCESS] Plot saved to: {full_path}")
            
            # Show statistics with physical units
            _print_sample_statistics(sample_data, units, seed)
            
            plt.close()
            return True
            
    except Exception as e:
        print(f"   [ERROR] Error plotting data: {e}")
        return False


def plot_vil_distribution(filepath, max_events=50, seed=42, output_dir='.'):
    """Plot distribution of VIL values"""
    print(f"\n[INFO] Plotting VIL distribution for: {os.path.basename(filepath)}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            if 'vil' not in f:
                print("   [ERROR] No 'vil' dataset found in file")
                return False
            
            vil_data = f['vil']
            print(f"   [INFO] VIL dataset shape: {vil_data.shape}")
            
            # Ensure we don't exceed available events
            num_available = vil_data.shape[0]
            if max_events > num_available:
                print(f"   [INFO] max_events ({max_events}) > available ({num_available}). Using {num_available}.")
                max_events = num_available
            
            # Sample events for distribution analysis
            np.random.seed(seed)
            sample_indices = np.random.choice(num_available, size=max_events, replace=False)
            print(f"   [INFO] Randomly selected {max_events} events (seed={seed}) from {num_available} total")
            
            # Load raw data and convert to physical units
            raw_sample = vil_data[sorted(sample_indices), :, :, :]
            physical_sample = convert_raw_vil_to_physical(raw_sample)
            
            # Flatten the array and filter out zero/NaN values for a cleaner plot
            flat_data = physical_sample.flatten()
            flat_data = flat_data[flat_data > 0.1]  # Ignore near-zero values
            
            if flat_data.size == 0:
                print("   [INFO] No significant VIL data to plot after filtering.")
                return False

            # Create plot
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.histplot(flat_data, bins=50, kde=True, ax=ax, color='skyblue')
            
            ax.set_title(f'Distribution of VIL Values (Physical Units)\n{os.path.basename(filepath)}', fontsize=16)
            ax.set_xlabel('VIL (kg/m²)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            # Add statistics
            mean_val = np.mean(flat_data)
            median_val = np.median(flat_data)
            ax.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='g', linestyle='-', label=f'Median: {median_val:.2f}')
            ax.legend()
            
            # Save plot
            data_type = determine_data_type_from_filepath(filepath)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vil_distribution_{data_type}_s{seed}_{timestamp}.png"
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                full_path = os.path.join(output_dir, filename)
            else:
                full_path = filename
                
            plt.tight_layout()
            plt.savefig(full_path, dpi=150)
            print(f"   [SUCCESS] Distribution plot saved to: {full_path}")
            plt.close(fig)
            
            return True
            
    except Exception as e:
        print(f"   [ERROR] Error plotting distribution: {e}")
        return False


def _print_sample_statistics(sample_data, units, seed):
    """Helper function to print sample statistics"""
    from custom.sevir_dataset.core.sevir_utils import print_intensity_breakdown
    
    print(f"\n   [INFO] Data Statistics (seed={seed}):")
    print(f"      Non-zero pixels: {np.count_nonzero(sample_data):,} / {sample_data.size:,}")
    print(f"      Mean intensity: {sample_data.mean():.2f} {units}")
    print(f"      Max intensity: {sample_data.max():.2f} {units}")
    
    # Print breakdown by intensity
    print_intensity_breakdown(sample_data, units)


def _create_distribution_subplots(converted_data, units, stats):
    """Helper function to create distribution subplots"""
    flat_data = converted_data.flatten()
    non_zero_data = flat_data[flat_data > 0]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Full distribution (including zeros)
    axes[0, 0].hist(flat_data, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title(f'Full Distribution (including zeros)\n{units}')
    axes[0, 0].set_xlabel(f'Value ({units})')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Non-zero distribution (log scale)
    if len(non_zero_data) > 0:
        axes[0, 1].hist(non_zero_data, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title(f'Non-zero Distribution\n{units}')
        axes[0, 1].set_xlabel(f'Value ({units})')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No non-zero data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Non-zero Distribution')
    
    # 3. Cumulative distribution
    _create_cumulative_plot(axes[1, 0], non_zero_data, units)
    
    # 4. Intensity categories bar chart
    _create_category_bar_chart(axes[1, 1], flat_data, units)
    
    plt.tight_layout()
    return fig


def _create_cumulative_plot(ax, non_zero_data, units):
    """Helper function to create cumulative distribution plot"""
    if len(non_zero_data) > 0:
        sorted_data = np.sort(non_zero_data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
        ax.plot(sorted_data, cumulative, linewidth=2, color='green')
        
        # Add percentile lines
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(sorted_data, p)
            ax.axvline(val, color='red', linestyle='--', alpha=0.7)
            ax.text(val, p, f'P{p}', rotation=90, va='bottom', ha='right')
    else:
        sorted_data = np.array([0])
        
    ax.set_title(f'Cumulative Distribution (Non-zero)\n{units}')
    ax.set_xlabel(f'Value ({units})')
    ax.set_ylabel('Cumulative Percentage (%)')
    ax.grid(True, alpha=0.3)


def _create_category_bar_chart(ax, flat_data, units):
    """Helper function to create weather category bar chart"""
    categories, counts, colors, category_labels = get_weather_category_counts(flat_data, units)
    
    bars = ax.bar(range(len(categories)), counts, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title(f'Weather Intensity Categories\n{units}')
    ax.set_xlabel('Category')
    ax.set_ylabel('Pixel Count')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([f'{cat}\n{label}' for cat, label in zip(categories, category_labels)], 
                      rotation=45, ha='right')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{count:,}', ha='center', va='bottom', fontsize=8)
