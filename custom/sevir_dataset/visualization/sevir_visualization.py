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
from datetime import datetime
import random

# Add the core directory to sys.path to import sevir_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from sevir_utils import (
    convert_to_physical_units_with_thresholds,
    get_weather_category_counts,
    calculate_statistics,
    determine_data_type_from_filepath
)


def plot_sevir_vil_data(filepath, num_events=3, frames_per_event=4, seed=42, output_dir=None):
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
            random.seed(seed)
            np.random.seed(seed)
            
            available_indices = list(range(vil_data.shape[0]))
            selected_indices = sorted(random.sample(available_indices, max_events))
            sample_data = np.array(vil_data[selected_indices])  # Ensure numpy array
            print(f"   🎲 Randomly selected events (seed={seed}): {selected_indices}")
            
            print(f"   📊 Loaded {max_events} events for visualization")
            print(f"   🌩️  Raw data range: {sample_data.min():.2f} to {sample_data.max():.2f}")
            
            # Convert to physical units with appropriate thresholds
            converted_data, vmin, vmax, units, thresholds_info = convert_to_physical_units_with_thresholds(
                sample_data, data_type="auto", filepath=filepath
            )
            
            # Create visualization with proper spacing for colorbar
            fig, axes = plt.subplots(max_events, frames_per_event, figsize=(22, 4*max_events))
            if max_events == 1:
                axes = axes.reshape(1, -1)
            
            last_im = None  # Keep track of last image for colorbar
            
            for event_idx in range(max_events):
                event_data = converted_data[event_idx]  # Use converted data
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
            print(f"   ✅ Plot saved to: {full_path}")
            
            # Show statistics with physical units
            _print_sample_statistics(converted_data, units, seed)
            
            plt.close()
            return True
            
    except Exception as e:
        print(f"   ❌ Error plotting data: {e}")
        return False


def plot_vil_distribution(filepath, max_events=50, seed=42, output_dir=None):
    """
    Plot distribution of VIL values to understand data characteristics
    """
    print(f"\n📊 Plotting VIL distribution from: {os.path.basename(filepath)}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            if 'vil' not in f:
                print("   ❌ No 'vil' dataset found in file")
                return False
            
            vil_data = f['vil']
            print(f"   📈 VIL dataset shape: {vil_data.shape}")
            
            # Sample data
            sample_data = _sample_data_for_distribution(vil_data, max_events, seed)
            
            # Convert to physical units
            converted_data, vmin, vmax, units, thresholds_info = convert_to_physical_units_with_thresholds(
                sample_data, data_type="auto", filepath=filepath
            )
            
            # Calculate statistics
            stats = calculate_statistics(converted_data, units)
            _print_distribution_stats(stats)
            
            # Create distribution plots
            fig = _create_distribution_subplots(converted_data, units, stats)
            
            # Save plot
            data_type = determine_data_type_from_filepath(filepath)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vil_distribution_{data_type}_s{seed}_{timestamp}.png"
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                full_path = os.path.join(output_dir, filename)
            else:
                full_path = filename
                
            plt.savefig(full_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Distribution plot saved to: {full_path}")
            
            # Print detailed statistics
            _print_detailed_statistics(stats)
            
            plt.close()
            return True
            
    except Exception as e:
        print(f"   ❌ Error plotting distribution: {e}")
        return False


def _sample_data_for_distribution(vil_data, max_events, seed):
    """Helper function to sample data for distribution analysis"""
    total_events = vil_data.shape[0]
    sample_size = min(max_events, total_events)
    
    # Set random seed for reproducible sampling
    random.seed(seed)
    np.random.seed(seed)
    
    if total_events > sample_size:
        available_indices = list(range(total_events))
        selected_indices = sorted(random.sample(available_indices, sample_size))
        sample_data = np.array(vil_data[selected_indices])
        print(f"   🎲 Randomly selected {sample_size} events (seed={seed}) from {total_events} total")
    else:
        sample_data = np.array(vil_data[:])
        print(f"   📊 Using all {total_events} events")
    
    return sample_data


def _print_distribution_stats(stats):
    """Helper function to print distribution statistics"""
    print(f"   📊 Distribution analysis:")
    print(f"      Total pixels: {stats['total_pixels']:,}")
    print(f"      Non-zero pixels: {stats['non_zero_pixels']:,} ({stats['non_zero_percentage']:.1f}%)")
    print(f"      Zero pixels: {stats['zero_pixels']:,} ({stats['zero_percentage']:.1f}%)")


def _print_detailed_statistics(stats):
    """Helper function to print detailed statistics"""
    if 'mean' in stats:
        units = stats['units']
        print(f"\n   📈 Key Statistics ({units}):")
        print(f"      Mean (non-zero): {stats['mean']:.2f}")
        print(f"      Median (non-zero): {stats['median']:.2f}")
        print(f"      Std Dev (non-zero): {stats['std']:.2f}")
        print(f"      P50: {stats['p50']:.2f}")
        print(f"      P90: {stats['p90']:.2f}")
        print(f"      P95: {stats['p95']:.2f}")
        print(f"      P99: {stats['p99']:.2f}")
        print(f"      Max: {stats['max']:.2f}")


def _print_sample_statistics(converted_data, units, seed):
    """Helper function to print sample statistics"""
    from sevir_utils import print_intensity_breakdown
    
    print(f"\n   📈 Data Statistics (seed={seed}):")
    print(f"      Non-zero pixels: {np.count_nonzero(converted_data):,} / {converted_data.size:,}")
    print(f"      Mean intensity: {converted_data.mean():.2f} {units}")
    print(f"      Max intensity: {converted_data.max():.2f} {units}")
    
    # Apply thresholds based on physical units
    print_intensity_breakdown(converted_data, units)


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
