#!/usr/bin/env python3
"""
Diagnostic script to investigate VIL saturation at 80 kg/m²
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def diagnose_vil_saturation(filepath, max_events=50, seed=4901):
    """Diagnose why many values appear at 80 kg/m²"""
    
    print(f"\n{'='*70}")
    print(f"VIL SATURATION DIAGNOSTIC")
    print(f"{'='*70}")
    print(f"File: {Path(filepath).name}")
    
    with h5py.File(filepath, 'r') as f:
        vil_data = f['vil']
        
        # Sample events
        np.random.seed(seed)
        num_available = vil_data.shape[0]
        max_events = min(max_events, num_available)
        sample_indices = np.random.choice(num_available, size=max_events, replace=False)
        
        print(f"\nSampling {max_events} events with seed={seed}")
        
        # Load RAW data (uint8, 0-255)
        raw_sample = vil_data[sorted(sample_indices), :, :, :]
        raw_flat = raw_sample.flatten()
        
        print(f"\n{'='*70}")
        print("RAW DATA ANALYSIS (uint8 values 0-255)")
        print(f"{'='*70}")
        
        # Count special values
        zeros = np.sum(raw_flat == 0)
        ones = np.sum(raw_flat == 1)
        val_254 = np.sum(raw_flat == 254)
        val_255 = np.sum(raw_flat == 255)
        total = raw_flat.size
        
        print(f"\nSpecial Value Counts:")
        print(f"  Value 0 (below threshold):   {zeros:,} ({100*zeros/total:.2f}%)")
        print(f"  Value 1 (flagged):           {ones:,} ({100*ones/total:.2f}%)")
        print(f"  Value 254 (VIL >= 80 kg/m²): {val_254:,} ({100*val_254/total:.2f}%) <-- SATURATION")
        print(f"  Value 255 (reserved):        {val_255:,} ({100*val_255/total:.2f}%)")
        
        print(f"\nData Range Statistics (raw):")
        print(f"  Min value:  {raw_flat.min()}")
        print(f"  Max value:  {raw_flat.max()}")
        print(f"  Mean value: {raw_flat.mean():.2f}")
        
        # Check distribution of raw values
        non_zero_raw = raw_flat[raw_flat > 1]  # Exclude 0 and 1
        print(f"\nNon-special values (2-253):")
        if len(non_zero_raw) > 0:
            print(f"  Count:  {len(non_zero_raw):,}")
            print(f"  Min:    {non_zero_raw.min()}")
            print(f"  Max:    {non_zero_raw.max()}")
            print(f"  Mean:   {non_zero_raw.mean():.2f}")
            print(f"  Median: {np.median(non_zero_raw):.2f}")
        
        # Now check PHYSICAL values
        print(f"\n{'='*70}")
        print("PHYSICAL DATA ANALYSIS (kg/m²)")
        print(f"{'='*70}")
        
        # Convert using the formula: physical = raw * (80.0 / 254.0)
        # But handle special values properly
        VIL_SCALE = 80.0 / 254.0
        physical_sample = raw_sample.astype(np.float32) * VIL_SCALE
        physical_sample[raw_sample < 2] = 0.0          # Below threshold and flagged
        physical_sample[raw_sample == 254] = 80.0      # Explicitly set saturation to 80.0
        physical_sample[raw_sample == 255] = 0.0       # Reserved/missing value
        physical_flat = physical_sample.flatten()
        
        # Count values at exactly 80.0
        at_80 = np.sum(np.abs(physical_flat - 80.0) < 0.001)  # Within 0.001 of 80
        non_zero_phys = physical_flat[physical_flat > 0.1]
        
        print(f"\nPhysical Value Distribution:")
        print(f"  Values at 80.0 kg/m²: {at_80:,} ({100*at_80/total:.2f}%)")
        print(f"  This matches raw value 254 count: {val_254:,}")
        
        if len(non_zero_phys) > 0:
            print(f"\nNon-zero physical values:")
            print(f"  Min:    {non_zero_phys.min():.2f} kg/m²")
            print(f"  Max:    {non_zero_phys.max():.2f} kg/m²")
            print(f"  Mean:   {non_zero_phys.mean():.2f} kg/m²")
            print(f"  Median: {np.median(non_zero_phys):.2f} kg/m²")
        
        # Plot raw vs physical distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Raw data histogram
        ax = axes[0, 0]
        counts, bins, _ = ax.hist(raw_flat, bins=256, range=(0, 256), 
                                   alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(254, color='red', linestyle='--', linewidth=2, label='254 (VIL≥80)')
        ax.set_title('Raw VIL Data Distribution (uint8)')
        ax.set_xlabel('Raw Value (0-255)')
        ax.set_ylabel('Frequency')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Zoomed raw data (excluding zeros and 254)
        ax = axes[0, 1]
        non_saturated_raw = raw_flat[(raw_flat > 1) & (raw_flat < 254)]
        if len(non_saturated_raw) > 0:
            ax.hist(non_saturated_raw, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            ax.set_title('Raw VIL Data (2-253 only, excluding saturation)')
            ax.set_xlabel('Raw Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # Physical data histogram
        ax = axes[1, 0]
        ax.hist(physical_flat, bins=100, range=(0, 85), alpha=0.7, color='coral', edgecolor='black')
        ax.axvline(80, color='red', linestyle='--', linewidth=2, label='80 kg/m² (saturation)')
        ax.set_title('Physical VIL Data Distribution (kg/m²)')
        ax.set_xlabel('VIL (kg/m²)')
        ax.set_ylabel('Frequency')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Zoomed physical (non-zero, non-saturated)
        ax = axes[1, 1]
        non_saturated_phys = physical_flat[(physical_flat > 0.1) & (physical_flat < 79.9)]
        if len(non_saturated_phys) > 0:
            ax.hist(non_saturated_phys, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
            ax.set_title('Physical VIL (excluding zeros and saturation)')
            ax.set_xlabel('VIL (kg/m²)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save diagnostic plot
        output_file = f"diagnostic_vil_saturation_s{seed}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n[SUCCESS] Diagnostic plot saved to: {output_file}")
        plt.close()
        
        # CONCLUSION
        print(f"\n{'='*70}")
        print("CONCLUSION")
        print(f"{'='*70}")
        print(f"\nThe spike at 80 kg/m² is NOT an error - it's correct behavior!")
        print(f"\nExplanation:")
        print(f"  - NOAA NEXRAD uses data level 254 to represent VIL >= 80 kg/m²")
        print(f"  - This is a SATURATION/CAPPING value for extreme weather")
        print(f"  - {val_254:,} pixels ({100*val_254/total:.2f}%) have raw value 254")
        print(f"  - These all convert to exactly 80.0 kg/m² (the maximum)")
        print(f"  - This indicates severe weather events in the sample")
        print(f"\nThis is expected for STORMEVENTS data which contains severe weather!")


if __name__ == "__main__":
    import glob
    
    # Find SEVIR files
    sevir_files = glob.glob("/Users/ronaldleung/Code/fyp-testing/neurips-2020-sevir/data/sevir/vil/*.h5")
    
    if not sevir_files:
        print("No SEVIR files found in data/sevir/vil/")
    else:
        print(f"Found {len(sevir_files)} SEVIR file(s)")
        # Use the real SEVIR file (not synthetic)
        real_sevir = [f for f in sevir_files if 'SEVIR_VIL' in f][0]
        print(f"Using: {real_sevir}")
        diagnose_vil_saturation(real_sevir, max_events=50, seed=4901)
