#!/usr/bin/env python3
"""
Inspect NPZ cache files - view metadata and visualize samples.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import centralized configuration
from config.project_paths import get_paths

# Initialize paths
paths = get_paths()
paths.setup_python_path()

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️  NumPy not available - install with: pip install numpy")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def inspect_npz(npz_path: Path, visualize: bool = False, sample_idx: int = 0):
    """Inspect contents of an NPZ file."""
    if not HAS_NUMPY:
        print("❌ NumPy is required for this script")
        return
    
    print(f"🔍 Inspecting: {npz_path}")
    print("=" * 60)
    
    if not npz_path.exists():
        print(f"❌ File not found: {npz_path}")
        return
    
    # Load the file
    data = np.load(npz_path)  # type: ignore
    
    # Show available keys
    print(f"\n📊 Available arrays: {list(data.keys())}")
    
    # Show shape and stats for each array
    for key in data.keys():
        arr = data[key]
        print(f"\n🔹 {key}:")
        print(f"   Shape: {arr.shape}")
        print(f"   Dtype: {arr.dtype}")
        print(f"   Range: [{arr.min():.4f}, {arr.max():.4f}]")
        print(f"   Mean: {arr.mean():.4f}, Std: {arr.std():.4f}")
        
        # Convert to dBZ if normalized
        if arr.min() >= -2 and arr.max() <= 2:  # Likely normalized
            arr_dbz = arr * paths.sevir_scale + paths.sevir_mean
            print(f"   Range (dBZ): [{arr_dbz.min():.1f}, {arr_dbz.max():.1f}]")
    
    # Visualize if requested
    if visualize and 'inputs' in data and 'targets' in data:
        visualize_sample(data, sample_idx, npz_path.parent)
    
    data.close()


def visualize_sample(data, sample_idx: int, output_dir: Path):
    """Visualize a single sample from the cached data."""
    if not HAS_MATPLOTLIB:
        print("⚠️  Matplotlib not available - install with: pip install matplotlib")
        return
    
    X = data['inputs']
    y = data['targets']
    
    if sample_idx >= len(X):
        print(f"⚠️  Sample {sample_idx} out of range (max: {len(X)-1})")
        return
    
    print(f"\n🎨 Visualizing sample {sample_idx}...")
    
    # Get sample and convert to dBZ
    input_frames = X[sample_idx] * paths.sevir_scale + paths.sevir_mean
    target_frames = y[sample_idx] * paths.sevir_scale + paths.sevir_mean
    
    # Select key frames to display
    input_indices = [0, 6, 12]  # First, middle, last of 13 frames
    target_indices = [0, 5, 11]  # First, middle, last of 12 frames
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # type: ignore
    
    # Plot input frames
    for i, idx in enumerate(input_indices):
        ax = axes[0, i]
        im = ax.imshow(input_frames[:, :, idx], cmap='viridis', vmin=0, vmax=70)
        ax.set_title(f'Input t={idx*5}min', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='dBZ')  # type: ignore
    
    # Plot target frames
    for i, idx in enumerate(target_indices):
        ax = axes[1, i]
        im = ax.imshow(target_frames[:, :, idx], cmap='viridis', vmin=0, vmax=70)
        ax.set_title(f'Target t={65+idx*5}min', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='dBZ')  # type: ignore
    
    plt.suptitle(f'Cached SEVIR Event {sample_idx} - Input (13 frames) → Target (12 frames)',   # type: ignore
                 fontsize=12, fontweight='bold')
    plt.tight_layout()  # type: ignore
    
    # Save visualization
    output_path = output_dir / f"inspect_sample_{sample_idx}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')  # type: ignore
    print(f"✅ Saved visualization: {output_path}")
    plt.close()  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Inspect NPZ cache files.")
    parser.add_argument(
        "npz_file",
        type=Path,
        nargs="?",
        default=paths.finetune_cache,
        help="Path to NPZ file (default: cache/finetune_events.npz)",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Generate visualization of a sample",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample index to visualize (default: 0)",
    )
    
    args = parser.parse_args()
    inspect_npz(args.npz_file, args.visualize, args.sample)


if __name__ == "__main__":
    main()
