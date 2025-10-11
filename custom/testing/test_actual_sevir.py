#!/usr/bin/env python3
"""
Test our trained model on actual SEVIR nowcast data
"""

import os
import sys
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import random
from datetime import datetime
from pathlib import Path

# Add project root and src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import centralized configuration
from config.project_paths import get_paths
paths = get_paths()
paths.setup_python_path()

# Try to import the reader, with fallback
try:
    from readers.nowcast_reader import read_data
except ImportError:
    print("⚠️  nowcast_reader not found, using direct H5 loading")
    read_data = None

# SEVIR normalization constants from config
SEVIR_MEAN = paths.sevir_mean
SEVIR_SCALE = paths.sevir_scale

# Project paths from config
DATA_DIR = paths.data
MODELS_DIR = paths.models
RESULTS_DIR = paths.results_dir

# Parameters
NUM_SAMPLES = 5  # Number of samples to test


def load_real_sevir_data_direct(data_file, num_samples=5, seed=42):
    """Load SEVIR data directly from H5 file with seeded random sampling"""
    print(f"📡 Loading SEVIR data directly from: {data_file}")
    
    try:
        with h5py.File(data_file, 'r') as f:
            # Check available datasets
            print(f"Available datasets: {list(f.keys())}")
            
            # Load VIL data - adjust key names based on actual file structure
            if 'vil' in f:
                vil_data = f['vil'][:]
            elif 'VIL' in f:
                vil_data = f['VIL'][:]
            elif 'data' in f:
                vil_data = f['data'][:]
            else:
                # Try first available dataset
                key = list(f.keys())[0]
                print(f"Using dataset: {key}")
                vil_data = f[key][:]
            
            print(f"Original data shape: {vil_data.shape}")
            
            # Set random seed and sample data
            if len(vil_data) > num_samples:
                random.seed(seed)
                np.random.seed(seed)
                available_indices = list(range(len(vil_data)))
                selected_indices = sorted(random.sample(available_indices, num_samples))
                vil_data = vil_data[selected_indices]
                print(f"🎲 Randomly selected events (seed={seed}): {selected_indices}")
            else:
                print(f"📊 Using all {len(vil_data)} available events")
            
            # SEVIR typically has shape (N, 384, 384, 25) for full sequences
            # Split into input (13 frames) and output (12 frames)
            if vil_data.shape[-1] == 25:
                X_test = vil_data[:, :, :, :13]
                y_test = vil_data[:, :, :, 13:]
            elif vil_data.shape[-1] == 49:  # Some files have longer sequences
                X_test = vil_data[:, :, :, :13]
                y_test = vil_data[:, :, :, 13:25]
            else:
                raise ValueError(f"Unexpected time dimension: {vil_data.shape[-1]}")
            
            # Normalize to SEVIR format (subtract mean, divide by scale)
            X_test = (X_test - SEVIR_MEAN) / SEVIR_SCALE
            y_test = (y_test - SEVIR_MEAN) / SEVIR_SCALE
            
            return X_test, y_test
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None


def load_real_sevir_data(num_samples=5, seed=42):
    """Load real SEVIR test data with seeded random sampling"""
    print("📡 Loading real SEVIR nowcast test data...")
    
    # Correct data file path
    data_file = DATA_DIR / "sevir" / "vil" / "SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
    
    if not data_file.exists():
        print(f"❌ SEVIR test data not found: {data_file}")
        print("💡 Please ensure the SEVIR data file is in the correct location")
        return None, None
    
    # Try official reader first, fall back to direct loading
    if read_data is not None:
        try:
            X_test, y_test = read_data(str(data_file), end=num_samples)
            print(f"✅ Loaded using official reader")
        except Exception as e:
            print(f"⚠️  Official reader failed: {e}")
            print("Falling back to direct H5 loading...")
            X_test, y_test = load_real_sevir_data_direct(str(data_file), num_samples, seed)
    else:
        X_test, y_test = load_real_sevir_data_direct(str(data_file), num_samples, seed)
    
    if X_test is not None and y_test is not None:
        print(f"✅ Loaded real SEVIR data:")
        print(f"   Input shape: {X_test.shape}")
        print(f"   Output shape: {y_test.shape}")
        print(f"   Input range: [{X_test.min():.3f}, {X_test.max():.3f}] (normalized)")
        print(f"   Output range: [{y_test.min():.3f}, {y_test.max():.3f}] (normalized)")
        
        return X_test, y_test
    else:
        return None, None


def load_real_sevir_data(num_samples=5):
    """Load real SEVIR test data"""
    print("📡 Loading real SEVIR nowcast test data...")
    
    # Correct data file path
    data_file = DATA_DIR / "sevir" / "vil" / "SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
    
    if not data_file.exists():
        print(f"❌ SEVIR test data not found: {data_file}")
        print("💡 Please ensure the SEVIR data file is in the correct location")
        return None, None
    
    # Try official reader first, fall back to direct loading
    if read_data is not None:
        try:
            X_test, y_test = read_data(str(data_file), end=num_samples)
            print(f"✅ Loaded using official reader")
        except Exception as e:
            print(f"⚠️  Official reader failed: {e}")
            print("Falling back to direct H5 loading...")
            X_test, y_test = load_real_sevir_data_direct(str(data_file), num_samples)
    else:
        X_test, y_test = load_real_sevir_data_direct(str(data_file), num_samples)
    
    if X_test is not None and y_test is not None:
        print(f"✅ Loaded real SEVIR data:")
        print(f"   Input shape: {X_test.shape}")
        print(f"   Output shape: {y_test.shape}")
        print(f"   Input range: [{X_test.min():.3f}, {X_test.max():.3f}] (normalized)")
        print(f"   Output range: [{y_test.min():.3f}, {y_test.max():.3f}] (normalized)")
        
        return X_test, y_test
    else:
        return None, None


def load_our_model(model_name=None):
    """Load our trained model - specify model_name to choose specific model"""
    print("🤖 Loading our trained model...")
    
    # Find all trained model directories
    model_dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith('trained_')]
    
    if not model_dirs:
        print("❌ No trained model found in models directory")
        print(f"💡 Please ensure you have a trained model in: {MODELS_DIR}")
        return None
    
    # List available models
    print("📁 Available trained models:")
    for i, model_dir in enumerate(sorted(model_dirs, key=lambda x: x.stat().st_mtime, reverse=True)):
        print(f"   {i+1}. {model_dir.name}")
    
    # Choose model
    if model_name:
        # Find specific model by name
        selected_dir = None
        for model_dir in model_dirs:
            if model_name in model_dir.name:
                selected_dir = model_dir
                break
        if not selected_dir:
            print(f"❌ Model '{model_name}' not found")
            return None
    else:
        # Use latest (current behavior)
        selected_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
    
    model_path = selected_dir / "model_mse.h5"
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return None
    
    print(f"📁 Loading model from: {model_path}")
    
    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {model.count_params():,}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_on_real_sevir(model, X_test, y_test):
    """Test model on real SEVIR data"""
    print(f"\n🧪 Testing on real SEVIR data...")
    
    try:
        # Make predictions
        y_pred = model.predict(X_test, batch_size=2, verbose=1)
        
        print(f"✅ Predictions completed!")
        print(f"   Prediction shape: {y_pred.shape}")
        print(f"   Prediction range: [{y_pred.min():.3f}, {y_pred.max():.3f}] (normalized)")
        
        # Calculate metrics on normalized data
        mse = np.mean((y_pred - y_test)**2)
        mae = np.mean(np.abs(y_pred - y_test))
        
        # Convert back to dBZ for interpretation
        y_pred_dbz = y_pred * SEVIR_SCALE + SEVIR_MEAN
        y_test_dbz = y_test * SEVIR_SCALE + SEVIR_MEAN
        
        mse_dbz = np.mean((y_pred_dbz - y_test_dbz)**2)
        mae_dbz = np.mean(np.abs(y_pred_dbz - y_test_dbz))
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Normalized scale:")
        print(f"      MSE: {mse:.6f}")
        print(f"      MAE: {mae:.6f}")
        print(f"   dBZ scale:")
        print(f"      MSE: {mse_dbz:.3f}")
        print(f"      MAE: {mae_dbz:.3f}")
        
        return y_pred, mse, mae, y_pred_dbz, y_test_dbz
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None, None, None, None, None

def compare_with_baseline(X_test, y_test):
    """Compare with simple baseline (persistence model)"""
    print(f"\n📊 Comparing with persistence baseline...")
    
    # Persistence model: last input frame repeated for all output frames
    last_frame = X_test[:, :, :, -1:] # Shape: (N, 384, 384, 1)
    y_baseline = np.repeat(last_frame, 12, axis=-1) # Shape: (N, 384, 384, 12)
    
    # Calculate baseline metrics
    mse_baseline = np.mean((y_baseline - y_test)**2)
    mae_baseline = np.mean(np.abs(y_baseline - y_test))
    
    print(f"   Persistence model:")
    print(f"      MSE: {mse_baseline:.6f}")
    print(f"      MAE: {mae_baseline:.6f}")
    
    return y_baseline, mse_baseline, mae_baseline

def visualize_real_sevir_results(X_test, y_test, y_pred, y_baseline=None, sample_idx=0):
    """Visualize results on real SEVIR data"""
    print(f"\n📊 Creating real SEVIR visualization...")
    
    # Convert to dBZ for better visualization
    X_dbz = X_test * SEVIR_SCALE + SEVIR_MEAN
    y_test_dbz = y_test * SEVIR_SCALE + SEVIR_MEAN
    y_pred_dbz = y_pred * SEVIR_SCALE + SEVIR_MEAN
    
    num_rows = 4 if y_baseline is not None else 3
    fig, axes = plt.subplots(num_rows, 4, figsize=(16, 4*num_rows))
    
    if num_rows == 3:
        axes = axes.reshape(3, 4)
    
    frames_to_show = [0, 3, 6, 9]
    
    # Row 1: Input frames (last 4 of 13)
    input_frames = [9, 10, 11, 12]  # Last 4 input frames
    for i, frame_idx in enumerate(input_frames):
        im = axes[0, i].imshow(X_dbz[sample_idx, :, :, frame_idx], cmap='viridis', vmin=0, vmax=70)
        axes[0, i].set_title(f'Input Frame {frame_idx+1}')
        axes[0, i].axis('off')
    
    # Row 2: Ground truth
    for i, frame_idx in enumerate(frames_to_show):
        axes[1, i].imshow(y_test_dbz[sample_idx, :, :, frame_idx], cmap='viridis', vmin=0, vmax=70)
        axes[1, i].set_title(f'True Frame {frame_idx+14}')
        axes[1, i].axis('off')
    
    # Row 3: Our prediction
    for i, frame_idx in enumerate(frames_to_show):
        axes[2, i].imshow(y_pred_dbz[sample_idx, :, :, frame_idx], cmap='viridis', vmin=0, vmax=70)
        axes[2, i].set_title(f'Our Prediction {frame_idx+14}')
        axes[2, i].axis('off')
    
    # Row 4: Baseline (if provided)
    if y_baseline is not None:
        y_baseline_dbz = y_baseline * SEVIR_SCALE + SEVIR_MEAN
        for i, frame_idx in enumerate(frames_to_show):
            axes[3, i].imshow(y_baseline_dbz[sample_idx, :, :, frame_idx], cmap='viridis', vmin=0, vmax=70)
            axes[3, i].set_title(f'Persistence {frame_idx+14}')
            axes[3, i].axis('off')
    
    # Apply tight_layout first, then add colorbar with proper spacing
    plt.tight_layout()
    
    # Create space for colorbar by adjusting subplot parameters
    plt.subplots_adjust(right=0.85)
    
    # Add colorbar
    cbar_ax = fig.add_axes((0.87, 0.1, 0.03, 0.8))  # (left, bottom, width, height)
    fig.colorbar(im, cax=cbar_ax, label='VIL (dBZ)')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RESULTS_DIR / f"real_sevir_test_{timestamp}.png"
    RESULTS_DIR.mkdir(exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Real SEVIR visualization saved to: {filename}")
    plt.close()
    
    return str(filename)

def analyze_frame_by_frame(y_test, y_pred, y_baseline=None):
    """Analyze performance frame by frame"""
    print(f"\n📈 Frame-by-frame analysis:")
    
    for t in range(12):
        mae_ours = np.mean(np.abs(y_pred[:, :, :, t] - y_test[:, :, :, t]))
        
        if y_baseline is not None:
            mae_baseline = np.mean(np.abs(y_baseline[:, :, :, t] - y_test[:, :, :, t]))
            improvement = ((mae_baseline - mae_ours) / mae_baseline) * 100
            print(f"   Frame {t+1:2d}: Our MAE={mae_ours:.4f}, Baseline={mae_baseline:.4f}, Improvement={improvement:+5.1f}%")
        else:
            print(f"   Frame {t+1:2d}: Our MAE={mae_ours:.4f}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test model on real SEVIR data')
    parser.add_argument('--model', type=str, help='Specific model directory name to test (partial name match)')
    args = parser.parse_args()
    
    print("🌩️  Testing Trained Model on Real SEVIR Data")
    print("=" * 60)
    
    # Load real SEVIR data
    X_test, y_test = load_real_sevir_data(num_samples=NUM_SAMPLES)
    if X_test is None:
        print("❌ Failed to load SEVIR data. Exiting.")
        return
    
    # Load our model (with optional specific model name)
    model = load_our_model(args.model)
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Test our model
    y_pred, mse, mae, y_pred_dbz, y_test_dbz = test_on_real_sevir(model, X_test, y_test)
    if y_pred is None:
        print("❌ Failed to make predictions. Exiting.")
        return
    
    # Compare with baseline
    y_baseline, mse_baseline, mae_baseline = compare_with_baseline(X_test, y_test)
    
    # Visualize results
    viz_file = visualize_real_sevir_results(X_test, y_test, y_pred, y_baseline)
    
    # Frame-by-frame analysis
    analyze_frame_by_frame(y_test, y_pred, y_baseline)
    
    # Final assessment
    print(f"\n🎯 REAL SEVIR TEST SUMMARY:")
    print(f"   🤖 Our Model:")
    print(f"      MAE: {mae:.4f} (normalized), {mae * SEVIR_SCALE:.1f} dBZ")
    print(f"      MSE: {mse:.4f} (normalized), {mse * SEVIR_SCALE**2:.1f} dBZ²")
    
    print(f"   📊 Persistence Baseline:")
    print(f"      MAE: {mae_baseline:.4f} (normalized), {mae_baseline * SEVIR_SCALE:.1f} dBZ")
    print(f"      MSE: {mse_baseline:.4f} (normalized), {mse_baseline * SEVIR_SCALE**2:.1f} dBZ²")
    
    improvement = ((mae_baseline - mae) / mae_baseline) * 100
    print(f"   📈 Improvement over baseline: {improvement:+.1f}%")
    
    # Performance interpretation
    if mae < 0.1:  # < ~4.8 dBZ
        print("🌟 Excellent performance on real SEVIR data!")
    elif mae < 0.2:  # < ~9.5 dBZ
        print("👍 Good performance on real SEVIR data!")
    elif mae < 0.3:  # < ~14.3 dBZ
        print("👌 Acceptable performance on real SEVIR data!")
    elif improvement > 0:
        print("📈 Model beats baseline but needs improvement!")
    else:
        print("⚠️  Model underperforms - needs more training on real data!")
    
    print(f"\n📊 Visualization saved: {viz_file}")

if __name__ == "__main__":
    main()
