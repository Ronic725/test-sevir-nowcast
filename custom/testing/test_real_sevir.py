#!/usr/bin/env python3
"""
Test our trained model on real SEVIR nowcast data
"""

import os
import sys
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.append('../../')
sys.path.append('../../src/')
from readers.nowcast_reader import read_data

# SEVIR normalization constants
SEVIR_MEAN = 33.44
SEVIR_SCALE = 47.54

def load_real_sevir_data(num_samples=5):
    """Load real SEVIR test data"""
    print("📡 Loading real SEVIR nowcast test data...")
    
    data_file = "data/sample/nowcast_testing.h5"
    
    if not os.path.exists(data_file):
        print(f"❌ SEVIR test data not found: {data_file}")
        print("💡 Please download from: https://www.dropbox.com/s/27pqogywg75as5f/nowcast_testing.h5")
        return None, None
    
    try:
        # Use the official SEVIR reader which handles normalization
        X_test, y_test = read_data(data_file, end=num_samples)
        
        print(f"✅ Loaded real SEVIR data:")
        print(f"   Input shape: {X_test.shape}")
        print(f"   Output shape: {y_test.shape}")
        print(f"   Input range: [{X_test.min():.3f}, {X_test.max():.3f}] (normalized)")
        print(f"   Output range: [{y_test.min():.3f}, {y_test.max():.3f}] (normalized)")
        
        return X_test, y_test
        
    except Exception as e:
        print(f"❌ Error loading SEVIR data: {e}")
        return None, None

def load_our_model():
    """Load our trained model"""
    models_dir = "../../models"
    model_dirs = [d for d in os.listdir(models_dir) if d.startswith('trained_')]
    
    if not model_dirs:
        print("❌ No trained model found")
        return None
    
    # Get latest trained model
    model_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    model_path = os.path.join(models_dir, model_dirs[0], "model_mse.h5")
    
    print(f"🤖 Loading our trained model: {model_path}")
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
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
        axes[0, i].imshow(X_dbz[sample_idx, :, :, frame_idx], cmap='viridis', vmin=0, vmax=70)
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
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"real_sevir_test_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Real SEVIR visualization saved to: {filename}")
    plt.close()
    
    return filename

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
    print("🌩️  Testing Trained Model on Real SEVIR Data")
    print("=" * 60)
    
    # Load real SEVIR data
    X_test, y_test = load_real_sevir_data(num_samples=10)
    if X_test is None:
        return
    
    # Load our model
    model = load_our_model()
    if model is None:
        return
    
    # Test our model
    y_pred, mse, mae, y_pred_dbz, y_test_dbz = test_on_real_sevir(model, X_test, y_test)
    if y_pred is None:
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
