#!/usr/bin/env python3
"""
Quick analysis of model performance and data scaling
"""

import numpy as np
import tensorflow as tf
import h5py

def analyze_model_performance():
    print("🔍 Analyzing Model Performance on SEVIR Data")
    print("=" * 50)
    
    # Load our test data
    with h5py.File('test_sevir_data.h5', 'r') as f:
        input_data = f['IN'][:]
        output_data = f['OUT'][:]
    
    print(f"📊 Test Data Analysis:")
    print(f"   Input range: [{input_data.min():.1f}, {input_data.max():.1f}] dBZ")
    print(f"   Output range: [{output_data.min():.1f}, {output_data.max():.1f}] dBZ")
    
    # Load our trained model
    model_path = "models/trained_mse_20250918_120133/model_mse.h5"
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Test with different normalizations
    SEVIR_MEAN = 33.44
    SEVIR_SCALE = 47.54
    
    # Normalize as in training (0-1 range)
    X_normalized = (input_data - SEVIR_MEAN) / SEVIR_SCALE
    y_normalized = (output_data - SEVIR_MEAN) / SEVIR_SCALE
    
    print(f"\n📊 Normalized Data Analysis:")
    print(f"   Input range: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
    print(f"   Output range: [{y_normalized.min():.3f}, {y_normalized.max():.3f}]")
    
    # Make prediction
    y_pred_norm = model.predict(X_normalized, verbose=0)
    
    # Convert back to dBZ
    y_pred_dbz = y_pred_norm * SEVIR_SCALE + SEVIR_MEAN
    
    print(f"\n📊 Prediction Analysis:")
    print(f"   Prediction (normalized): [{y_pred_norm.min():.3f}, {y_pred_norm.max():.3f}]")
    print(f"   Prediction (dBZ): [{y_pred_dbz.min():.1f}, {y_pred_dbz.max():.1f}]")
    
    # Calculate various metrics
    mse_norm = np.mean((y_pred_norm - y_normalized)**2)
    mae_norm = np.mean(np.abs(y_pred_norm - y_normalized))
    
    mse_dbz = np.mean((y_pred_dbz - output_data)**2)
    mae_dbz = np.mean(np.abs(y_pred_dbz - output_data))
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Normalized scale:")
    print(f"      MSE: {mse_norm:.6f}")
    print(f"      MAE: {mae_norm:.6f}")
    print(f"   dBZ scale:")
    print(f"      MSE: {mse_dbz:.1f}")
    print(f"      MAE: {mae_dbz:.1f}")
    
    # Check if model was trained on similar data distribution
    training_range_min = -SEVIR_MEAN / SEVIR_SCALE  # ~-0.7
    training_range_max = (70 - SEVIR_MEAN) / SEVIR_SCALE  # ~0.77
    
    print(f"\n🎯 Training vs Test Data Compatibility:")
    print(f"   Expected training range: [{training_range_min:.3f}, {training_range_max:.3f}]")
    print(f"   Test data range: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
    
    if X_normalized.min() >= training_range_min and X_normalized.max() <= training_range_max:
        print("   ✅ Test data is within expected range")
    else:
        print("   ⚠️  Test data may be outside training distribution")
    
    # Performance interpretation
    if mae_norm < 0.1:
        print(f"\n🌟 Model shows good performance on normalized scale!")
        print(f"💡 High dBZ errors are due to scale amplification (×{SEVIR_SCALE:.1f})")
    else:
        print(f"\n⚠️  Model may need more training")
    
    # Show frame-by-frame analysis for first sample
    print(f"\n📊 Frame-by-frame analysis (Sample 0):")
    for t in range(0, 12, 3):  # Every 3rd frame
        frame_mae = np.mean(np.abs(y_pred_dbz[0, :, :, t] - output_data[0, :, :, t]))
        print(f"   Frame {t+1}: MAE = {frame_mae:.1f} dBZ")

if __name__ == "__main__":
    analyze_model_performance()
