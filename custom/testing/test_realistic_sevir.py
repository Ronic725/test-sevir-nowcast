#!/usr/bin/env python3
"""
Create realistic SEVIR-like test data and test our model
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

# SEVIR normalization constants
SEVIR_MEAN = 33.44
SEVIR_SCALE = 47.54

def create_realistic_sevir_data(num_samples=10):
    """Create more realistic SEVIR-like radar data"""
    print(f"🌩️  Creating realistic SEVIR-like test data ({num_samples} samples)...")
    
    np.random.seed(42)  # For reproducible results
    
    input_data = []
    output_data = []
    
    for sample in range(num_samples):
        # Create more complex weather patterns
        sequence = []
        
        # Multiple storm systems with different characteristics
        num_storms = np.random.randint(2, 5)
        storms = []
        
        for _ in range(num_storms):
            storms.append({
                'x': np.random.randint(50, 334),
                'y': np.random.randint(50, 334),
                'intensity': np.random.uniform(40, 70),  # dBZ range
                'radius': np.random.randint(15, 40),
                'vx': np.random.uniform(-2, 2),
                'vy': np.random.uniform(-2, 2),
                'growth_rate': np.random.uniform(-0.5, 0.3)  # Storm evolution
            })
        
        # Create 25-frame sequence (13 input + 12 output)
        for t in range(25):
            frame = np.zeros((384, 384))
            
            # Add background precipitation (light rain)
            bg_intensity = np.random.uniform(0, 15)
            if bg_intensity > 5:
                bg_mask = np.random.random((384, 384)) < 0.3
                frame[bg_mask] = np.random.uniform(5, bg_intensity, bg_mask.sum())
            
            # Add each storm system
            y_coords, x_coords = np.meshgrid(np.arange(384), np.arange(384), indexing='ij')
            
            for storm in storms:
                # Update storm position
                curr_x = storm['x'] + t * storm['vx']
                curr_y = storm['y'] + t * storm['vy']
                
                # Update storm intensity (growth/decay)
                curr_intensity = storm['intensity'] + t * storm['growth_rate']
                curr_intensity = np.clip(curr_intensity, 0, 75)
                
                # Create storm with realistic radar characteristics
                if curr_intensity > 5:  # Only if storm is significant
                    distance = np.sqrt((x_coords - curr_x)**2 + (y_coords - curr_y)**2)
                    
                    # Core with high reflectivity
                    core_mask = distance <= storm['radius'] * 0.3
                    frame[core_mask] = np.maximum(frame[core_mask], 
                                                curr_intensity * np.random.uniform(0.9, 1.0))
                    
                    # Mid-level precipitation
                    mid_mask = (distance > storm['radius'] * 0.3) & (distance <= storm['radius'] * 0.7)
                    frame[mid_mask] = np.maximum(frame[mid_mask], 
                                               curr_intensity * np.random.uniform(0.5, 0.8))
                    
                    # Outer edge
                    outer_mask = (distance > storm['radius'] * 0.7) & (distance <= storm['radius'])
                    frame[outer_mask] = np.maximum(frame[outer_mask], 
                                                 curr_intensity * np.random.uniform(0.2, 0.5))
            
            # Add realistic noise and artifacts
            frame += np.random.gamma(1, 2, (384, 384)) * 0.5  # Ground clutter
            frame = np.clip(frame, 0, 75)  # Realistic dBZ range
            
            # Apply beam blockage and range effects (realistic radar limitations)
            center_x, center_y = 192, 192  # Radar center
            distance_from_radar = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            range_effect = np.clip(1.0 - distance_from_radar / 400, 0.7, 1.0)
            frame *= range_effect
            
            sequence.append(frame)
        
        # Convert to SEVIR format (normalized)
        input_seq = np.stack(sequence[:13], axis=-1)
        output_seq = np.stack(sequence[13:], axis=-1)
        
        input_data.append(input_seq)
        output_data.append(output_seq)
    
    input_data = np.array(input_data)
    output_data = np.array(output_data)
    
    print(f"✅ Created realistic SEVIR-like data:")
    print(f"   Input shape: {input_data.shape}")
    print(f"   Output shape: {output_data.shape}")
    print(f"   Input range: [{input_data.min():.1f}, {input_data.max():.1f}] dBZ")
    print(f"   Output range: [{output_data.min():.1f}, {output_data.max():.1f}] dBZ")
    
    return input_data, output_data

def save_as_sevir_format(input_data, output_data, filename="realistic_sevir_test.h5"):
    """Save data in SEVIR H5 format"""
    print(f"💾 Saving as SEVIR format: {filename}")
    
    with h5py.File(filename, 'w') as f:
        f.create_dataset('IN', data=input_data, compression='gzip')
        f.create_dataset('OUT', data=output_data, compression='gzip')
    
    print(f"✅ Saved to {filename}")
    return filename

def test_model_on_realistic_data():
    """Test our model on realistic SEVIR-like data"""
    print("🌩️  Testing Model on Realistic SEVIR-like Data")
    print("=" * 60)
    
    # Create realistic test data
    input_data, output_data = create_realistic_sevir_data(num_samples=5)
    
    # Save in SEVIR format
    test_file = save_as_sevir_format(input_data, output_data, "data/sample/realistic_sevir_test.h5")
    
    # Load our model
    models_dir = "../../models"
    model_dirs = [d for d in os.listdir(models_dir) if d.startswith('trained_')]
    if not model_dirs:
        print("❌ No trained model found")
        return
    
    model_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    model_path = os.path.join(models_dir, model_dirs[0], "model_mse.h5")
    
    print(f"🤖 Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Normalize data for model (same as training)
    X_test = (input_data - SEVIR_MEAN) / SEVIR_SCALE
    y_test = (output_data - SEVIR_MEAN) / SEVIR_SCALE
    
    print(f"\n📊 Normalized data ranges:")
    print(f"   Input: [{X_test.min():.3f}, {X_test.max():.3f}]")
    print(f"   Output: [{y_test.min():.3f}, {y_test.max():.3f}]")
    
    # Make predictions
    print(f"\n🧪 Making predictions...")
    y_pred = model.predict(X_test, batch_size=2, verbose=1)
    
    # Calculate metrics
    mse = np.mean((y_pred - y_test)**2)
    mae = np.mean(np.abs(y_pred - y_test))
    
    # Convert back to dBZ
    y_pred_dbz = y_pred * SEVIR_SCALE + SEVIR_MEAN
    y_test_dbz = y_test * SEVIR_SCALE + SEVIR_MEAN
    
    mse_dbz = np.mean((y_pred_dbz - output_data)**2)
    mae_dbz = np.mean(np.abs(y_pred_dbz - output_data))
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Normalized scale:")
    print(f"      MSE: {mse:.6f}")
    print(f"      MAE: {mae:.6f}")
    print(f"   dBZ scale:")
    print(f"      MSE: {mse_dbz:.3f}")
    print(f"      MAE: {mae_dbz:.3f}")
    
    # Baseline comparison (persistence)
    last_frame = X_test[:, :, :, -1:]
    y_baseline = np.repeat(last_frame, 12, axis=-1)
    
    mse_baseline = np.mean((y_baseline - y_test)**2)
    mae_baseline = np.mean(np.abs(y_baseline - y_test))
    
    print(f"   Persistence Baseline:")
    print(f"      MSE: {mse_baseline:.6f}")
    print(f"      MAE: {mae_baseline:.6f}")
    
    improvement = ((mae_baseline - mae) / mae_baseline) * 100
    print(f"   📈 Improvement: {improvement:+.1f}%")
    
    # Visualize results
    visualize_realistic_results(input_data, output_data, y_pred_dbz, sample_idx=0)
    
    # Frame-by-frame analysis
    print(f"\n📈 Frame-by-frame performance:")
    for t in range(0, 12, 2):
        frame_mae = np.mean(np.abs(y_pred[:, :, :, t] - y_test[:, :, :, t]))
        frame_mae_dbz = frame_mae * SEVIR_SCALE
        print(f"   Frame {t+1:2d}: MAE = {frame_mae:.4f} ({frame_mae_dbz:.1f} dBZ)")
    
    # Assessment
    print(f"\n🎯 ASSESSMENT:")
    if mae < 0.15:  # < ~7 dBZ
        print("🌟 Excellent performance on realistic weather data!")
    elif mae < 0.25:  # < ~12 dBZ
        print("👍 Good performance on realistic weather data!")
    elif mae < 0.35:  # < ~17 dBZ
        print("👌 Acceptable performance - shows promise!")
    elif improvement > 0:
        print("📈 Beats baseline but needs more training!")
    else:
        print("⚠️  Model struggles with complex weather patterns")
    
    return model, X_test, y_test, y_pred

def visualize_realistic_results(input_data, output_data, predictions, sample_idx=0):
    """Visualize results on realistic data"""
    print(f"\n📊 Creating visualization...")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Show some input frames
    input_frames = [9, 10, 11, 12]  # Last 4 input frames
    for i, frame_idx in enumerate(input_frames):
        axes[0, i].imshow(input_data[sample_idx, :, :, frame_idx], cmap='viridis', vmin=0, vmax=70)
        axes[0, i].set_title(f'Input Frame {frame_idx+1}')
        axes[0, i].axis('off')
    
    # Show ground truth
    frames_to_show = [0, 3, 6, 9]
    for i, frame_idx in enumerate(frames_to_show):
        axes[1, i].imshow(output_data[sample_idx, :, :, frame_idx], cmap='viridis', vmin=0, vmax=70)
        axes[1, i].set_title(f'True Frame {frame_idx+14}')
        axes[1, i].axis('off')
    
    # Show predictions
    for i, frame_idx in enumerate(frames_to_show):
        axes[2, i].imshow(predictions[sample_idx, :, :, frame_idx], cmap='viridis', vmin=0, vmax=70)
        axes[2, i].set_title(f'Predicted Frame {frame_idx+14}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"realistic_sevir_test_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {filename}")
    plt.close()

if __name__ == "__main__":
    test_model_on_realistic_data()
