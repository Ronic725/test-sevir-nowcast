#!/usr/bin/env python3
"""
Test our trained model against SEVIR-compatible data and compare with pretrained models
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

# SEVIR normalization constants (from original paper)
SEVIR_MEAN = 33.44
SEVIR_SCALE = 47.54

def create_sevir_compatible_data(num_samples=10):
    """Create synthetic data in SEVIR format"""
    print(f"🌩️  Creating {num_samples} SEVIR-compatible test samples...")
    
    # SEVIR format: (samples, height, width, time_steps)
    # Input: 13 frames, Output: 12 frames
    input_data = []
    output_data = []
    
    for sample in range(num_samples):
        sequence = []
        
        # Random storm parameters for each sample
        start_x = np.random.randint(50, 334)  # Keep storm in frame
        start_y = np.random.randint(50, 334)
        radius = np.random.randint(20, 40)
        intensity = np.random.uniform(0.6, 1.0)
        velocity_x = np.random.uniform(-3, 3)
        velocity_y = np.random.uniform(-3, 3)
        
        # Create 25-frame sequence
        for t in range(25):
            curr_x = start_x + t * velocity_x
            curr_y = start_y + t * velocity_y
            
            # Create 384x384 frame
            frame = np.zeros((384, 384))
            
            # Add main storm cell
            y_coords, x_coords = np.meshgrid(np.arange(384), np.arange(384), indexing='ij')
            mask = (x_coords - curr_x)**2 + (y_coords - curr_y)**2 <= radius**2
            frame[mask] = intensity * np.exp(-((x_coords[mask] - curr_x)**2 + (y_coords[mask] - curr_y)**2) / (radius**2 / 2))
            
            # Add smaller storm cells
            for _ in range(np.random.randint(1, 4)):
                small_x = curr_x + np.random.normal(0, 30)
                small_y = curr_y + np.random.normal(0, 30)
                small_radius = np.random.randint(10, 20)
                small_intensity = np.random.uniform(0.3, 0.7)
                
                small_mask = (x_coords - small_x)**2 + (y_coords - small_y)**2 <= small_radius**2
                frame[small_mask] += small_intensity * np.exp(-((x_coords[small_mask] - small_x)**2 + (y_coords[small_mask] - small_y)**2) / (small_radius**2 / 2))
            
            # Add noise
            frame += np.random.exponential(0.05, (384, 384)) * 0.2
            frame = np.clip(frame, 0, 1)
            
            sequence.append(frame)
        
        # Convert to SEVIR format and normalize
        input_seq = np.stack(sequence[:13], axis=-1)  # (384, 384, 13)
        output_seq = np.stack(sequence[13:], axis=-1)  # (384, 384, 12)
        
        # Convert to SEVIR reflectivity scale (dBZ)
        input_seq = input_seq * 70.0  # Scale to 0-70 dBZ range
        output_seq = output_seq * 70.0
        
        input_data.append(input_seq)
        output_data.append(output_seq)
    
    # Convert to numpy arrays
    input_data = np.array(input_data)  # (samples, 384, 384, 13)
    output_data = np.array(output_data)  # (samples, 384, 384, 12)
    
    print(f"✅ Created SEVIR-compatible test data:")
    print(f"   Input shape: {input_data.shape}")
    print(f"   Output shape: {output_data.shape}")
    print(f"   Input range: [{input_data.min():.1f}, {input_data.max():.1f}] dBZ")
    print(f"   Output range: [{output_data.min():.1f}, {output_data.max():.1f}] dBZ")
    
    return input_data, output_data

def save_sevir_test_file(input_data, output_data, filename="test_sevir_data.h5"):
    """Save data in SEVIR HDF5 format"""
    print(f"💾 Saving test data to {filename}...")
    
    with h5py.File(filename, 'w') as f:
        f.create_dataset('IN', data=input_data, compression='gzip')
        f.create_dataset('OUT', data=output_data, compression='gzip')
    
    print(f"✅ Test data saved to {filename}")
    return filename

def load_and_test_model(model_path, input_data, output_data):
    """Load model and test on SEVIR data"""
    print(f"\n🧪 Testing model: {os.path.basename(model_path)}")
    
    try:
        # Load model
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Normalize input data for model (same as training)
        X_test = (input_data - SEVIR_MEAN) / SEVIR_SCALE
        y_true = (output_data - SEVIR_MEAN) / SEVIR_SCALE
        
        # Make predictions
        y_pred = model.predict(X_test, batch_size=4, verbose=0)
        
        # Scale predictions back to dBZ
        y_pred_dbz = y_pred * SEVIR_SCALE + SEVIR_MEAN
        
        # Calculate metrics
        mse = np.mean((y_pred_dbz - output_data)**2)
        mae = np.mean(np.abs(y_pred_dbz - output_data))
        
        # Calculate metrics on normalized data too
        mse_norm = np.mean((y_pred - y_true)**2)
        mae_norm = np.mean(np.abs(y_pred - y_true))
        
        print(f"   ✅ Prediction successful")
        print(f"   📊 Metrics (dBZ scale):")
        print(f"      MSE: {mse:.3f}")
        print(f"      MAE: {mae:.3f}")
        print(f"   📊 Metrics (normalized):")
        print(f"      MSE: {mse_norm:.6f}")
        print(f"      MAE: {mae_norm:.6f}")
        
        return y_pred_dbz, mse, mae
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None, None, None

def download_pretrained_model(model_name="mse_model.h5"):
    """Download a pretrained model for comparison"""
    model_path = f"models/nowcast/{model_name}"
    
    if os.path.exists(model_path):
        print(f"✅ Pretrained model already exists: {model_path}")
        return model_path
    
    print(f"📥 Downloading pretrained model: {model_name}")
    
    # Download using the existing download script
    try:
        os.chdir("models")
        import subprocess
        result = subprocess.run(["python", "download_models.py"], capture_output=True, text=True)
        os.chdir("..")
        
        if result.returncode == 0 and os.path.exists(model_path):
            print(f"✅ Downloaded: {model_path}")
            return model_path
        else:
            print(f"⚠️  Could not download pretrained model")
            return None
    except Exception as e:
        os.chdir("..")
        print(f"⚠️  Download failed: {e}")
        return None

def visualize_comparison(input_data, output_data, our_pred, pretrained_pred=None, sample_idx=0):
    """Visualize predictions from different models"""
    print(f"\n📊 Creating comparison visualization...")
    
    num_models = 3 if pretrained_pred is not None else 2
    fig, axes = plt.subplots(num_models, 4, figsize=(16, 4*num_models))
    
    if num_models == 2:
        axes = axes.reshape(2, 4)  # Ensure 2D array
    
    # Show some frames for comparison
    frames_to_show = [0, 3, 6, 9]  # Show every 3rd frame
    
    # Row 1: Ground truth
    for i, frame_idx in enumerate(frames_to_show):
        axes[0, i].imshow(output_data[sample_idx, :, :, frame_idx], cmap='viridis', vmin=0, vmax=50)
        axes[0, i].set_title(f'Ground Truth Frame {frame_idx+14}')
        axes[0, i].axis('off')
    
    # Row 2: Our model prediction
    for i, frame_idx in enumerate(frames_to_show):
        axes[1, i].imshow(our_pred[sample_idx, :, :, frame_idx], cmap='viridis', vmin=0, vmax=50)
        axes[1, i].set_title(f'Our Model Frame {frame_idx+14}')
        axes[1, i].axis('off')
    
    # Row 3: Pretrained model (if available)
    if pretrained_pred is not None:
        for i, frame_idx in enumerate(frames_to_show):
            axes[2, i].imshow(pretrained_pred[sample_idx, :, :, frame_idx], cmap='viridis', vmin=0, vmax=50)
            axes[2, i].set_title(f'Pretrained Model Frame {frame_idx+14}')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sevir_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Comparison saved to: {filename}")
    plt.close()

def main():
    print("🌩️  SEVIR-Compatible Model Testing")
    print("=" * 60)
    
    # Create test data
    input_data, output_data = create_sevir_compatible_data(num_samples=5)
    
    # Save test data file
    test_file = save_sevir_test_file(input_data, output_data)
    
    # Find our trained model
    models_dir = "../../models"
    model_dirs = [d for d in os.listdir(models_dir) if d.startswith('trained_')]
    if not model_dirs:
        print("❌ No trained model found. Please run training first.")
        return
    
    # Get latest trained model
    model_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    our_model_path = os.path.join(models_dir, model_dirs[0], "model_mse.h5")
    
    # Test our model
    our_pred, our_mse, our_mae = load_and_test_model(our_model_path, input_data, output_data)
    
    if our_pred is None:
        print("❌ Our model test failed")
        return
    
    # Try to download and test pretrained model
    pretrained_path = download_pretrained_model()
    pretrained_pred = None
    
    if pretrained_path:
        pretrained_pred, pre_mse, pre_mae = load_and_test_model(pretrained_path, input_data, output_data)
    
    # Create visualization
    visualize_comparison(input_data, output_data, our_pred, pretrained_pred)
    
    # Summary
    print(f"\n🎯 SUMMARY:")
    print(f"   📈 Our Model Performance:")
    print(f"      MSE: {our_mse:.3f} dBZ²")
    print(f"      MAE: {our_mae:.3f} dBZ")
    
    if pretrained_pred is not None:
        print(f"   📈 Pretrained Model Performance:")
        print(f"      MSE: {pre_mse:.3f} dBZ²") 
        print(f"      MAE: {pre_mae:.3f} dBZ")
        
        if our_mse < pre_mse:
            print(f"   🏆 Our model performs BETTER than pretrained!")
        else:
            print(f"   📚 Pretrained model has lower error (more training data)")
    
    print(f"\n✅ SEVIR-compatible testing completed!")
    print(f"🗂️  Test data saved: {test_file}")
    print(f"📊 Results visualization created")
    
    # Performance interpretation
    if our_mae < 5.0:
        print("🌟 Excellent nowcasting performance!")
    elif our_mae < 10.0:
        print("👍 Good nowcasting performance!")
    elif our_mae < 15.0:
        print("👌 Acceptable nowcasting performance!")
    else:
        print("⚠️  Model needs more training for better accuracy")

if __name__ == "__main__":
    main()
