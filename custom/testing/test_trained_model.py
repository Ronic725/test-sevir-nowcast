#!/usr/bin/env python3
"""
Test the trained model on synthetic data to verify it works correctly
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.append('src/')

def create_test_sequence():
    """Create a single test sequence similar to training data"""
    print("🔬 Creating test sequence...")
    
    sequence = []
    start_x, start_y = 150, 150
    radius = 30
    intensity = 0.8
    
    # Create 25-frame sequence
    for t in range(25):
        # Storm moves slightly
        curr_x = start_x + t * 2
        curr_y = start_y + t * 1.5
        
        # Create 384x384 frame
        frame = np.zeros((384, 384))
        
        # Add main storm cell
        y_coords, x_coords = np.meshgrid(np.arange(384), np.arange(384), indexing='ij')
        mask = (x_coords - curr_x)**2 + (y_coords - curr_y)**2 <= radius**2
        frame[mask] = intensity * np.exp(-((x_coords[mask] - curr_x)**2 + (y_coords[mask] - curr_y)**2) / (radius**2 / 2))
        
        # Add some noise
        frame += np.random.exponential(0.1, (384, 384)) * 0.3
        frame = np.clip(frame, 0, 1)
        
        sequence.append(frame)
    
    # Input: first 13 frames, Expected output: last 12 frames
    X = np.stack(sequence[:13], axis=-1)[np.newaxis, ...]  # Shape: (1, 384, 384, 13)
    y_true = np.stack(sequence[13:], axis=-1)[np.newaxis, ...]  # Shape: (1, 384, 384, 12)
    
    print(f"✅ Test sequence created:")
    print(f"   Input shape: {X.shape}")
    print(f"   Expected output shape: {y_true.shape}")
    
    return X, y_true

def load_model(model_path):
    """Load the trained model"""
    print(f"📁 Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {model.count_params():,}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_prediction(model, X, y_true):
    """Test model prediction"""
    print("\n🧪 Testing model prediction...")
    
    try:
        # Make prediction
        y_pred = model.predict(X, verbose=0)
        
        print(f"✅ Prediction successful!")
        print(f"   Prediction shape: {y_pred.shape}")
        print(f"   Prediction range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
        
        # Calculate metrics
        mse = np.mean((y_pred - y_true)**2)
        mae = np.mean(np.abs(y_pred - y_true))
        
        print(f"\n📊 Prediction metrics:")
        print(f"   MSE: {mse:.6f}")
        print(f"   MAE: {mae:.6f}")
        
        return y_pred, mse, mae
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None, None, None

def visualize_results(X, y_true, y_pred, save_path="test_results.png"):
    """Visualize input, ground truth, and prediction"""
    print(f"\n📊 Creating visualization...")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Show some input frames
    for i in range(4):
        frame_idx = i * 3  # Show frames 0, 3, 6, 9
        axes[0, i].imshow(X[0, :, :, frame_idx], cmap='viridis', vmin=0, vmax=1)
        axes[0, i].set_title(f'Input Frame {frame_idx+1}')
        axes[0, i].axis('off')
    
    # Show some ground truth frames
    for i in range(4):
        frame_idx = i * 3  # Show frames 0, 3, 6, 9
        axes[1, i].imshow(y_true[0, :, :, frame_idx], cmap='viridis', vmin=0, vmax=1)
        axes[1, i].set_title(f'True Frame {frame_idx+14}')
        axes[1, i].axis('off')
    
    # Show predictions
    for i in range(4):
        frame_idx = i * 3  # Show frames 0, 3, 6, 9
        axes[2, i].imshow(y_pred[0, :, :, frame_idx], cmap='viridis', vmin=0, vmax=1)
        axes[2, i].set_title(f'Predicted Frame {frame_idx+14}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")
    plt.close()

def find_latest_model():
    """Find the most recently trained model"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return None
    
    # Look for directories with trained models
    model_dirs = [d for d in os.listdir(models_dir) if d.startswith('trained_')]
    if not model_dirs:
        return None
    
    # Sort by modification time and get the latest
    model_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    latest_dir = model_dirs[0]
    
    # Look for .h5 model file
    model_path = os.path.join(models_dir, latest_dir, "model_mse.h5")
    if os.path.exists(model_path):
        return model_path
    
    return None

def main():
    print("🧪 Testing Trained Weather Nowcasting Model")
    print("=" * 60)
    
    # Find the latest trained model
    model_path = find_latest_model()
    if model_path is None:
        print("❌ No trained model found. Please run training first.")
        return
    
    print(f"🎯 Using model: {model_path}")
    
    # Load model
    model = load_model(model_path)
    if model is None:
        return
    
    # Create test data
    X, y_true = create_test_sequence()
    
    # Test prediction
    y_pred, mse, mae = test_prediction(model, X, y_true)
    if y_pred is None:
        return
    
    # Visualize results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = f"test_results_{timestamp}.png"
    visualize_results(X, y_true, y_pred, viz_path)
    
    print(f"\n🎉 Model testing completed successfully!")
    print(f"📈 The model can predict weather radar sequences")
    print(f"📊 Performance: MSE={mse:.6f}, MAE={mae:.6f}")
    print(f"🖼️  Visualization saved to: {viz_path}")
    
    # Performance interpretation
    if mse < 0.01:
        print("🌟 Excellent performance!")
    elif mse < 0.05:
        print("👍 Good performance!")
    elif mse < 0.1:
        print("👌 Acceptable performance!")
    else:
        print("⚠️  Model may need more training")

if __name__ == "__main__":
    main()
