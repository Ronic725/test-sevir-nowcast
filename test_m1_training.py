#!/usr/bin/env python3
"""
Test script for M1 MacBook Pro - Train a simple SEVIR model
"""
import os
import sys
sys.path.append('src/')

import numpy as np
import tensorflow as tf
from models.nowcast_unet import create_model

def create_dummy_data(num_samples=100):
    """Create dummy data for testing"""
    print("Creating dummy training data...")
    
    # Input: 13 radar frames (384x384x13)
    X = np.random.randn(num_samples, 384, 384, 13).astype(np.float32)
    
    # Output: 12 future frames (384x384x12) 
    y = np.random.randn(num_samples, 384, 384, 12).astype(np.float32)
    
    print(f"✓ Created {num_samples} training samples")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {y.shape}")
    
    return X, y

def test_training():
    """Test model training on M1"""
    print("🔧 Testing SEVIR model training on M1 MacBook Pro")
    print("=" * 50)
    
    # Create model
    print("\n1. Creating model...")
    inputs, outputs = create_model(input_shape=(384,384,13))
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile with simple MSE loss (memory efficient)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print(f"✓ Model created and compiled")
    print(f"  Total parameters: {model.count_params():,}")
    
    # Create dummy data
    X_train, y_train = create_dummy_data(num_samples=32)  # Small batch for testing
    
    # Test training step
    print("\n2. Testing training...")
    print("   Batch size: 2 (M1 optimized)")
    print("   Epochs: 2 (quick test)")
    
    # Train for a few epochs
    history = model.fit(
        X_train, y_train,
        batch_size=2,  # Small batch size for M1
        epochs=2,
        validation_split=0.2,
        verbose=1
    )
    
    print("\n✅ Training test completed successfully!")
    print(f"   Final loss: {history.history['loss'][-1]:.4f}")
    print(f"   Final MAE: {history.history['mae'][-1]:.4f}")
    
    # Test prediction
    print("\n3. Testing prediction...")
    test_input = X_train[:1]  # Single sample
    prediction = model.predict(test_input, verbose=0)
    
    print(f"✓ Prediction successful")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {prediction.shape}")
    
    print("\n🎉 M1 MacBook Pro SEVIR Test: PASSED!")
    print("Your system is ready for SEVIR training and inference.")

if __name__ == "__main__":
    test_training()
