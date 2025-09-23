#!/usr/bin/env python3
"""
Limited Data Training Setup for SEVIR on M1 MacBook Pro
Supports: synthetic data, small datasets, and efficient batch processing
"""
import os
import sys

# Add parent directories to path to access original repo modules
sys.path.append('../../')
sys.path.append('../../src/')

import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import argparse
import logging
from datetime import datetime

# Import SEVIR modules
from models.nowcast_unet import create_model
from losses.style_loss import vggloss_scaled
from readers.nowcast_reader import read_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LimitedDataTrainer:
    def __init__(self, data_type='synthetic', num_samples=1024, validation_split=0.2):
        """
        Initialize trainer for limited data scenarios
        
        Args:
            data_type: 'synthetic' or 'real' (if you have sample data)
            num_samples: Number of training samples to generate/use
            validation_split: Fraction of data for validation
        """
        self.data_type = data_type
        self.num_samples = num_samples
        self.validation_split = validation_split
        self.model = None
        
    def create_synthetic_data(self):
        """Create synthetic radar-like data for training"""
        logger.info(f"Creating {self.num_samples} synthetic radar sequences...")
        
        # Create synthetic radar patterns (not just random noise)
        X_data = []
        y_data = []
        
        for i in range(self.num_samples):
            # Generate a moving weather pattern
            sequence_length = 13 + 12  # 13 input + 12 output frames
            
            # Create a moving circular pattern (simulating storm cell)
            center_x = np.random.randint(50, 334)  # Stay within bounds
            center_y = np.random.randint(50, 334)
            radius = np.random.randint(20, 60)
            intensity = np.random.uniform(0.3, 1.0)
            
            # Movement velocity
            dx = np.random.uniform(-3, 3)
            dy = np.random.uniform(-3, 3)
            
            sequence = []
            for t in range(sequence_length):
                # Current position of storm center
                curr_x = center_x + dx * t
                curr_y = center_y + dy * t
                
                # Create 384x384 frame
                frame = np.zeros((384, 384))
                
                # Add main storm cell
                y_coords, x_coords = np.meshgrid(np.arange(384), np.arange(384), indexing='ij')
                mask = (x_coords - curr_x)**2 + (y_coords - curr_y)**2 <= radius**2
                frame[mask] = intensity * np.exp(-((x_coords[mask] - curr_x)**2 + (y_coords[mask] - curr_y)**2) / (radius**2 / 2))
                
                # Add some noise and smaller patterns
                frame += np.random.exponential(0.1, (384, 384)) * 0.3
                frame = np.clip(frame, 0, 1)
                
                sequence.append(frame)
            
            # Split into input (first 13) and output (last 12)
            X_data.append(np.stack(sequence[:13], axis=-1))
            y_data.append(np.stack(sequence[13:], axis=-1))
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{self.num_samples} sequences")
        
        X = np.array(X_data, dtype=np.float32)
        y = np.array(y_data, dtype=np.float32)
        
        logger.info(f"✓ Created synthetic dataset:")
        logger.info(f"  Input shape: {X.shape}")
        logger.info(f"  Output shape: {y.shape}")
        logger.info(f"  Data range: [{X.min():.3f}, {X.max():.3f}]")
        
        return X, y
    
    def load_or_create_data(self):
        """Load existing data or create synthetic data"""
        if self.data_type == 'synthetic':
            return self.create_synthetic_data()
        
        # Try to load real data if available
        sample_data_paths = [
            'data/sample/nowcast_testing.h5',
            'data/interim/nowcast_training.h5',
            'data/interim/nowcast_testing.h5'
        ]
        
        for path in sample_data_paths:
            if os.path.exists(path):
                logger.info(f"Loading data from {path}")
                try:
                    with h5py.File(path, 'r') as f:
                        # Limit to requested number of samples
                        max_samples = min(self.num_samples, len(f['IN']))
                        X = f['IN'][:max_samples]
                        y = f['OUT'][:max_samples]
                        
                        logger.info(f"✓ Loaded {max_samples} samples from {path}")
                        return X, y
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    continue
        
        # Fallback to synthetic data
        logger.info("No real data found, creating synthetic data...")
        return self.create_synthetic_data()
    
    def create_model_with_loss(self, loss_type='mse'):
        """Create and compile model with specified loss"""
        logger.info(f"Creating model with {loss_type} loss...")
        
        inputs, outputs = create_model(input_shape=(384, 384, 13))
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Choose loss function
        if loss_type == 'mse':
            loss_fn = 'mse'
            batch_size = 8  # Can use larger batch for MSE
        elif loss_type == 'mae':
            loss_fn = 'mae'
            batch_size = 8
        elif loss_type == 'vgg':
            loss_fn = vggloss_scaled
            batch_size = 2  # Smaller batch for VGG loss
        else:
            loss_fn = 'mse'
            batch_size = 8
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss_fn,
            metrics=['mae']
        )
        
        logger.info(f"✓ Model compiled with {loss_type} loss")
        logger.info(f"  Parameters: {model.count_params():,}")
        logger.info(f"  Recommended batch size: {batch_size}")
        
        self.model = model
        return model, batch_size
    
    def train_with_limited_data(self, loss_type='mse', epochs=10, save_model=True):
        """Train model with limited data and efficient batching"""
        logger.info("=" * 60)
        logger.info("🚀 Starting Limited Data Training")
        logger.info("=" * 60)
        
        # Load data
        X, y = self.load_or_create_data()
        
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=42
        )
        
        logger.info(f"Data split:")
        logger.info(f"  Training: {X_train.shape[0]} samples")
        logger.info(f"  Validation: {X_val.shape[0]} samples")
        
        # Create model
        model, batch_size = self.create_model_with_loss(loss_type)
        
        # Setup callbacks
        callbacks = self._setup_callbacks(loss_type)
        
        # Train model
        logger.info(f"\n🔥 Starting training...")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Loss function: {loss_type}")
        
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model if requested
        if save_model:
            self._save_model_and_history(model, history, loss_type)
        
        # Test prediction
        self._test_prediction(model, X_val[:1])
        
        logger.info("✅ Training completed successfully!")
        return model, history
    
    def _setup_callbacks(self, loss_type):
        """Setup training callbacks"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/limited_training_{loss_type}_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(log_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1
            )
        ]
        
        logger.info(f"✓ Callbacks configured, logs will be saved to: {log_dir}")
        return callbacks
    
    def _save_model_and_history(self, model, history, loss_type):
        """Save trained model and training history"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"models/trained_{loss_type}_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, f'model_{loss_type}.h5')
        model.save(model_path)
        
        # Save history
        import pandas as pd
        history_df = pd.DataFrame(history.history)
        history_path = os.path.join(save_dir, 'training_history.csv')
        history_df.to_csv(history_path, index=False)
        
        logger.info(f"✓ Model saved to: {model_path}")
        logger.info(f"✓ History saved to: {history_path}")
    
    def _test_prediction(self, model, test_input):
        """Test model prediction"""
        logger.info("\n🔍 Testing prediction...")
        
        prediction = model.predict(test_input, verbose=0)
        
        logger.info(f"✓ Prediction successful")
        logger.info(f"  Input shape: {test_input.shape}")
        logger.info(f"  Output shape: {prediction.shape}")
        logger.info(f"  Output range: [{prediction.min():.3f}, {prediction.max():.3f}]")

def main():
    parser = argparse.ArgumentParser(description='Limited Data SEVIR Training')
    parser.add_argument('--data_type', type=str, default='synthetic', 
                       choices=['synthetic', 'real'], help='Type of data to use')
    parser.add_argument('--num_samples', type=int, default=1024, 
                       help='Number of training samples')
    parser.add_argument('--loss_type', type=str, default='mse', 
                       choices=['mse', 'mae', 'vgg'], help='Loss function to use')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='Number of training epochs')
    parser.add_argument('--validation_split', type=float, default=0.2, 
                       help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = LimitedDataTrainer(
        data_type=args.data_type,
        num_samples=args.num_samples,
        validation_split=args.validation_split
    )
    
    # Start training
    model, history = trainer.train_with_limited_data(
        loss_type=args.loss_type,
        epochs=args.epochs
    )
    
    print("\n🎉 Training completed! Check the logs directory for saved models and TensorBoard logs.")
    print("To visualize training: tensorboard --logdir logs/")

if __name__ == "__main__":
    main()
