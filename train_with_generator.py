#!/usr/bin/env python3
"""
Memory-Efficient Data Generator for Large Datasets
Handles data that doesn't fit in memory by generating batches on-the-fly
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import h5py
import os

class EfficientSEVIRGenerator(Sequence):
    """
    Memory-efficient data generator for SEVIR training
    Generates batches on-the-fly to handle large datasets
    """
    
    def __init__(self, data_source='synthetic', num_samples=10000, batch_size=4, 
                 input_frames=13, output_frames=12, shuffle=True):
        """
        Initialize the data generator
        
        Args:
            data_source: 'synthetic' or path to H5 file
            num_samples: Total number of samples to generate
            batch_size: Batch size for training
            input_frames: Number of input frames (default 13)
            output_frames: Number of output frames (default 12)
            shuffle: Whether to shuffle data between epochs
        """
        self.data_source = data_source
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.shuffle = shuffle
        
        self.indices = np.arange(num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Generate batch
        X_batch = []
        y_batch = []
        
        for idx in batch_indices:
            X_sample, y_sample = self._generate_sample(idx)
            X_batch.append(X_sample)
            y_batch.append(y_sample)
        
        return np.array(X_batch), np.array(y_batch)
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _generate_sample(self, idx):
        """Generate a single sample"""
        if self.data_source == 'synthetic':
            return self._generate_synthetic_sample(idx)
        else:
            return self._load_real_sample(idx)
    
    def _generate_synthetic_sample(self, idx):
        """Generate synthetic radar sequence"""
        # Set random seed based on index for reproducibility
        np.random.seed(idx % 10000)
        
        # Generate realistic weather pattern
        sequence_length = self.input_frames + self.output_frames
        
        # Storm parameters
        center_x = np.random.randint(50, 334)
        center_y = np.random.randint(50, 334)
        radius = np.random.randint(15, 50)
        intensity = np.random.uniform(0.4, 1.0)
        
        # Movement
        dx = np.random.uniform(-2, 2)
        dy = np.random.uniform(-2, 2)
        
        # Evolution (storm growing/shrinking)
        radius_change = np.random.uniform(-0.5, 0.5)
        intensity_change = np.random.uniform(-0.02, 0.02)
        
        sequence = []
        for t in range(sequence_length):
            # Current position and size
            curr_x = center_x + dx * t
            curr_y = center_y + dy * t
            curr_radius = max(10, radius + radius_change * t)
            curr_intensity = max(0.1, min(1.0, intensity + intensity_change * t))
            
            # Create frame
            frame = np.zeros((384, 384), dtype=np.float32)
            
            # Add main storm
            y_coords, x_coords = np.ogrid[:384, :384]
            distance = np.sqrt((x_coords - curr_x)**2 + (y_coords - curr_y)**2)
            mask = distance <= curr_radius
            
            if np.any(mask):
                frame[mask] = curr_intensity * np.exp(-(distance[mask]**2) / (curr_radius**2 / 3))
            
            # Add smaller cells and noise
            num_small_cells = np.random.randint(0, 3)
            for _ in range(num_small_cells):
                small_x = np.random.randint(0, 384)
                small_y = np.random.randint(0, 384)
                small_radius = np.random.randint(5, 20)
                small_intensity = np.random.uniform(0.1, 0.5)
                
                small_distance = np.sqrt((x_coords - small_x)**2 + (y_coords - small_y)**2)
                small_mask = small_distance <= small_radius
                if np.any(small_mask):
                    frame[small_mask] += small_intensity * np.exp(-(small_distance[small_mask]**2) / (small_radius**2))
            
            # Add background noise
            frame += np.random.exponential(0.05, (384, 384))
            frame = np.clip(frame, 0, 1)
            
            sequence.append(frame)
        
        # Split into input and output
        X_sample = np.stack(sequence[:self.input_frames], axis=-1)
        y_sample = np.stack(sequence[self.input_frames:], axis=-1)
        
        return X_sample, y_sample
    
    def _load_real_sample(self, idx):
        """Load sample from real H5 file"""
        # This would load from actual SEVIR data
        # Implementation depends on your data format
        raise NotImplementedError("Real data loading not implemented yet")

def create_efficient_training_setup(num_samples=5000, batch_size=4, validation_split=0.2):
    """
    Create memory-efficient training setup
    
    Args:
        num_samples: Total number of samples to generate
        batch_size: Batch size for training
        validation_split: Fraction of data for validation
    
    Returns:
        train_generator, val_generator
    """
    # Calculate splits
    val_samples = int(num_samples * validation_split)
    train_samples = num_samples - val_samples
    
    print(f"Creating efficient data generators:")
    print(f"  Total samples: {num_samples}")
    print(f"  Training samples: {train_samples}")
    print(f"  Validation samples: {val_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Memory usage: ~{batch_size * 384 * 384 * 13 * 4 / 1e6:.1f} MB per batch")
    
    # Create generators
    train_generator = EfficientSEVIRGenerator(
        data_source='synthetic',
        num_samples=train_samples,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = EfficientSEVIRGenerator(
        data_source='synthetic',
        num_samples=val_samples,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_generator, val_generator

def train_with_generator(num_samples=5000, epochs=10, batch_size=4):
    """
    Train model using data generator for memory efficiency
    """
    print("🔄 Memory-Efficient Training with Data Generator")
    print("=" * 50)
    
    # Import here to avoid import issues in main script
    import sys
    sys.path.append('src/')
    from models.nowcast_unet import create_model
    
    # Create model
    inputs, outputs = create_model(input_shape=(384, 384, 13))
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print(f"✓ Model created with {model.count_params():,} parameters")
    
    # Create data generators
    train_gen, val_gen = create_efficient_training_setup(
        num_samples=num_samples,
        batch_size=batch_size
    )
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint('models/efficient_model.h5', save_best_only=True)
    ]
    
    # Train model
    print(f"\n🚀 Starting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Steps per epoch: {len(train_gen)}")
    print(f"  Validation steps: {len(val_gen)}")
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ Training completed!")
    print("Model saved to: models/efficient_model.h5")
    
    return model, history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Memory-Efficient SEVIR Training')
    parser.add_argument('--num_samples', type=int, default=5000, 
                       help='Number of samples to generate')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Run training
    model, history = train_with_generator(
        num_samples=args.num_samples,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
