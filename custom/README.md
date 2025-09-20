# Custom SEVIR Training and Testing Scripts

This directory contains custom scripts and utilities for weather nowcasting with SEVIR data, optimized for M1 MacBook Pro (16GB RAM).

## Directory Structure

```
custom/
├── training/           # Custom training scripts
│   ├── quick_start_training.py     # Interactive training menu
│   └── train_limited_data.py       # Limited data training for resource-constrained systems
├── testing/            # Model testing and evaluation
│   ├── test_trained_model.py       # Basic model testing on synthetic data
│   ├── test_sevir_compatible.py    # SEVIR format compatibility testing
│   ├── test_realistic_sevir.py     # Testing on realistic weather patterns
│   ├── test_real_sevir.py          # Testing on real SEVIR data (requires download)
│   └── analyze_performance.py      # Detailed performance analysis
├── results/            # Generated outputs and visualizations
│   ├── *.png           # Visualization results
│   └── *.h5            # Test data files
└── experiments/        # Future experimental scripts
```

## Quick Start

### 1. Interactive Training
```bash
cd custom/training
python quick_start_training.py
```

Choose from optimized training options:
- **Option 1**: Quick test (256 samples, 5 epochs) - 5-10 minutes
- **Option 2**: Medium training (1024 samples, 15 epochs) - 20-30 minutes  
- **Option 3**: Advanced training (2048 samples, 20 epochs) - 1-2 hours

### 2. Direct Training
```bash
cd custom/training
python train_limited_data.py --num_samples 512 --epochs 10 --loss_type mse --batch_size 4
```

### 3. Test Your Model
```bash
cd custom/testing
python test_trained_model.py           # Basic synthetic test
python test_realistic_sevir.py         # Realistic weather patterns
python test_real_sevir.py              # Real SEVIR data (if available)
```

## Training Parameters

### Recommended Settings for M1 MacBook Pro (16GB)

| Configuration | Samples | Epochs | Batch Size | Time | Memory Usage |
|---------------|---------|--------|------------|------|--------------|
| **Quick Test** | 256 | 5 | 8 | ~10 min | Low |
| **Development** | 512 | 10 | 4 | ~20 min | Medium |
| **Production** | 1024+ | 20+ | 4 | ~1+ hour | High |

### Available Parameters

- `--num_samples`: Number of training samples (256, 512, 1024, 2048)
- `--epochs`: Training epochs (5, 10, 15, 20, 25, 30)
- `--loss_type`: Loss function (`mse`, `mae`, `vgg`)
- `--batch_size`: Batch size (2, 4, 8, 16)
- `--validation_split`: Validation ratio (0.1, 0.2, 0.3)

## Loss Functions

- **MSE**: Fast training, good for initial experiments
- **MAE**: More robust to outliers  
- **VGG**: Perceptual loss, better visual quality (requires more memory)

## Model Testing

### Performance Metrics
- **MSE/MAE**: Lower is better
- **dBZ Scale**: Weather radar reflectivity scale (0-70 dBZ typical)
- **Baseline Comparison**: Persistence model (last frame repeated)

### Expected Performance
- **Excellent**: MAE < 10 dBZ
- **Good**: MAE < 15 dBZ  
- **Acceptable**: MAE < 20 dBZ
- **Needs Improvement**: MAE > 20 dBZ

## Data Formats

### Synthetic Data
- Simple moving geometric patterns
- Good for architecture testing
- Fast generation and training

### Realistic SEVIR-like Data  
- Complex storm systems
- Multiple precipitation types
- Realistic radar artifacts
- Better domain transfer

### Real SEVIR Data
- Actual weather radar sequences
- Complex meteorological phenomena
- Requires download from AWS S3
- Best for production models

## Hardware Optimization

### M1 MacBook Pro Features Used
- **Metal GPU acceleration**: TensorFlow-Metal for training
- **Unified memory**: Efficient data loading
- **ARM optimization**: Native TensorFlow builds
- **Thermal management**: Batch size optimization

### Memory Management
- Batch sizes optimized for 16GB RAM
- Gradient checkpointing for large models
- Data generators to reduce memory footprint
- Automatic mixed precision (future)

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch_size and num_samples
2. **Slow Training**: Check GPU acceleration with `tf.config.list_physical_devices('GPU')`
3. **Import Errors**: Ensure you're in the custom subdirectories
4. **Path Issues**: Scripts automatically adjust paths to find original repo modules

### Performance Tips
- Start with quick test to verify setup
- Monitor memory usage with Activity Monitor
- Use VGG loss only after MSE convergence
- Close other applications during training

## Future Enhancements

- [ ] Real SEVIR data integration
- [ ] Distributed training support  
- [ ] Advanced data augmentation
- [ ] Model ensemble techniques
- [ ] Deployment optimization
- [ ] MLOps integration

## Results Interpretation

### Training Curves
- Loss should decrease steadily
- Validation loss should follow training loss
- Early stopping if validation loss increases

### Visual Results
- Check `custom/results/` for generated visualizations
- Compare input sequences, ground truth, and predictions
- Look for temporal consistency across frames

### Quantitative Metrics
- MSE/MAE in normalized space (0-1 range)
- dBZ scale metrics for weather interpretation
- Frame-by-frame analysis for temporal evaluation

---

**Created for M1 MacBook Pro optimization**  
**Compatible with original SEVIR repository structure**