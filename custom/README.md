# Custom SEVIR Weather Nowcasting

Scripts for weather prediction with SEVIR data, optimized for M1 MacBook Pro (16GB RAM).

## Directory Structure

```
custom/
├── training/           # Model training (quick_start_training.py, train_limited_data.py)
├── testing/            # Model evaluation (test_*.py, analyze_performance.py)
├── streaming/          # Real-time demo (run_realtime_demo.py, realtime_predictor.py)
├── results/            # Generated outputs and visualizations
└── experiments/        # Data download utilities
```

## Quick Start

### 1. Train a Model
```bash
cd custom/training
python quick_start_training.py         # Interactive menu (recommended)
# OR
python train_limited_data.py --num_samples 512 --epochs 10
```

### 2. Test Your Model  
```bash
cd custom/testing
python test_actual_sevir.py --model 20250918  # Test on real SEVIR data
```

### 3. Run Real-Time Demo
```bash
cd custom/streaming
python run_realtime_demo.py            # Use recommended model
python run_realtime_demo.py --model 20250918  # Use specific model
```

## Real-Time Demo

Simulates continuous weather prediction with sliding window approach and anomaly detection.

### Usage
```bash
python custom/streaming/run_realtime_demo.py                # Recommended model
python custom/streaming/run_realtime_demo.py --model 20250918  # Specific model
python custom/streaming/run_realtime_demo.py --test-streamer   # Test streaming only
```

### Understanding Results
- **Good predictions**: 80-150 dBZ (realistic storm values)
- **Bad predictions**: 1000+ dBZ (indicates model problems)
- **Anomaly alerts**: 🚨 when values exceed 70 dBZ threshold
- **Output**: Visualizations saved to `custom/results/realtime_predictions/`

### Recommended Model
Use `trained_mse_20250918_120133` for best performance - avoids unrealistic 1000+ dBZ predictions.

## Training Options

| Configuration | Samples | Epochs | Time | Memory |
|---------------|---------|--------|------|--------|
| **Quick Test** | 256 | 5 | ~10 min | Low |
| **Development** | 512 | 10 | ~20 min | Medium |
| **Production** | 1024+ | 20+ | ~1+ hour | High |

### Key Parameters
- `--num_samples`: Training samples (256, 512, 1024, 2048)
- `--epochs`: Training epochs (5, 10, 15, 20+)
- `--loss_type`: Loss function (`mse`, `mae`, `vgg`)
- `--batch_size`: Batch size (2, 4, 8, 16)

## Performance Metrics
- **Excellent**: MAE < 10 dBZ
- **Good**: MAE < 15 dBZ  
- **Acceptable**: MAE < 20 dBZ

## Troubleshooting
- **Out of Memory**: Reduce `batch_size` and `num_samples`
- **Slow Training**: Check GPU with `tf.config.list_physical_devices('GPU')`
- **Import Errors**: Run scripts from their respective directories

---
*Optimized for M1 MacBook Pro (16GB RAM)*