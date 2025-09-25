# SEVIR Real-time Streaming and Prediction System

This directory contains a real-time data streaming simulator and prediction system for SEVIR weather radar data. The system simulates continuous data arrival and makes sliding window predictions, laying the foundation for online fine-tuning capabilities.

## Components

### 1. `sevir_data_streamer.py`
**SEVIR Data Streaming Simulator** - Simulates real-time weather radar data stream from SEVIR dataset.

**Key Features:**
- Configurable sliding window size (default: 13 frames to match model input)
- Adjustable stream interval (5 minutes real-time, faster for demos)
- Thread-safe data buffering with queue management
- Automatic data normalization using SEVIR constants
- Support for both real and synthetic SEVIR data

**Usage:**
```python
from sevir_data_streamer import SEVIRDataStreamer

streamer = SEVIRDataStreamer(
    data_file="path/to/sevir.h5",
    window_size=13,
    stream_interval=5.0,  # minutes
    normalize=True
)

streamer.start_streaming(event_idx=0)
window, metadata = streamer.get_sliding_window()
streamer.stop_streaming()
```

### 2. `realtime_predictor.py`
**Real-time Weather Prediction System** - Continuously processes streaming data and makes predictions using trained models.

**Key Features:**
- Sliding window prediction with configurable horizon
- Anomaly detection using threshold-based approach
- Automatic visualization and result saving
- Performance metrics tracking
- Compatible with existing trained models

**Usage:**
```python
from realtime_predictor import RealtimePredictor

predictor = RealtimePredictor(
    model_path="path/to/model.h5",
    anomaly_threshold=60.0,  # dBZ
    save_predictions=True
)

predictor.process_stream(streamer, max_predictions=10)
```

### 3. `run_realtime_demo.py`
**Demo Script** - Complete demonstration of the real-time system with automatic model and data file detection.

**Usage:**
```bash
# Test just the data streamer
python custom/streaming/run_realtime_demo.py --test-streamer

# Run full real-time prediction demo
python custom/streaming/run_realtime_demo.py
```

### 4. `test_streaming.py`
**Test Script** - Basic functionality tests for the streaming components.

**Usage:**
```bash
python custom/streaming/test_streaming.py
```

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  SEVIR Data     │───▶│  Data Streamer   │───▶│  Sliding Window │
│  (H5 Files)     │    │  (Buffer Queue)  │    │  (13 frames)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Visualization  │◀───│  Real-time       │◀───│  Trained Model  │
│  & Results      │    │  Predictor       │    │  (MSE/GAN)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Anomaly        │
                       │  Detection      │
                       └─────────────────┘
```

## Configuration

### Timeframe Settings
Based on analysis of your existing models:
- **Input Window**: 13 frames (matches trained model requirements)
- **Prediction Horizon**: 12 frames (standard SEVIR nowcasting setup)
- **Stream Interval**: 5 minutes (real SEVIR temporal resolution) or faster for testing

### Model Compatibility
The system automatically detects and works with:
- MSE-trained models (`model_mse.h5`)
- GAN-trained models (`gan_generator.h5`)
- Models in `models/trained_*` directories
- Models in `models/nowcast/` directory

### Data Requirements
- SEVIR VIL data in H5 format
- Expected shape: `(N_events, 384, 384, 25)` or similar
- Automatic normalization using SEVIR constants (mean=33.44, scale=47.54)

## Quick Start

1. **Test the system:**
   ```bash
   python custom/streaming/test_streaming.py
   ```

2. **Run streaming demo:**
   ```bash
   python custom/streaming/run_realtime_demo.py --test-streamer
   ```

3. **Run full prediction demo:**
   ```bash
   python custom/streaming/run_realtime_demo.py
   ```

## Output

The system generates:
- **Real-time predictions**: Saved as PNG files with timestamps
- **Anomaly alerts**: Console notifications when extreme weather detected
- **Performance metrics**: Prediction timing and accuracy statistics
- **Visualization plots**: Input frames, predictions, and anomaly highlighting

Results are saved to: `custom/results/realtime_predictions/`

## Future Extensions

This system provides the foundation for:
1. **Online Fine-tuning**: Adapt models based on streaming prediction errors
2. **Multi-model Ensemble**: Combine predictions from multiple models
3. **Advanced Anomaly Detection**: ML-based anomaly detection beyond thresholds
4. **Real-time Data Integration**: Connect to actual weather radar feeds
5. **Interactive Dashboard**: Web-based real-time monitoring interface

## Dependencies

- `numpy`, `h5py`, `tensorflow/keras`
- `matplotlib` (for visualization)
- `threading`, `queue` (for concurrency)
- Existing SEVIR project dependencies

## Notes

- The system uses Python threading for concurrent data streaming and prediction
- Buffer management prevents memory overflow during long runs
- Type checking may show warnings for h5py operations (these are normal)
- Stream intervals can be adjusted for real-time vs. demo scenarios
