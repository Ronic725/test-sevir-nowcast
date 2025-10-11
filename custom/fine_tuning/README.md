# LoRA Fine-Tuning for Nowcast Model

Quick Parameter-Efficient Fine-Tuning (PEFT) using **LoRA** (Low-Rank Adaptation) for real-time extreme weather tracking.

## 🎯 Centralized Configuration

All scripts now use **centralized path management** via `config/paths.yaml` and `config/project_paths.py`:
- ✅ Run from **workspace root** (no need to `cd` into directories)
- ✅ All paths auto-resolved from YAML configuration
- ✅ Single source of truth for data/model locations
- ✅ Easy to modify paths for different environments

## Overview

Three Python scripts for <5 minute fine-tuning on 50 similar weather events:
- **`select_events_v2.py`** – Cache training data (migrated to centralized config)
- **`quick_peft_finetune.py`** – Train LoRA adapter (~500 KB)
- **`evaluate_adapter.py`** – Compare baseline vs adapted model
- **`inspect_cache.py`** – View NPZ files and visualize samples

## Why LoRA?

| Feature | Bottleneck Adapters | LoRA ✅ | Prefix Tuning |
|---------|-------------------|---------|---------------|
| Inference speed | +5-10% slower | **No overhead** | +2-5% slower |
| Training time | ~3-5 min | ~3-5 min | ~3-5 min |
| Storage | 2-5 MB | **~500 KB** | <1 MB |
| CNN support | ✅ Native | ✅ Native | ❌ Transformers only |
| Model portability | ⚠️ Architecture-specific | ✅ **Easy transfer** | ⚠️ Input-dependent |

**Best for real-time:** LoRA has zero forward-time penalty and works seamlessly with larger models later

## Quick Start

**Run from workspace root** (scripts auto-detect paths from `config/paths.yaml`):

### 1. Sample Events (10 seconds)
```bash
# New version using centralized config
python custom/fine_tuning/select_events_v2.py --num-events 50 --seed 123

# Or use the legacy version
python custom/fine_tuning/select_events.py --num-events 50 --seed 123
```
✅ Caches 50 random SEVIR events to `custom/fine_tuning/cache/finetune_events.npz`

### 2. Fine-Tune with LoRA (<5 minutes)
```bash
python custom/fine_tuning/quick_peft_finetune.py --rank 8 --epochs 3
```
✅ Trains lightweight LoRA adapters → `custom/fine_tuning/outputs/lora_adapter.npz`

**Arguments:**
- `--rank`: Low-rank dimension (default: 8). Higher = more capacity but slower.
- `--epochs`: Training epochs (default: 3)
- `--batch-size`: Mini-batch size (default: 2)
- `--learning-rate`: Optimizer LR (default: 3e-4)

### 3. Evaluate & Compare
```bash
python custom/fine_tuning/evaluate_adapter.py --rank 8 --visualize
```
✅ Compares baseline vs LoRA-adapted model, shows MSE/MAE improvements + visualizations

### Inspect Cached Data (Optional)
```bash
# View metadata about cached events
python custom/fine_tuning/inspect_cache.py

# Visualize a specific sample
python custom/fine_tuning/inspect_cache.py --visualize --sample 0
```
✅ Shows shape, statistics, and optionally visualizes radar frames from cached `.npz` files

**Note:** All scripts resolve paths automatically - works from workspace root or `custom/fine_tuning/`

## Usage in Real-Time System

### Scenario: Extreme Weather Event Detected

1. **Identify similar historical events** (your separate system)
2. **Cache those events:**
   ```python
   from custom.testing.test_actual_sevir import load_real_sevir_data_direct
   X, y = load_real_sevir_data_direct(data_file, num_samples=50, seed=42)
   np.savez_compressed("custom/fine_tuning/cache/extreme_events.npz", inputs=X, targets=y)
   ```

3. **Fast fine-tune (< 5 min):**
   ```bash
   python custom/fine_tuning/quick_peft_finetune.py --cache custom/fine_tuning/cache/extreme_events.npz --epochs 3
   ```

4. **Deploy instantly** – load LoRA weights without restarting inference:
   ```python
   from custom.fine_tuning.quick_peft_finetune import build_lora_model
   
   lora_model = build_lora_model(rank=8)
   lora_data = np.load("custom/fine_tuning/outputs/lora_adapter.npz")
   # Assign weights...
   predictions = lora_model.predict(current_radar_data)
   ```

5. **Swap to bigger model later** – reuse adapter architecture with new base weights.

## Architecture Details

**LoRA Injection:**
- Targets Conv2D layers with ≥16 filters
- Each layer gets two trainable matrices: A (input→rank) and B (rank→output)
- Forward pass: `output = base_conv(x) + conv(conv(x, A), B)`
- Typical parameter reduction: **~2-5% of full model**

**File Sizes:**
- Cached events (50): ~150 MB
- LoRA adapter (rank=8): ~500 KB
- Full base model: ~80 MB

## Hyperparameter Tuning

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `--rank` | Adapter capacity | 8 for quick, 16-32 for complex patterns |
| `--epochs` | Training iterations | 3-5 sufficient for 50 events |
| `--learning-rate` | Step size | 1e-4 (conservative) to 5e-4 (aggressive) |
| `--num-events` | Training set size | 30-100 (more = better but slower) |

## Inspecting NPZ Files

View `.npz` cache files to verify data and understand structure:

```bash
# Quick inspection - shows shape, dtype, statistics
python custom/fine_tuning/inspect_cache.py

# Visualize sample radar frames
python custom/fine_tuning/inspect_cache.py --visualize --sample 0

# Inspect custom NPZ file
python custom/fine_tuning/inspect_cache.py path/to/custom.npz --visualize
```

**Example output:**
```
📊 Available arrays: ['inputs', 'targets']

🔹 inputs:
   Shape: (50, 384, 384, 13)      # 50 events, 13 input frames
   Dtype: float32
   Range: [-0.7023, 0.7646]       # Normalized
   Range (dBZ): [0.0, 70.0]       # In dBZ units

🔹 targets:
   Shape: (50, 384, 384, 12)      # 12 target frames to predict
```

## Troubleshooting

**"conda/python command not found" in VS Code terminal**:
```bash
# Source your shell profile first:
source ~/.bash_profile    # or source ~/.zshrc
# Then run the scripts
python custom/fine_tuning/select_events.py
```

**"No trained model found"**: Run `python custom/training/train_limited_data.py` first.

**Poor performance**: Try higher rank (16 or 32) or more events (100).

**Out of memory**: Reduce `--batch-size` to 1.

**Want faster**: Lower `--rank` to 4 or use fewer events.

**Inspect cached data**: Use `inspect_cache.py` to verify the data looks correct.

## Technical Details

### LoRA Architecture
```python
# LoRA injects into Conv2D layers with ≥16 filters:
output = base_conv(x) + conv(conv(x, A), B)

# Where:
# A: (kernel_h, kernel_w, in_channels, rank)  - Down-projection
# B: (1, 1, rank, filters)                    - Up-projection
# Trainable params: ~2-5% of full model
```

### Performance Metrics
- **Cached events (50)**: ~150 MB
- **LoRA adapter (rank=8)**: ~500 KB  
- **Full base model**: ~80 MB
- **Training time**: 3-5 minutes on 50 events
- **Inference overhead**: 0% (after optional weight merging)

## Production Deployment

### Weight Merging for Zero Latency
```python
def merge_lora_weights(base_model, lora_weights):
    """One-time merge: W_new = W_base + (B @ A)"""
    for layer in base_model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            # Merge LoRA into base weights
            # Result: no runtime overhead
            ...
```

### Real-Time Integration
1. Run alongside `custom/streaming/realtime_predictor.py`
2. Hot-swap adapter when extreme weather detected
3. Continue predictions without restarting service
4. Optionally merge weights for production deployment

## Next Steps

- **Model comparison**: `custom/testing/compare_models.py`
- **Streaming integration**: `custom/streaming/realtime_predictor.py`
- **A/B testing**: Compare adapted vs baseline in production
- **Continuous learning**: Auto-retrain on new extreme events
