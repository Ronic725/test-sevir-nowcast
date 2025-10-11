# ✅ Centralized Configuration - Complete Guide

**Status**: ✅ Applied and Verified | **Date**: October 11, 2025

## 🎯 Quick Start

All scripts now run from workspace root:

```bash
cd /Users/ronaldleung/Code/fyp-testing/neurips-2020-sevir

# Training
python3 custom/training/quick_start_training.py
python3 custom/training/train_limited_data.py --num_samples 512 --epochs 10

# Testing  
python3 custom/testing/test_actual_sevir.py
python3 custom/testing/compare_models.py

# Real-time Demo
python3 custom/streaming/run_realtime_demo.py
python3 custom/streaming/run_realtime_demo.py --model 20250918

# Fine-tuning
python3 custom/fine_tuning/quick_peft_finetune.py
python3 custom/fine_tuning/select_events_v2.py --num-events 50

# Data Analysis
python3 custom/sevir_dataset/inspect_sevir_data_refactored.py

# Download Data
python3 custom/experiments/download_sevir_data.py
```

## 📁 What Changed

### ✅ Migrated Folders (17 files total)

1. **`custom/fine_tuning/`** (5 files) - Already done previously
2. **`custom/streaming/`** (3 files) - `run_realtime_demo.py`, `realtime_predictor.py`, `sevir_data_streamer.py`
3. **`custom/testing/`** (3 files) - `test_actual_sevir.py`, `compare_models.py`, `test_trained_model.py`
4. **`custom/training/`** (2 files) - `quick_start_training.py`, `train_limited_data.py`
5. **`custom/experiments/`** (1 file) - `download_sevir_data.py`
6. **`custom/sevir_dataset/`** (3 files) - `inspect_sevir_data_refactored.py`, `examples/example_usage.py`, `analysis/sevir_analysis.py`

### Standard Import Pattern

All scripts now use this consistent pattern:

```python
#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import centralized configuration
from config.project_paths import get_paths
paths = get_paths()
paths.setup_python_path()

# Use configured paths
data_file = paths.sevir_vil_file  # Instead of hardcoded paths
models_dir = paths.models
SEVIR_MEAN = paths.sevir_mean     # Instead of 33.44 everywhere
```

## 📦 Available Paths & Constants

```python
from config.project_paths import get_paths
paths = get_paths()

# Directories
paths.root              # Project root
paths.data              # data/
paths.models            # models/
paths.src               # src/
paths.logs              # logs/

# Specific files
paths.sevir_vil_file    # data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5
paths.catalog_file      # data/CATALOG.csv
paths.finetune_cache    # custom/fine_tuning/cache/finetune_events.npz
paths.lora_adapter      # custom/fine_tuning/outputs/lora_adapter.npz
paths.results_dir       # custom/results/model_performance

# Constants
paths.sevir_mean        # 33.44
paths.sevir_scale       # 47.54
```

## 🎯 Key Benefits

1. ✅ **Run from workspace root** - No more `cd` into subdirectories
2. ✅ **Centralized paths** - All paths in `config/paths.yaml`
3. ✅ **No relative path issues** - No more `../../`
4. ✅ **Shared constants** - SEVIR constants from config
5. ✅ **Easy environment switching** - Update config, not every script

## 🧪 Verify Your Setup

```bash
python3 verify_config.py
```

Expected: ✅ All tests passed!

## 📝 Configuration Files

- **`config/paths.yaml`** - All project paths (YAML format)
- **`config/project_paths.py`** - Python module to access paths
- **`verify_config.py`** - Test script to verify everything works

## 🔧 Before vs After

### Before (Inconsistent):
```python
sys.path.append('../../')
sys.path.append('../../src/')
DATA_DIR = Path(__file__).parent.parent / "data"
SEVIR_MEAN = 33.44  # Hardcoded everywhere
```

### After (Consistent):
```python
from config.project_paths import get_paths
paths = get_paths()
paths.setup_python_path()
DATA_DIR = paths.data
SEVIR_MEAN = paths.sevir_mean
```

---

**Need more details?** The config auto-detects project root, has fallback hardcoded paths if PyYAML unavailable, and returns `pathlib.Path` objects for better path handling.
