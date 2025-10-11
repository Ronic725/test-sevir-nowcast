# Migration Guide: Centralized Path Configuration

This document explains the migration from hardcoded paths to centralized YAML-based configuration.

## What Changed

### Before Migration
Scripts used manual path calculations relative to `__file__`:
```python
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CACHE = PROJECT_ROOT / "custom" / "fine_tuning" / "cache" / "finetune_events.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "custom" / "fine_tuning" / "outputs" / "lora_adapter.npz"
models_root = PROJECT_ROOT / "models"
```

### After Migration
Scripts import and use centralized configuration:
```python
from config.project_paths import get_paths
paths = get_paths()
paths.setup_python_path()

# Access configured paths
cache = paths.finetune_cache
adapter = paths.lora_adapter
models = paths.models
```

## Benefits

✅ **Single Source of Truth** - All paths defined in `config/paths.yaml`  
✅ **Auto-Resolution** - Relative paths converted to absolute automatically  
✅ **Easy Maintenance** - Change paths once, affects all scripts  
✅ **Environment Flexibility** - Easy to configure for different setups  
✅ **Type-Safe Access** - IDE autocomplete for common paths  

## Migrated Scripts

All fine-tuning scripts now use centralized configuration:

### 1. `select_events_v2.py`
- ✅ Uses `paths.finetune_cache` for output
- ✅ Uses `paths.sevir_vil_file` for SEVIR data
- ✅ Backward compatible with CLI arguments

### 2. `quick_peft_finetune.py`
- ✅ Default cache: `paths.finetune_cache`
- ✅ Default output: `paths.lora_adapter`
- ✅ Models directory: `paths.models`

### 3. `evaluate_adapter.py`
- ✅ Default cache: `paths.finetune_cache`
- ✅ Default adapter: `paths.lora_adapter`
- ✅ Models directory: `paths.models`

### 4. `inspect_cache.py`
- ✅ Default NPZ: `paths.finetune_cache`
- ✅ SEVIR constants: `paths.sevir_mean`, `paths.sevir_scale`

## Configuration Files

### `config/paths.yaml`
Central configuration defining all project paths:
```yaml
project:
  root: "."
  src: "src"
  data: "data"
  models: "models"
  logs: "logs"

data:
  sevir:
    vil_2019_h1: "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
    catalog: "data/CATALOG.csv"

custom:
  fine_tuning:
    events_cache: "custom/fine_tuning/cache/finetune_events.npz"
    lora_adapter: "custom/fine_tuning/outputs/lora_adapter.npz"

sevir:
  mean: 33.44
  scale: 47.54
```

### `config/project_paths.py`
Python utility class for loading and accessing paths:
```python
from config.project_paths import get_paths

paths = get_paths()

# Convenience properties
cache = paths.finetune_cache  # Auto-resolved to absolute path
adapter = paths.lora_adapter
vil_file = paths.sevir_vil_file

# Dot notation for any path
results = paths.get('custom.results.model_performance')

# Constants
mean = paths.sevir_mean
scale = paths.sevir_scale
```

## Usage Examples

### Running Scripts from Workspace Root
All scripts now work from the workspace root directory:

```bash
# Sample events
python custom/fine_tuning/select_events_v2.py --num-events 50

# Train LoRA adapter
python custom/fine_tuning/quick_peft_finetune.py --epochs 3 --rank 8

# Evaluate adapter
python custom/fine_tuning/evaluate_adapter.py --visualize

# Inspect cache
python custom/fine_tuning/inspect_cache.py
```

### Custom Paths (Optional)
You can still override paths with CLI arguments:
```bash
python custom/fine_tuning/select_events_v2.py \
  --num-events 50 \
  --output custom/experiments/my_cache.npz

python custom/fine_tuning/quick_peft_finetune.py \
  --cache custom/experiments/my_cache.npz \
  --output custom/experiments/my_adapter.npz
```

## Migrating Other Scripts

To migrate additional scripts to use centralized configuration:

### Step 1: Import and Initialize
```python
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import and setup paths
from config.project_paths import get_paths
paths = get_paths()
paths.setup_python_path()
```

### Step 2: Replace Hardcoded Paths
```python
# OLD:
data_file = PROJECT_ROOT / "data" / "sevir" / "vil" / "SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
models_dir = PROJECT_ROOT / "models"

# NEW:
data_file = paths.sevir_vil_file
models_dir = paths.models
```

### Step 3: Update Constants
```python
# OLD:
SEVIR_MEAN = 33.44
SEVIR_SCALE = 47.54

# NEW:
mean = paths.sevir_mean
scale = paths.sevir_scale
```

### Step 4: Test
```bash
python path/to/your/script.py --help
```

## Available Path Properties

Common paths available as properties on `ProjectPaths`:

| Property | Description | Example |
|----------|-------------|---------|
| `paths.root` | Project root directory | `/Users/.../neurips-2020-sevir` |
| `paths.src` | Source code directory | `.../src` |
| `paths.data` | Data directory | `.../data` |
| `paths.models` | Models directory | `.../models` |
| `paths.logs` | Logs directory | `.../logs` |
| `paths.sevir_vil_file` | SEVIR VIL H5 file | `.../data/sevir/vil/SEVIR_VIL...h5` |
| `paths.catalog_file` | SEVIR catalog CSV | `.../data/CATALOG.csv` |
| `paths.finetune_cache` | Fine-tuning cache NPZ | `.../cache/finetune_events.npz` |
| `paths.lora_adapter` | LoRA adapter weights | `.../outputs/lora_adapter.npz` |
| `paths.results_dir` | Results directory | `.../custom/results/model_performance` |
| `paths.sevir_mean` | SEVIR normalization mean | `33.44` |
| `paths.sevir_scale` | SEVIR normalization scale | `47.54` |

## Troubleshooting

### ModuleNotFoundError
If you see `ModuleNotFoundError: No module named 'models'`:
- **Cause**: `paths.setup_python_path()` called after imports
- **Fix**: Call `paths.setup_python_path()` before importing from `src/` or `models/`

```python
# WRONG:
from config.project_paths import get_paths
from models.nowcast_unet import create_model  # ❌ Error!
paths = get_paths()
paths.setup_python_path()

# CORRECT:
from config.project_paths import get_paths
paths = get_paths()
paths.setup_python_path()  # Call first
from models.nowcast_unet import create_model  # ✅ Works!
```

### PyYAML Not Installed
If you see "PyYAML not available, using fallback configuration":
```bash
pip install pyyaml
```

The system will work with fallback defaults, but installing PyYAML is recommended.

### Path Not Found
If a path doesn't exist:
1. Check `config/paths.yaml` for correct relative path
2. Verify the file/directory exists relative to project root
3. Use `paths.get('key.path')` to debug the resolved absolute path

## Next Steps

Consider migrating these directories:
- [ ] `custom/testing/` - Test scripts with hardcoded paths
- [ ] `custom/streaming/` - Real-time prediction scripts
- [ ] `custom/training/` - Training scripts

Each directory can benefit from the same centralized configuration approach.
