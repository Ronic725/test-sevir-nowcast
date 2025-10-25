#!/usr/bin/env python3
"""Compare baseline and LoRA-adapted nowcast models on cached events."""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras  # type: ignore[attr-defined]

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import centralized configuration and setup paths
from config.project_paths import get_paths
paths = get_paths()
paths.setup_python_path()

# Now import from custom/
from custom.testing.test_actual_sevir import test_on_real_sevir
from custom.fine_tuning.quick_peft_finetune import LoRAConv2D, apply_lora_to_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapter against the baseline model.")
    parser.add_argument("--cache", type=Path, default=paths.finetune_cache, help="NPZ file with evaluation samples.")
    parser.add_argument("--adapter", type=Path, default=paths.lora_adapter, help="LoRA weights (NPZ) produced by fine-tuning.")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank used during training.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Emit a matplotlib visualization using the evaluation dataset.",
    )
    return parser.parse_args()


def find_latest_model_dir() -> Path:
    models_root = paths.models
    candidates = [p for p in models_root.iterdir() if p.is_dir() and p.name.startswith("trained_")]
    if not candidates:
        raise FileNotFoundError("No trained model directory found under models/.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_cached_events(cache_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Cached dataset not found: {cache_path}")
    data = np.load(cache_path)
    return data["inputs"].astype(np.float32), data["targets"].astype(np.float32)


def build_lora_model(rank: int, lora_weights_path: Path) -> keras.Model:
    """Build LoRA model and load adapter weights."""
    latest_dir = find_latest_model_dir()
    model_path = latest_dir / "model_mse.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    # Load with custom objects to avoid warnings
    base_model = keras.models.load_model(
        str(model_path), 
        compile=False,
        custom_objects={'LoRAConv2D': LoRAConv2D}
    )
    
    # Freeze base model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Apply LoRA adapters
    lora_model = apply_lora_to_model(base_model, rank)
    
    # Load LoRA weights
    if not lora_weights_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_weights_path}")
    
    lora_data = np.load(lora_weights_path)
    
    # Assign LoRA weights to the model
    for layer in lora_model.layers:
        if isinstance(layer, LoRAConv2D):
            for weight in layer.trainable_weights:
                weight_name = weight.name
                if weight_name in lora_data:
                    weight.assign(lora_data[weight_name])
    
    return lora_model


def evaluate_model(model: keras.Model, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model and return metrics plus predictions for visualization."""
    y_pred, mse, mae, y_pred_dbz, y_test_dbz = test_on_real_sevir(model, X, y)
    if mse is None or mae is None or y_pred is None or y_pred_dbz is None or y_test_dbz is None:
        raise RuntimeError("test_on_real_sevir returned empty metrics; verify inputs.")
    return float(mse), float(mae), y_pred, y_pred_dbz, y_test_dbz


def print_data_statistics(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, 
                         y_pred_dbz: np.ndarray, y_test_dbz: np.ndarray) -> None:
    """Print comprehensive statistics about input data and model output."""
    from config.project_paths import get_paths
    paths = get_paths()
    SEVIR_MEAN = paths.sevir_mean
    SEVIR_SCALE = paths.sevir_scale
    
    print("\n" + "=" * 60)
    print("📊 DATA STATISTICS")
    print("=" * 60)
    
    # Input statistics (normalized)
    X_dbz = X * SEVIR_SCALE + SEVIR_MEAN
    print("\n🔹 Input Data (X):")
    print(f"   Shape: {X.shape}")
    print(f"   Normalized range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"   Normalized mean: {X.mean():.4f}, std: {X.std():.4f}")
    print(f"   dBZ range: [{X_dbz.min():.2f}, {X_dbz.max():.2f}]")
    print(f"   dBZ mean: {X_dbz.mean():.2f}, std: {X_dbz.std():.2f}")
    
    # Target statistics (normalized)
    print("\n🔹 Target Data (y):")
    print(f"   Shape: {y.shape}")
    print(f"   Normalized range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"   Normalized mean: {y.mean():.4f}, std: {y.std():.4f}")
    print(f"   dBZ range: [{y_test_dbz.min():.2f}, {y_test_dbz.max():.2f}]")
    print(f"   dBZ mean: {y_test_dbz.mean():.2f}, std: {y_test_dbz.std():.2f}")
    
    # Prediction statistics (normalized)
    print("\n🔹 Model Predictions (y_pred):")
    print(f"   Shape: {y_pred.shape}")
    print(f"   Normalized range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print(f"   Normalized mean: {y_pred.mean():.4f}, std: {y_pred.std():.4f}")
    print(f"   dBZ range: [{y_pred_dbz.min():.2f}, {y_pred_dbz.max():.2f}]")
    print(f"   dBZ mean: {y_pred_dbz.mean():.2f}, std: {y_pred_dbz.std():.2f}")
    
    # Distribution comparison
    print("\n🔹 Distribution Comparison:")
    print(f"   Target vs Prediction (normalized):")
    print(f"      Mean difference: {abs(y.mean() - y_pred.mean()):.4f}")
    print(f"      Std difference: {abs(y.std() - y_pred.std()):.4f}")
    print(f"   Target vs Prediction (dBZ):")
    print(f"      Mean difference: {abs(y_test_dbz.mean() - y_pred_dbz.mean()):.2f} dBZ")
    print(f"      Std difference: {abs(y_test_dbz.std() - y_pred_dbz.std()):.2f} dBZ")


def maybe_visualize(model: keras.Model, X: np.ndarray, y: np.ndarray) -> None:
    """Visualize predictions using properly scaled data."""
    from custom.testing.test_actual_sevir import visualize_real_sevir_results
    
    # Get predictions in normalized form
    predictions = model.predict(X, batch_size=2, verbose=0)
    
    # visualize_real_sevir_results expects normalized data and will convert to dBZ internally
    # So we pass normalized X, y, and predictions
    visualize_real_sevir_results(X, y, predictions, sample_idx=0)


def main(args: argparse.Namespace) -> None:
    print("🔍 LoRA Adapter Evaluation")
    print("=" * 60)
    
    X, y = load_cached_events(args.cache)
    print(f"✅ Loaded {len(X)} cached events for evaluation")
    
    # Load baseline model
    latest_dir = find_latest_model_dir()
    model_path = latest_dir / "model_mse.h5"
    baseline = keras.models.load_model(str(model_path), compile=False)
    print(f"📁 Baseline model: {model_path.name}")

    # Load LoRA-adapted model
    lora_model = build_lora_model(args.rank, args.adapter)
    print(f"📁 LoRA adapter: {args.adapter.name} (rank={args.rank})")

    # Evaluate both models
    print("\n🧪 Evaluating baseline model...")
    base_mse, base_mae, base_pred, base_pred_dbz, base_test_dbz = evaluate_model(baseline, X, y)
    print_data_statistics(X, y, base_pred, base_pred_dbz, base_test_dbz)
    
    print("\n🧪 Evaluating LoRA model...")
    lora_mse, lora_mae, lora_pred, lora_pred_dbz, lora_test_dbz = evaluate_model(lora_model, X, y)
    print_data_statistics(X, y, lora_pred, lora_pred_dbz, lora_test_dbz)

    # Display results
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    print("Baseline -> MSE: {:.4f}, MAE: {:.4f}".format(base_mse, base_mae))
    print("LoRA     -> MSE: {:.4f}, MAE: {:.4f}".format(lora_mse, lora_mae))
    
    improvement_mse = ((base_mse - lora_mse) / base_mse) * 100
    improvement_mae = ((base_mae - lora_mae) / base_mae) * 100
    
    print("\n📈 Improvement:")
    print(f"   MSE: {improvement_mse:+.2f}%")
    print(f"   MAE: {improvement_mae:+.2f}%")

    if args.visualize:
        print("\n🎨 Generating visualization...")
        maybe_visualize(lora_model, X, y)


if __name__ == "__main__":
    main(parse_args())
