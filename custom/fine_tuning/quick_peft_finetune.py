#!/usr/bin/env python3
"""Fine-tune the nowcast model with LoRA (Low-Rank Adaptation)."""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict

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

# Now import from src/
from models.nowcast_unet import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quick LoRA fine-tune on cached SEVIR events.")
    parser.add_argument("--cache", type=Path, default=paths.finetune_cache, help="NPZ file with inputs/targets.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="LoRA learning rate.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Fine-tuning batch size.")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank (dimensionality of low-rank matrices).")
    parser.add_argument("--output", type=Path, default=paths.lora_adapter, help="NPZ file for LoRA adapter weights.")
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


class LoRAConv2D(keras.layers.Layer):
    """Conv2D layer with LoRA (Low-Rank Adaptation)."""
    
    def __init__(self, base_layer: keras.layers.Conv2D, rank: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.base_layer = base_layer
        self.base_layer.trainable = False
        self.rank = rank
        
        # Get base layer properties
        filters = base_layer.filters
        kernel_size = base_layer.kernel_size
        
        # LoRA matrices: W = W_base + B @ A
        # A: (kernel_h, kernel_w, in_channels, rank)
        # B: (1, 1, rank, filters)
        self.lora_A = self.add_weight(
            name=f"{base_layer.name}_lora_A",
            shape=(*kernel_size, base_layer.input_shape[-1], rank),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.lora_B = self.add_weight(
            name=f"{base_layer.name}_lora_B",
            shape=(1, 1, rank, filters),
            initializer="zeros",
            trainable=True,
        )
    
    def call(self, inputs):
        # Base convolution (frozen)
        base_output = self.base_layer(inputs)
        
        # LoRA path: Conv(input, A) then Conv(result, B)
        lora_out = tf.nn.conv2d(inputs, self.lora_A, strides=1, padding='SAME')
        lora_out = tf.nn.conv2d(lora_out, self.lora_B, strides=1, padding='SAME')
        
        return base_output + lora_out


def apply_lora_to_model(model: keras.Model, rank: int) -> keras.Model:
    """Replace Conv2D layers with LoRA-enhanced versions."""
    # Create a mapping of layer outputs
    layer_dict = {layer.name: layer for layer in model.layers}
    
    # Build new model with LoRA layers
    def clone_with_lora(layer, input_tensors):
        if isinstance(layer, keras.layers.Conv2D) and layer.filters >= 16:
            # Apply LoRA to significant Conv2D layers
            return LoRAConv2D(layer, rank=rank, name=f"lora_{layer.name}")(input_tensors)
        else:
            # Keep other layers as-is
            return layer(input_tensors)
    
    # Reconstruct the model
    inputs = model.input
    x = inputs
    
    for layer in model.layers[1:]:  # Skip input layer
        x = clone_with_lora(layer, x)
    
    return keras.Model(inputs=inputs, outputs=x, name="nowcast_lora")


def build_lora_model(rank: int) -> keras.Model:
    """Load base model and apply LoRA."""
    latest_dir = find_latest_model_dir()
    model_path = latest_dir / "model_mse.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    base_model = keras.models.load_model(str(model_path), compile=False)
    
    # Freeze all base weights
    for layer in base_model.layers:
        layer.trainable = False
    
    # Apply LoRA adapters
    lora_model = apply_lora_to_model(base_model, rank)
    
    return lora_model


def extract_lora_weights(model: keras.Model) -> Dict[str, np.ndarray]:
    """Extract only LoRA trainable weights."""
    lora_weights = {}
    for layer in model.layers:
        if isinstance(layer, LoRAConv2D):
            for weight in layer.trainable_weights:
                lora_weights[weight.name] = weight.numpy()
    return lora_weights


def main(args: argparse.Namespace) -> None:
    print(f"🚀 LoRA Fine-tuning (rank={args.rank})")
    print("=" * 60)
    
    X, y = load_cached_events(args.cache)
    print(f"✅ Loaded {len(X)} cached events")
    
    model = build_lora_model(args.rank)
    
    # Count trainable parameters
    trainable_params = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
    total_params = int(np.sum([np.prod(w.shape) for w in model.weights]))
    print(f"📊 Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    print(f"\n🔧 Training for {args.epochs} epochs...")
    history = model.fit(
        X,
        y,
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
        verbose=2,
    )

    # Save only LoRA weights
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lora_weights = extract_lora_weights(model)
    np.savez(args.output, **lora_weights)
    
    print(f"\n✅ Saved {len(lora_weights)} LoRA weight tensors to {args.output}")
    print(f"💾 File size: {args.output.stat().st_size / 1024:.1f} KB")
    print("📈 Training history (last epoch):", {k: float(v[-1]) for k, v in history.history.items()})


if __name__ == "__main__":
    main(parse_args())
