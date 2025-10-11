#!/usr/bin/env python3
"""Sample SEVIR events for quick PEFT fine-tuning - Using centralized config."""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add config to path and import centralized paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.project_paths import get_paths
from custom.testing.test_actual_sevir import load_real_sevir_data_direct

# Initialize paths
paths = get_paths()
paths.setup_python_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache a random subset of SEVIR events.")
    parser.add_argument("--num-events", type=int, default=50, help="Number of SEVIR events to sample.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,  # Will use config default
        help="Path to the cached NPZ file that stores inputs/targets.",
    )
    return parser.parse_args()


def main(num_events: int, seed: int, output_path: Path | None = None) -> None:
    # Use configured path if not specified
    if output_path is None:
        output_path = paths.finetune_cache
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use configured SEVIR file path
    data_file = paths.sevir_vil_file
    
    X, y = load_real_sevir_data_direct(str(data_file), num_events, seed)
    if X is None or y is None:
        raise SystemExit("Failed to load SEVIR samples; ensure the dataset is available.")
    
    np.savez_compressed(output_path, inputs=X.astype(np.float32), targets=y.astype(np.float32))
    print(f"✅ Saved {num_events} events to {output_path}")
    print(f"   File size: {output_path.stat().st_size / (1024**2):.1f} MB")


if __name__ == "__main__":
    args = parse_args()
    main(args.num_events, args.seed, args.output)
