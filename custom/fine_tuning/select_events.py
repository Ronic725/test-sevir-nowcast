#!/usr/bin/env python3
"""Sample SEVIR events for quick PEFT fine-tuning."""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from custom.testing.test_actual_sevir import DATA_DIR, load_real_sevir_data_direct

DEFAULT_OUTPUT = PROJECT_ROOT / "custom" / "fine_tuning" / "cache" / "finetune_events.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache a random subset of SEVIR events.")
    parser.add_argument("--num-events", type=int, default=50, help="Number of SEVIR events to sample.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the cached NPZ file that stores inputs/targets.",
    )
    return parser.parse_args()


def main(num_events: int, seed: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_file = DATA_DIR / "sevir" / "vil" / "SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
    X, y = load_real_sevir_data_direct(str(data_file), num_events, seed)
    if X is None or y is None:
        raise SystemExit("Failed to load SEVIR samples; ensure the dataset is available.")
    np.savez_compressed(output_path, inputs=X.astype(np.float32), targets=y.astype(np.float32))
    print(f"Saved {num_events} events to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args.num_events, args.seed, args.output)
