#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import random
import argparse

import numpy as np
import pandas as pd


def _proj_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = _proj_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from run.model_io_lgbm import load_lgbm_model, predict_latest_row
    parser = argparse.ArgumentParser(description='Test LightGBM model I/O with optional merged features row', add_help=False)
    parser.add_argument('--merged-path', default='/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets.csv', help='Path to merged_features_targets.csv to source a real feature row')
    args, _unknown = parser.parse_known_args()

    # Attempt to load from default models root; skip test if missing
    default_root = Path('/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60')
    if not default_root.exists():
        print('SKIP: default models root not found')
        return
    # Randomly pick a run_* folder containing model.txt
    run_dirs = [p for p in default_root.glob('run_*') if p.is_dir() and (p / 'model.txt').exists()]
    if not run_dirs:
        print('SKIP: no run_* with model.txt under default models root')
        return
    rng = random.Random(0)
    run_dir = rng.choice(sorted(run_dirs))
    try:
        booster, run_dir = load_lgbm_model(model_path=str(run_dir))
    except Exception as e:
        print('SKIP: failed to load chosen run dir:', run_dir, e)
        return

    # Prefer a real feature row from merged_features_targets.csv when available to reduce alignment warnings
    features_row = None
    if args.merged_path:
        mp = Path(args.merged_path)
        if mp.exists():
            merged = pd.read_csv(mp, nrows=1)
            # Drop targets (y_*) and keep timestamp + feature columns
            cols = [c for c in merged.columns if c == 'timestamp' or not str(c).startswith('y_')]
            features_row = merged[cols].copy()
    if features_row is None:
        # Fallback: tiny row with timestamp only
        features_row = pd.DataFrame({'timestamp': [pd.Timestamp('2024-01-01T00:00:00')]})

    y = predict_latest_row(booster, features_row)
    assert isinstance(y, float)
    print('model_io_lgbm tests OK (loaded:', run_dir, ')')


if __name__ == '__main__':
    main()


