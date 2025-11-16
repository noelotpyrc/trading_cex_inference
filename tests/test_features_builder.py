#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd


def _proj_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _make_lb(n: int) -> pd.DataFrame:
    ts = pd.date_range('2024-01-01', periods=n, freq='h')
    rng = np.random.default_rng(2)
    base = 100 + np.cumsum(rng.normal(0, 0.5, size=n))
    noise = rng.normal(0, 0.2, size=n)
    open_ = base
    close = base + noise
    spread = np.abs(rng.normal(0.2, 0.05, size=n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = np.abs(rng.normal(1000, 50, size=n))
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=ts)


def main() -> None:
    root = _proj_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from run.features_builder import compute_latest_features_from_lookbacks
    # Optional CLI param to provide a specific merged_features_targets.csv path
    parser = argparse.ArgumentParser(description='Test features_builder parity with merged features', add_help=False)
    parser.add_argument('--merged-path', default='/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets.csv', help='Path to merged_features_targets.csv to validate against')
    args, _unknown = parser.parse_known_args()

    lookbacks = {
        '1H': _make_lb(720),
        '4H': _make_lb(180),
        '12H': _make_lb(60),
        '1D': _make_lb(30),
    }
    feats = compute_latest_features_from_lookbacks(lookbacks)
    assert 'timestamp' in feats.columns
    assert len(feats) == 1
    # Spot-check a few expected families
    expected_substrings = ['close_sma_', 'close_rsi_', 'close_macd', 'volume_ma_']
    joined_cols = ','.join(feats.columns)
    assert any(s in joined_cols for s in expected_substrings)
    # Print all feature names (excluding timestamp)
    feature_names = [c for c in feats.columns if c != 'timestamp']
    print(f"Feature columns ({len(feature_names)}):")
    for name in sorted(feature_names):
        print(name)

    # Cross-check against an existing merged features file (if available)
    # Do not hardcode exact CSV path: search training dir for a merged_features_targets.csv
    training_base = Path('/Volumes/Extreme SSD/trading_data/cex/training')
    merged_path = None
    if args.merged_path:
        cand = Path(args.merged_path)
        if cand.exists():
            merged_path = cand
        else:
            print(f"WARN: provided --merged-path does not exist: {cand}")
    if merged_path is None and training_base.exists():
        candidates = sorted(training_base.glob('*/merged_features_targets.csv'))
        if not candidates:
            candidates = sorted(training_base.rglob('merged_features_targets.csv'))
        preferred = [p for p in candidates if p.parent.name == 'BINANCE_BTCUSDT.P, 60']
        if preferred:
            merged_path = preferred[0]
        elif candidates:
            merged_path = candidates[0]

    if merged_path is not None and merged_path.exists():
        merged_head = pd.read_csv(merged_path, nrows=1)
        merged_cols = [c for c in merged_head.columns if c != 'timestamp' and not str(c).startswith('y_')]
        missing = [c for c in merged_cols if c not in feats.columns]
        print(f"Merged feature columns found: {len(merged_cols)} from {merged_path}")
        if missing:
            # Print a short diagnostic but still assert to enforce parity
            print(f"Missing {len(missing)} columns from computed features (showing up to 30): {missing[:30]}")
        assert not missing, "Computed features must include all columns present in merged_features_targets.csv (excluding timestamp/targets)"
    else:
        print('SKIP: No merged_features_targets.csv found under training directory, parity check skipped.')
    print('features_builder tests OK')


if __name__ == '__main__':
    main()

