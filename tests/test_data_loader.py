#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _proj_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _make_ohlcv(hours: int, with_partial_last: bool = False) -> pd.DataFrame:
    ts = pd.date_range('2024-01-01', periods=hours, freq='h')
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.normal(0, 0.5, size=hours))
    noise = rng.normal(0, 0.2, size=hours)
    open_ = base
    close = base + noise
    spread = np.abs(rng.normal(0.2, 0.05, size=hours))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = np.abs(rng.normal(1000, 50, size=hours))
    df = pd.DataFrame({
        'timestamp': ts,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })
    if with_partial_last:
        last_row = df.iloc[[-1]].copy()
        last_row['timestamp'] = last_row['timestamp'] + pd.Timedelta(minutes=30)
        df = pd.concat([df, last_row], ignore_index=True)
    return df


def main() -> None:
    root = _proj_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from run.data_loader import load_ohlcv_csv, latest_complete_hour, trim_to_latest_complete_hour, ensure_min_history

    # Write synthetic OHLCV with a partial last row
    tmp_dir = root / '.tmp' / 'run_tests'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_dir / 'ohlcv_synth_partial.csv'
    df_src = _make_ohlcv(50, with_partial_last=True)
    df_src.to_csv(csv_path, index=False)

    df = load_ohlcv_csv(str(csv_path))
    assert 'timestamp' in df.columns
    last_full_hour = latest_complete_hour(df)
    trimmed = trim_to_latest_complete_hour(df)
    assert trimmed['timestamp'].iloc[-1] == last_full_hour
    assert len(trimmed) == len(df_src) - 1, "Expected partial last row to be dropped"

    # History checks
    trimmed_ok, latest_ts = ensure_min_history(df, hours_required=48)
    assert len(trimmed_ok) >= 48
    assert isinstance(latest_ts, pd.Timestamp)

    failed = False
    try:
        ensure_min_history(df, hours_required=72)
    except ValueError:
        failed = True
    assert failed, "ensure_min_history should fail when insufficient coverage"
    print('data_loader tests OK')


if __name__ == '__main__':
    main()















