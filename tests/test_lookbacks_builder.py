#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _proj_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _make_hourly(n: int) -> pd.DataFrame:
    ts = pd.date_range('2024-01-01', periods=n, freq='h')
    rng = np.random.default_rng(1)
    base = 100 + np.cumsum(rng.normal(0, 0.5, size=n))
    noise = rng.normal(0, 0.2, size=n)
    open_ = base
    close = base + noise
    spread = np.abs(rng.normal(0.2, 0.05, size=n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = np.abs(rng.normal(1000, 50, size=n))
    return pd.DataFrame({'timestamp': ts, 'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})


def main() -> None:
    root = _proj_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from run.lookbacks_builder import build_latest_lookbacks

    df = _make_hourly(800)
    lbs = build_latest_lookbacks(df, window_hours=720, timeframes=["1H", "4H", "12H", "1D"])
    assert set(lbs.keys()) == {"1H","4H","12H","1D"}
    # Latest index must equal last timestamp of input
    last_ts = df['timestamp'].iloc[-1]
    for tf, lb in lbs.items():
        # Last index should be the latest complete hour for 1H, or the timeframe floor otherwise
        expected_last = last_ts if tf == '1H' else pd.Timestamp(last_ts).floor(tf)
        assert lb.index[-1] == expected_last
        assert all(c in lb.columns for c in ['open','high','low','close','volume'])
    # Expected number of rows given our synthetic alignment (start at 2024-01-01 00:00)
    expected_rows = {"1H": 720, "4H": 180, "12H": 60, "1D": 30}
    for tf, exp_n in expected_rows.items():
        assert len(lbs[tf]) == exp_n, f"{tf} expected {exp_n} rows, got {len(lbs[tf])}"
    print('lookbacks_builder tests OK')


if __name__ == '__main__':
    main()

