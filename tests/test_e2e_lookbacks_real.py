#!/usr/bin/env python3
from __future__ import annotations

"""
E2E lookback-building test against stored lookback PKLs.

For selected timestamps between 2024 and 2025, recompute lookbacks from 1H OHLCV
using the production path and compare each timeframe's lookback DataFrame to the
stored lookback PKLs under the lookbacks directory.
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd


def _proj_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ts_key(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime('%Y%m%d_%H%M%S')


def _compare_frames(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if a is None or b is None:
        return (a is None) and (b is None)
    if a.empty and b.empty:
        return True
    if a.shape != b.shape:
        return False
    if list(a.columns) != list(b.columns):
        return False
    # Align index types
    try:
        a_index = pd.to_datetime(a.index, errors='coerce', utc=True)
        a_index = a_index.tz_convert('UTC').tz_localize(None)
    except Exception:
        a_index = a.index
    try:
        b_index = pd.to_datetime(b.index, errors='coerce', utc=True)
        b_index = b_index.tz_convert('UTC').tz_localize(None)
    except Exception:
        b_index = b.index
    if not pd.Index(a_index).equals(pd.Index(b_index)):
        return False
    # Compare values with normalized index to avoid tz metadata differences
    a_norm = a.copy()
    b_norm = b.copy()
    try:
        a_norm.index = a_index
        b_norm.index = b_index
    except Exception:
        pass
    return a_norm.equals(b_norm)


def main() -> None:
    root = _proj_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from run.data_loader import load_ohlcv_csv
    from run.lookbacks_builder import build_latest_lookbacks

    parser = argparse.ArgumentParser(description='E2E lookback recompute vs stored PKLs')
    parser.add_argument('--ohlcv-path', default=str(root / 'data' / 'BINANCE_BTCUSDT.P, 60.csv'), help='Path to 1H OHLCV CSV')
    parser.add_argument('--lookbacks-dir', default='/Volumes/Extreme SSD/trading_data/cex/lookbacks/BINANCE_BTCUSDT.P, 60', help='Dataset lookbacks directory containing lookbacks_*.pkl')
    parser.add_argument('--timeframes', nargs='+', default=['1H','4H','12H','1D'], help='Timeframes to test')
    parser.add_argument('--num-checks', type=int, default=5, help='Number of timestamps to test between 2024 and 2025')
    parser.add_argument('--ts', nargs='*', default=None, help='Explicit timestamp(s) to check (e.g., 2024-05-25 17:00:00)')
    args = parser.parse_args()

    # Load OHLCV and lookback stores
    df = load_ohlcv_csv(args.ohlcv_path)
    stores: Dict[str, Dict] = {}
    for tf in args.timeframes:
        pkl = Path(args.lookbacks_dir) / f'lookbacks_{tf}.pkl'
        if not pkl.exists():
            raise FileNotFoundError(f'Missing lookback file: {pkl}')
        stores[tf] = pd.read_pickle(pkl)

    # Determine required base window in hours from 1H store
    base_rows = int(stores['1H'].get('lookback_base_rows', 720))

    # Build candidate timestamps from 1H store base_index
    base_index = stores['1H'].get('base_index')
    if base_index is None or len(base_index) == 0:
        raise ValueError('1H store missing base_index or it is empty')
    # Normalize to UTC-naive for consistent comparisons
    ts_series = pd.to_datetime(pd.Series(base_index), errors='coerce', utc=True)
    # If conversion failed to DatetimeIndex (object), coerce via Series.dt
    if hasattr(ts_series, 'dt'):
        ts_series = ts_series.dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        ts_series = pd.to_datetime(ts_series, errors='coerce').tz_localize(None)

    if args.ts:
        explicit_parsed = [pd.to_datetime(t, errors='coerce', utc=True) for t in args.ts]
        explicit = []
        for t in explicit_parsed:
            if pd.isna(t):
                continue
            try:
                explicit.append(t.tz_convert('UTC').tz_localize(None))
            except Exception:
                explicit.append(pd.to_datetime(t, errors='coerce').tz_localize(None))
        test_timestamps: List[pd.Timestamp] = [pd.Timestamp(t) for t in explicit]
    else:
        # filter to 2024..2025 inclusive
        mask = (ts_series >= pd.Timestamp('2024-01-01')) & (ts_series <= pd.Timestamp('2025-12-31 23:00:00'))
        ts_candidates = ts_series[mask].reset_index(drop=True)
        if ts_candidates.empty:
            raise ValueError('No timestamps in 2024-2025 range in 1H store base_index')
        sel_idx = np.linspace(0, len(ts_candidates) - 1, num=args.num_checks, dtype=int)
        test_timestamps = [pd.Timestamp(ts_candidates.iloc[i]) for i in sel_idx]
    print('Selected timestamps:', test_timestamps)

    failures = 0
    for ts in test_timestamps:
        # Recompute lookbacks up to ts
        df_slice = df[df['timestamp'] <= ts].copy()
        if len(df_slice) < base_rows:
            print(f'SKIP {ts}: insufficient history ({len(df_slice)} < base_rows={base_rows})')
            continue
        recomputed = build_latest_lookbacks(df_slice, window_hours=base_rows, timeframes=args.timeframes)

        # Compare per timeframe with stored
        ts_key = _ts_key(ts)
        for tf in args.timeframes:
            store = stores[tf]
            rows_map = store.get('rows', {})
            stored_df: pd.DataFrame | None = rows_map.get(ts_key)
            recomputed_df: pd.DataFrame | None = recomputed.get(tf)
            ok = _compare_frames(stored_df, recomputed_df)
            print(f'{ts} [{tf}]: match={ok} stored_rows={0 if stored_df is None else len(stored_df)} vs recomp_rows={0 if recomputed_df is None else len(recomputed_df)}')
            if not ok:
                failures += 1
                # Log concise diffs
                if stored_df is None or recomputed_df is None:
                    print('  One of the DataFrames is None')
                    continue
                if stored_df.shape != recomputed_df.shape:
                    print(f'  Shape mismatch: stored={stored_df.shape} vs recomputed={recomputed_df.shape}')
                if list(stored_df.columns) != list(recomputed_df.columns):
                    only_store = [c for c in stored_df.columns if c not in recomputed_df.columns]
                    only_recomp = [c for c in recomputed_df.columns if c not in stored_df.columns]
                    print(f'  Columns differ. Only in store: {only_store[:10]} Only in recomputed: {only_recomp[:10]}')
                # Index diff
                try:
                    idx_store = pd.to_datetime(stored_df.index, errors='coerce', utc=True).tz_convert('UTC').tz_localize(None)
                    idx_recomp = pd.to_datetime(recomputed_df.index, errors='coerce', utc=True).tz_convert('UTC').tz_localize(None)
                    if not idx_store.equals(idx_recomp):
                        print('  Index differs:')
                        print(f'    store head: {list(idx_store[:3])} tail: {list(idx_store[-3:])}')
                        print(f'    reomp head: {list(idx_recomp[:3])} tail: {list(idx_recomp[-3:])}')
                except Exception:
                    pass
                # First numeric mismatch
                try:
                    aligned = stored_df.reset_index(drop=True).compare(recomputed_df.reset_index(drop=True))
                    print('  Example diffs (first 3 rows):')
                    print(aligned.head(3))
                except Exception:
                    pass

    assert failures == 0, f'Lookback recompute mismatches: {failures}'
    print('E2E lookback recompute test OK')


if __name__ == '__main__':
    main()


