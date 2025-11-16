#!/usr/bin/env python3
from __future__ import annotations

"""
E2E feature-building test against stored features CSVs using real lookbacks.

For a specific timestamp (default: 2024-05-25 17:00:00), load stored lookback
PKLs for 1H/4H/12H/1D, compute features via the feature builder, and compare to
the stored feature CSVs (features_1h.csv and features_4h12h1d.csv).
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


def _read_features_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        # first col as timestamp
        first = df.columns[0]
        df = df.rename(columns={first: 'timestamp'})
        ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    return df


def main() -> None:
    root = _proj_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from run.features_builder import compute_latest_features_from_lookbacks

    parser = argparse.ArgumentParser(description='E2E features recompute vs stored features CSVs and training/splits')
    parser.add_argument('--lookbacks-dir', default='/Volumes/Extreme SSD/trading_data/cex/lookbacks/BINANCE_BTCUSDT.P, 60', help='Dataset lookbacks directory containing lookbacks_*.pkl')
    parser.add_argument('--features1h', default='features_1h.csv', help='Features 1H CSV filename in lookbacks dir')
    parser.add_argument('--features_multi', default='features_4h12h1d.csv', help='Features multi-TF CSV filename in lookbacks dir')
    parser.add_argument('--training-merged-path', default='/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets.csv', help='Path to training merged features+targets CSV')
    parser.add_argument('--splits-dir', default='/Volumes/Extreme SSD/trading_data/cex/tests/inference_pipeline/splits_and_preds', help='Directory containing X_*.csv and prep_metadata.json used to train the model')
    parser.add_argument('--timeframes', nargs='+', default=['1H','4H','12H','1D'], help='Timeframes to test')
    parser.add_argument('--ts', default='2024-05-25 17:00:00', help='Timestamp to test (e.g., 2024-05-25 17:00:00)')
    parser.add_argument('--rtol', type=float, default=1e-9, help='Relative tolerance for feature comparison')
    parser.add_argument('--atol', type=float, default=1e-12, help='Absolute tolerance for feature comparison')
    args = parser.parse_args()

    dataset_dir = Path(args.lookbacks_dir)
    stores: Dict[str, Dict] = {}
    for tf in args.timeframes:
        pkl = dataset_dir / f'lookbacks_{tf}.pkl'
        if not pkl.exists():
            raise FileNotFoundError(f'Missing lookback file: {pkl}')
        stores[tf] = pd.read_pickle(pkl)

    # Parse target timestamp
    ts = pd.to_datetime(args.ts, errors='coerce', utc=True)
    if pd.isna(ts):
        raise ValueError(f'Invalid timestamp arg: {args.ts}')
    ts = ts.tz_convert('UTC').tz_localize(None)
    ts_key = _ts_key(ts)

    # Build lookbacks dict for feature builder from stored rows
    lookbacks_by_tf: Dict[str, pd.DataFrame] = {}
    for tf in args.timeframes:
        row = stores[tf].get('rows', {}).get(ts_key)
        if row is None:
            raise ValueError(f'Lookback row missing for {tf} at {ts_key}')
        lookbacks_by_tf[tf] = row

    # Compute features via builder
    feats_row = compute_latest_features_from_lookbacks(lookbacks_by_tf)

    # Load stored features CSVs and merge the row at ts
    f1h = _read_features_csv(dataset_dir / args.features1h)
    fmul = _read_features_csv(dataset_dir / args.features_multi)
    f1h_row = f1h[f1h['timestamp'] == ts]
    fmul_row = fmul[fmul['timestamp'] == ts]
    if f1h_row.empty or fmul_row.empty:
        raise AssertionError(f'No feature rows found at {ts} in one or both CSVs')
    stored_row = pd.merge(f1h_row, fmul_row, on='timestamp', how='inner', validate='one_to_one')

    # Align columns: compare intersection (excluding timestamp)
    comp_cols = sorted(set(feats_row.columns) & set(stored_row.columns) - {'timestamp'})
    assert comp_cols, 'No overlapping feature columns to compare'
    a = pd.to_numeric(feats_row[comp_cols].iloc[0], errors='coerce')
    b = pd.to_numeric(stored_row[comp_cols].iloc[0], errors='coerce')
    diffs = np.abs(a.values.astype(float) - b.values.astype(float))
    ok = np.allclose(a.values.astype(float), b.values.astype(float), rtol=float(args.rtol), atol=float(args.atol), equal_nan=True)
    print(f'{ts}: features match={ok} compared_cols={len(comp_cols)}')
    if not ok:
        order = np.argsort(diffs)[-20:][::-1]
        print('Top differing features (up to 20):')
        for k in order:
            print(f'  {comp_cols[k]} | built={a.values[k]} vs stored={b.values[k]} | absdiff={diffs[k]}')
    assert ok, 'Feature row does not match stored features within tolerance'

    # Compare with training merged_features_targets.csv (excludes targets y_*)
    merged = _read_features_csv(Path(args.training_merged_path))
    merged_row = merged[merged['timestamp'] == ts]
    assert not merged_row.empty, f'No training merged row at {ts}'
    merged_cols = [c for c in merged_row.columns if c != 'timestamp' and not str(c).startswith('y_')]
    comp_cols2 = sorted(set(feats_row.columns) & set(merged_cols))
    assert comp_cols2, 'No overlapping columns with training merged'
    a2 = pd.to_numeric(feats_row[comp_cols2].iloc[0], errors='coerce')
    b2 = pd.to_numeric(merged_row[comp_cols2].iloc[0], errors='coerce')
    diffs2 = np.abs(a2.values.astype(float) - b2.values.astype(float))
    ok2 = np.allclose(a2.values.astype(float), b2.values.astype(float), rtol=float(args.rtol), atol=float(args.atol), equal_nan=True)
    print(f'{ts}: features vs training merged match={ok2} compared_cols={len(comp_cols2)}')
    if not ok2:
        order2 = np.argsort(diffs2)[-20:][::-1]
        print('Top differing features vs merged (up to 20):')
        for k in order2:
            print(f'  {comp_cols2[k]} | built={a2.values[k]} vs merged={b2.values[k]} | absdiff={diffs2[k]}')
    assert ok2, 'Feature row does not match training merged within tolerance'

    # Compare with X_* split row via prep_metadata.json mapping
    prep_meta = Path(args.splits_dir) / 'prep_metadata.json'
    if prep_meta.exists():
        import json
        meta = json.loads(prep_meta.read_text())
        split_ts = meta.get('split_timestamps', {})
        # Find which split contains ts
        found = False
        for split_name, arr in split_ts.items():
            for i, ts_str in enumerate(arr):
                t = pd.to_datetime(ts_str, errors='coerce', utc=True)
                if pd.isna(t):
                    continue
                t = t.tz_convert('UTC').tz_localize(None)
                if t == ts:
                    x_path = Path(args.splits_dir) / f'X_{split_name}.csv'
                    X_df = pd.read_csv(x_path)
                    row = X_df.iloc[[i]].copy()
                    # Compare intersection
                    comp_cols3 = sorted(set(feats_row.columns) & set(row.columns))
                    a3 = pd.to_numeric(feats_row[comp_cols3].iloc[0], errors='coerce')
                    b3 = pd.to_numeric(row[comp_cols3].iloc[0], errors='coerce')
                    diffs3 = np.abs(a3.values.astype(float) - b3.values.astype(float))
                    ok3 = np.allclose(a3.values.astype(float), b3.values.astype(float), rtol=float(args.rtol), atol=float(args.atol), equal_nan=True)
                    print(f'{ts}: features vs X_{split_name} match={ok3} compared_cols={len(comp_cols3)}')
                    if not ok3:
                        order3 = np.argsort(diffs3)[-20:][::-1]
                        print(f'Top differing features vs X_{split_name} (up to 20):')
                        for k in order3:
                            print(f'  {comp_cols3[k]} | built={a3.values[k]} vs split={b3.values[k]} | absdiff={diffs3[k]}')
                    assert ok3, f'Feature row does not match X_{split_name} within tolerance'
                    found = True
                    break
            if found:
                break
        assert found, f'Timestamp {ts} not found in split ts list'
    else:
        print('WARNING: splits prep_metadata.json not found; skipping split comparison')
    print('E2E features test OK')


if __name__ == '__main__':
    main()


