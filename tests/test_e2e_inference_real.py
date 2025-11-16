#!/usr/bin/env python3
from __future__ import annotations

"""
E2E inference test using real BINANCE_BTCUSDT.P 1H OHLCV and a supplied model/predictions.

Requirements:
- Ignore DuckDB (file-based debug only)
- Simulate run time at different points between 2024 and 2025
- Use model under tests/inference_pipeline/model and predictions under tests/inference_pipeline/splits_and_preds
"""

import argparse
from pathlib import Path
import sys
from typing import List, Dict, Tuple
import json

import numpy as np
import pandas as pd


def _proj_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_truth_preds(truth_dir: Path, truth_file: str | None) -> pd.DataFrame:
    def _read_one(p: Path) -> pd.DataFrame:
        df = pd.read_csv(p)
        # normalize timestamp
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
        else:
            first = df.columns[0]
            df = df.rename(columns={first: 'timestamp'})
            ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
        if 'y_pred' not in df.columns:
            raise ValueError(f"Missing 'y_pred' in truth file {p}")
        return df[['timestamp', 'y_pred']]

    if truth_file:
        path = truth_dir / truth_file
        if not path.exists():
            raise FileNotFoundError(f"Truth file not found: {path}")
        truth = _read_one(path)
    else:
        # Union across splits: pred_train.csv, pred_val.csv, pred_test.csv (or any pred_*.csv)
        paths = sorted(truth_dir.glob('pred_*.csv'))
        if not paths:
            # fallback: any csv files
            paths = sorted(truth_dir.glob('*.csv'))
        if not paths:
            raise FileNotFoundError(f"No predictions CSVs found under: {truth_dir}")
        frames = []
        for p in paths:
            try:
                frames.append(_read_one(p))
            except Exception:
                continue
        if not frames:
            raise FileNotFoundError(f"No usable predictions CSVs under: {truth_dir}")
        truth = pd.concat(frames, ignore_index=True)
    truth = truth.dropna(subset=['timestamp', 'y_pred']).sort_values('timestamp')
    # If duplicates across splits, keep the last occurrence
    truth = truth.drop_duplicates(subset=['timestamp'], keep='last')
    return truth


def _select_check_timestamps(truth: pd.DataFrame, num_checks: int) -> List[pd.Timestamp]:
    # Filter to 2024-01-01..2025-12-31
    lo = pd.Timestamp('2024-01-01')
    hi = pd.Timestamp('2025-12-31 23:00:00')
    sub = truth[(truth['timestamp'] >= lo) & (truth['timestamp'] <= hi)].reset_index(drop=True)
    if sub.empty:
        raise ValueError("No truth predictions in the 2024-2025 range")
    idxs = np.linspace(0, len(sub) - 1, num=num_checks, dtype=int)
    return [pd.Timestamp(sub.loc[i, 'timestamp']) for i in idxs]


def main() -> None:
    root = _proj_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from run.data_loader import load_ohlcv_csv
    from run.lookbacks_builder import build_latest_lookbacks, trim_lookbacks_to_base_window
    from run.features_builder import compute_latest_features_from_lookbacks
    from run.model_io_lgbm import load_lgbm_model, predict_latest_row
    from model.lgbm_inference import align_features_for_booster

    parser = argparse.ArgumentParser(description='E2E inference test using real OHLCV and provided model/preds')
    parser.add_argument('--ohlcv-path', default=str(root / 'data' / 'BINANCE_BTCUSDT.P, 60.csv'), help='Path to 1H OHLCV CSV')
    parser.add_argument('--model-path', default='/Volumes/Extreme SSD/trading_data/cex/tests/inference_pipeline/model', help='Path to model run dir or model.txt')
    # Source-of-truth predictions live under the model folder (pred_train/val/test.csv)
    parser.add_argument('--truth-dir', default='/Volumes/Extreme SSD/trading_data/cex/tests/inference_pipeline/model', help='Directory containing source-of-truth predictions CSV(s)')
    parser.add_argument('--truth-file', default=None, help='Specific truth predictions CSV to use (e.g., pred_test.csv)')
    parser.add_argument('--num-checks', type=int, default=5, help='Number of timestamps to test between 2024 and 2025')
    parser.add_argument('--buffer-hours', type=int, default=6, help='Extra hours on top of 30d (720h) required history')
    parser.add_argument('--rtol', type=float, default=1e-6, help='Relative tolerance for y_pred comparison')
    parser.add_argument('--atol', type=float, default=1e-3, help='Absolute tolerance for y_pred comparison')
    parser.add_argument('--splits-dir', default='/Volumes/Extreme SSD/trading_data/cex/tests/inference_pipeline/splits_and_preds', help='Directory containing X_*.csv and prep_metadata.json used to train the model')
    args = parser.parse_args()

    # Load resources
    df_ohlcv = load_ohlcv_csv(args.ohlcv_path)
    truth = _load_truth_preds(Path(args.truth_dir), args.truth_file)
    booster, run_dir = load_lgbm_model(model_path=args.model_path)
    print('Using model from:', run_dir)

    required_hours = 720 + max(0, int(args.buffer_hours))
    checks = _select_check_timestamps(truth, args.num_checks)
    print('Selected timestamps:', checks)

    # Build timestamp -> (split, index) map from prep_metadata.json in splits-dir
    ts_to_split_index: Dict[pd.Timestamp, Tuple[str, int]] = {}
    prep_meta_path = Path(args.splits_dir) / 'prep_metadata.json'
    if prep_meta_path.exists():
        meta = json.loads(prep_meta_path.read_text())
        split_ts = meta.get('split_timestamps', {})
        for split_name, arr in split_ts.items():
            for i, ts_str in enumerate(arr):
                try:
                    ts = pd.to_datetime(ts_str, errors='coerce', utc=True)
                    ts = ts.tz_convert('UTC').tz_localize(None)
                except Exception:
                    try:
                        ts = pd.to_datetime(ts_str, errors='coerce')
                    except Exception:
                        continue
                if pd.isna(ts):
                    continue
                ts_to_split_index[pd.Timestamp(ts)] = (split_name, i)
    else:
        print('WARNING: prep_metadata.json not found; will fall back to OHLCV-derived features for all checks')

    # Lazy-loaded splits cache
    loaded_X: Dict[str, pd.DataFrame] = {}

    failures = 0
    for ts in checks:
        # 1) Original split feature row if available
        orig_feats_row = None
        if ts in ts_to_split_index:
            split_name, idx = ts_to_split_index[ts]
            x_path = Path(args.splits_dir) / f'X_{split_name}.csv'
            if x_path.exists():
                if split_name not in loaded_X:
                    loaded_X[split_name] = pd.read_csv(x_path)
                X_df = loaded_X[split_name]
                if 0 <= idx < len(X_df):
                    row = X_df.iloc[[idx]].copy()
                    row.insert(0, 'timestamp', ts)
                    orig_feats_row = row

        # 2) Recomputed features from OHLCV (production path)
        df_slice = df_ohlcv[df_ohlcv['timestamp'] <= ts].copy()
        if len(df_slice) < required_hours + 1:
            print(f"SKIP {ts}: insufficient history ({len(df_slice)} rows < {required_hours} required)")
            continue
        lbs = build_latest_lookbacks(df_slice, window_hours=required_hours, timeframes=["1H","4H","12H","1D"])
        lbs = trim_lookbacks_to_base_window(lbs, base_hours=720)
        recomp_feats_row = compute_latest_features_from_lookbacks(lbs)
        if not pd.Timestamp(recomp_feats_row['timestamp'].iloc[0]) == ts:
            print(f"WARN {ts}: recomputed feature timestamp {recomp_feats_row['timestamp'].iloc[0]} differs from target ts")

        # 3) Predict on recomputed row
        y_hat = predict_latest_row(booster, recomp_feats_row)
        y_true = float(truth.loc[truth['timestamp'] == ts, 'y_pred'].iloc[0]) if (truth['timestamp'] == ts).any() else None
        if y_true is None:
            print(f"SKIP {ts}: no matching truth y_pred")
            continue

        ok = np.allclose([y_hat], [y_true], rtol=float(args.rtol), atol=float(args.atol))
        print(f"{ts}: y_hat={y_hat:.10f}, y_true={y_true:.10f}, match={ok}")
        if not ok:
            # Optional: compare against original split row prediction if available
            if orig_feats_row is not None:
                y_hat_orig = predict_latest_row(booster, orig_feats_row)
                print(f"  DEBUG {ts}: y_hat_orig_split={y_hat_orig:.10f}")
                # Compare feature vectors fed into the model
                try:
                    aligned_orig = align_features_for_booster(orig_feats_row.drop(columns=['timestamp'], errors='ignore'), booster)
                    aligned_recomp = align_features_for_booster(recomp_feats_row.drop(columns=['timestamp'], errors='ignore'), booster)
                    v0 = aligned_orig.iloc[0].to_numpy(dtype=float)
                    v1 = aligned_recomp.iloc[0].to_numpy(dtype=float)
                    feature_names = list(booster.feature_name())
                    diffs = np.abs(v0 - v1)
                    # Top-10 differences
                    order = np.argsort(diffs)[-10:][::-1]
                    print("  DEBUG top differing features:")
                    for k in order:
                        if diffs[k] == 0 or not np.isfinite(diffs[k]):
                            continue
                        print(f"    {feature_names[k]} | split={v0[k]:.6g} vs recomp={v1[k]:.6g} | absdiff={diffs[k]:.6g}")
                    # Summary stats
                    finite = diffs[np.isfinite(diffs)]
                    if finite.size:
                        print(f"  DEBUG diff summary: max={finite.max():.6g}, mean={finite.mean():.6g}, nonzero={(finite>0).sum()}/{finite.size}")
                except Exception as e:
                    print(f"  DEBUG {ts}: failed to analyze feature diffs: {e}")
            failures += 1

    assert failures == 0, f"E2E inference mismatches: {failures} (rtol={args.rtol}, atol={args.atol})"
    print('E2E real inference test OK')


if __name__ == '__main__':
    main()


