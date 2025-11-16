"""
Merge two feature CSVs (current_bar_with_lags and features_all_tf_with_norm) after a cutoff
timestamp, then score with a LightGBM model and write predictions to the model folder.

Defaults match your disk layout. You can override paths via CLI args.

Example:
  python run/merge_and_score_current_norm.py \
    --lookbacks-dir "/Volumes/Extreme SSD/trading_data/cex/lookbacks/binance_btcusdt_perp_1h" \
    --cutoff "2025-03-21 04:00:00" \
    --model-path "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60/diagnosis/y_logret_24h_huber_all" \
    --dataset "binance_btcusdt_perp_1h"
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.lgbm_inference import predict_from_csv


def _find_file(dir_path: Path, pattern: str) -> Path:
    candidates = sorted(glob.glob(str(dir_path / pattern)))
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern} under {dir_path}")
    # Pick the latest modified
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return Path(candidates[-1])


def _load_csv_with_ts(path: Path, ts_col_candidates=("timestamp", "time", "datetime", "date")) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Identify timestamp column or fallback to first
    ts_col: Optional[str] = None
    for c in ts_col_candidates:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        ts_col = df.columns[0]
    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    if hasattr(ts, "dt"):
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        ts = ts.tz_convert("UTC").tz_localize(None)
    df = df.drop(columns=[ts_col])
    df.index = pd.DatetimeIndex(ts, name="timestamp")
    df = df.sort_index()
    return df


def _merge_features(
    cur: pd.DataFrame,
    norm: pd.DataFrame,
    *,
    prefer: str = "cur",
) -> pd.DataFrame:
    # Drop overlapping columns from the second according to preference to avoid suffixes
    overlap = [c for c in cur.columns if c in norm.columns]
    if overlap:
        if prefer == "cur":
            norm = norm.drop(columns=overlap, errors="ignore")
        else:
            cur = cur.drop(columns=overlap, errors="ignore")
    merged = cur.join(norm, how="inner")
    merged = merged.reset_index()
    return merged


def _default_cur_pattern() -> str:
    return "current_bar_with_lags_1H-4H-12H-1D_lags14.csv"


def _default_norm_pattern() -> str:
    # Be flexible: users sometimes have suffixes; prefer exact name first else wildcard
    return "features_all_tf_with_norm*.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge current-bar-with-lags and normalized features, then score with LGBM model")
    ap.add_argument("--lookbacks-dir", default="/Volumes/Extreme SSD/trading_data/cex/lookbacks/binance_btcusdt_perp_1h")
    ap.add_argument("--current-file", default=None, help="Explicit path to current_bar_with_lags CSV")
    ap.add_argument("--norm-file", default=None, help="Explicit path to features_all_tf_with_norm CSV")
    ap.add_argument("--cutoff", required=True, help="Only include rows on/after this timestamp (YYYY-mm-dd HH:MM:SS)")
    ap.add_argument("--until", default=None, help="Optional upper bound (inclusive)")
    ap.add_argument("--prefer", choices=["cur", "norm"], default="cur", help="When columns overlap, which side to keep")
    ap.add_argument("--dataset", default="binance_btcusdt_perp_1h", help="Dataset label for logging/metadata")
    ap.add_argument("--model-path", default="/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60/diagnosis/y_logret_24h_huber_all")
    ap.add_argument("--out-merged", default=None, help="Optional path to save merged features CSV (for inspection)")
    ap.add_argument("--out-preds", default=None, help="Optional path to save predictions CSV (default: inside model-path)")
    args = ap.parse_args()

    base = Path(args.lookbacks_dir)
    cur_path = Path(args.current_file) if args.current_file else _find_file(base, _default_cur_pattern())
    try:
        norm_path = Path(args.norm_file) if args.norm_file else _find_file(base, _default_norm_pattern())
    except FileNotFoundError:
        # Fallback to a common alternative name
        norm_path = _find_file(base, "features_all_tf.csv")

    print("Current-bar with lags:", cur_path)
    print("Normalized features:", norm_path)

    cur_df = _load_csv_with_ts(cur_path)
    norm_df = _load_csv_with_ts(norm_path)

    cutoff_ts = pd.to_datetime(args.cutoff)
    if args.until:
        until_ts = pd.to_datetime(args.until)
        cur_df = cur_df[(cur_df.index >= cutoff_ts) & (cur_df.index <= until_ts)]
        norm_df = norm_df[(norm_df.index >= cutoff_ts) & (norm_df.index <= until_ts)]
    else:
        cur_df = cur_df[cur_df.index >= cutoff_ts]
        norm_df = norm_df[norm_df.index >= cutoff_ts]

    # Align to intersection timestamps to avoid NaN rows when scoring
    inter = cur_df.index.intersection(norm_df.index)
    if len(inter) == 0:
        raise SystemExit("No overlapping timestamps after cutoff (and until if provided)")
    cur_df = cur_df.loc[inter]
    norm_df = norm_df.loc[inter]

    merged = _merge_features(cur_df, norm_df, prefer=args.prefer)

    # Save merged features if requested
    if args.out_merged:
        out_m = Path(args.out_merged)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_m, index=False)
        print("Merged features saved:", out_m)

    # Predict using model.lgbm_inference (aligns + drops extras automatically)
    model_dir = Path(args.model_path)
    if model_dir.is_dir():
        out_preds = Path(args.out_preds) if args.out_preds else model_dir / "predictions_from_merged_current_norm.csv"
        res = predict_from_csv(
            input_csv=str(Path(args.out_merged) if args.out_merged else _temp_write_csv_beside(cur_path, merged)),
            model_root=None,
            model_path=str(model_dir),
            output_csv=str(out_preds),
            timestamp_column="timestamp",
            merge_input=False,
        )
        print(f"Predictions written: {out_preds} (rows={len(res)})")
    else:
        # If a model.txt path was given
        out_preds = Path(args.out_preds) if args.out_preds else Path(args.model_path).parent / "predictions_from_merged_current_norm.csv"
        res = predict_from_csv(
            input_csv=str(Path(args.out_merged) if args.out_merged else _temp_write_csv_beside(cur_path, merged)),
            model_root=None,
            model_path=str(model_dir),
            output_csv=str(out_preds),
            timestamp_column="timestamp",
            merge_input=False,
        )
        print(f"Predictions written: {out_preds} (rows={len(res)})")


def _temp_write_csv_beside(anchor_path: Path, df: pd.DataFrame) -> Path:
    # Write next to anchor with a temp-ish name
    temp_path = anchor_path.with_name("merged_current_norm_tmp.csv")
    df.to_csv(temp_path, index=False)
    return temp_path


if __name__ == "__main__":
    main()

