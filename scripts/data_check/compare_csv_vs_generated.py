#!/usr/bin/env python3
"""
Compare features from a historical CSV (e.g., features_all_tf.csv) with
features generated on-the-fly from OHLCV via the current backfill pipeline.

Use this to diagnose discrepancies between older feature builds and the
current computation stack (lookback building, resampling, feature functions).

By default, it filters both sides to the default production feature list
(configs/feature_lists/binance_btcusdt_p60_default.json) and compares values
for selected timestamps within a tolerance.

Examples:
  python run/data_check/compare_csv_vs_generated.py \
    --features-csv "/Volumes/Extreme SSD/trading_data/cex/lookbacks/binance_btcusdt_perp_1h/features_all_tf.csv" \
    --duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb" \
    --table ohlcv_btcusdt_1h --limit 5 --tol 1e-7

  python run/data_check/compare_csv_vs_generated.py \
    --features-csv /path/features_all_tf.csv --duckdb /path/ohlcv.duckdb \
    --start "2025-08-01 00:00:00" --end "2025-08-07 23:00:00"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from run.data_loader import load_ohlcv_duckdb, validate_hourly_continuity
from run.lookbacks_builder import build_latest_lookbacks, trim_lookbacks_to_base_window, validate_lookbacks_exact
from run.features_builder import compute_latest_features_from_lookbacks


DEFAULT_TABLE = "ohlcv_btcusdt_1h"
DEFAULT_FEATURE_LIST = Path(__file__).resolve().parents[2] / "configs/feature_lists/binance_btcusdt_p60_default.json"


def _load_feature_list(path: Optional[Path]) -> List[str]:
    p = path or DEFAULT_FEATURE_LIST
    try:
        if p and p.exists():
            raw = json.loads(Path(p).read_text())
            if isinstance(raw, dict) and "features" in raw:
                seq = raw["features"]
            else:
                seq = raw
            return [str(x) for x in seq] if isinstance(seq, list) else []
    except Exception:
        pass
    return []


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'timestamp' not in df.columns:
        raise ValueError("CSV must include 'timestamp' column")
    ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    return df


def _select_timestamps(df_csv: pd.DataFrame, start: Optional[str], end: Optional[str], limit: Optional[int], ts_list: List[str]) -> List[pd.Timestamp]:
    if ts_list:
        parsed = [pd.to_datetime(t, utc=True).tz_convert('UTC').tz_localize(None) for t in ts_list]
        ts = [t for t in parsed if t in set(df_csv['timestamp'])]
        return sorted(ts)[: (limit or len(ts))]
    if start or end:
        lo = pd.to_datetime(start, utc=True).tz_convert('UTC').tz_localize(None) if start else df_csv['timestamp'].min()
        hi = pd.to_datetime(end, utc=True).tz_convert('UTC').tz_localize(None) if end else df_csv['timestamp'].max()
        sub = df_csv[(df_csv['timestamp'] >= lo) & (df_csv['timestamp'] <= hi)]
        return sub['timestamp'].head(limit or len(sub)).tolist()
    # Default: take the most recent N
    return df_csv['timestamp'].tail(limit or 5).tolist()


def _compute_features_for_ts(
    duckdb_path: Path,
    table: str,
    ts: pd.Timestamp,
    *,
    base_hours: int,
    buffer_hours: int,
    timeframes: List[str],
) -> pd.DataFrame:
    required_hours = int(base_hours + max(0, buffer_hours))
    earliest_needed = ts - pd.Timedelta(hours=required_hours - 1)
    df_all = load_ohlcv_duckdb(str(duckdb_path), table=table, start=earliest_needed, end=ts)
    validate_hourly_continuity(df_all, end_ts=ts, required_hours=required_hours)
    df_slice = df_all[df_all['timestamp'] <= ts].copy()
    lookbacks = build_latest_lookbacks(df_slice, window_hours=required_hours, timeframes=timeframes)
    lookbacks = trim_lookbacks_to_base_window(lookbacks, base_hours=base_hours)
    validate_lookbacks_exact(lookbacks, base_hours=base_hours, end_ts=ts)
    return compute_latest_features_from_lookbacks(lookbacks)


def _series_from_row(frame: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    # frame has 'timestamp' + features
    cols = [c for c in feature_cols if c in frame.columns]
    return frame.iloc[0][cols]


def _compare_maps(a: Dict[str, float], b: Dict[str, float], tol: float) -> Tuple[List[str], List[str], List[Tuple[str, float, float, float]]]:
    ka, kb = set(a.keys()), set(b.keys())
    miss_b = sorted(list(ka - kb))
    miss_a = sorted(list(kb - ka))
    diffs: List[Tuple[str, float, float, float]] = []
    for k in sorted(ka & kb):
        va, vb = float(a[k]), float(b[k])
        if np.isnan(va) and np.isnan(vb):
            continue
        if np.isnan(va) or np.isnan(vb):
            diffs.append((k, va, vb, np.nan))
            continue
        d = abs(va - vb)
        if d > tol:
            diffs.append((k, va, vb, d))
    return miss_b, miss_a, diffs


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compare features_all_tf.csv vs generated features from OHLCV")
    ap.add_argument("--features-csv", type=Path, required=True, help="Path to historical features CSV (features_all_tf.csv)")
    ap.add_argument("--duckdb", type=Path, required=True, help="OHLCV DuckDB path")
    ap.add_argument("--table", type=str, default=DEFAULT_TABLE, help="OHLCV table name")
    ap.add_argument("--start", type=str, default=None, help="Start timestamp (inclusive)")
    ap.add_argument("--end", type=str, default=None, help="End timestamp (inclusive)")
    ap.add_argument("--ts", nargs="*", default=[], help="Explicit timestamps to compare")
    ap.add_argument("--limit", type=int, default=5, help="Number of timestamps to compare if no --ts given")
    ap.add_argument("--base-hours", type=int, default=30 * 24, help="Base window hours (default 720)")
    ap.add_argument("--buffer-hours", type=int, default=6, help="Extra hours on top of base window")
    ap.add_argument("--timeframes", nargs="+", default=["1H", "4H", "12H", "1D"], help="Timeframes used for lookbacks")
    ap.add_argument("--feature-list-json", type=Path, default=None, help="Feature list JSON to filter both sides (defaults to production list)")
    ap.add_argument("--tol", type=float, default=1e-9, help="Absolute tolerance for value comparison")
    args = ap.parse_args(argv)

    df_csv = _load_csv(args.features_csv)
    feature_list = _load_feature_list(args.feature_list_json)
    targets = _select_timestamps(df_csv, args.start, args.end, args.limit, list(args.ts or []))
    if not targets:
        print("No timestamps selected from CSV; nothing to compare.")
        return 0

    total = 0
    ok = 0
    for ts in targets:
        total += 1
        try:
            gen_row = _compute_features_for_ts(
                args.duckdb, args.table, ts,
                base_hours=int(args.base_hours), buffer_hours=int(args.buffer_hours), timeframes=list(args.timeframes)
            )
        except Exception as e:
            print(f"{ts}: ERROR generating features ({e})")
            continue

        # Filter both sides to the feature list intersection if provided
        if feature_list:
            present = [c for c in feature_list if c in gen_row.columns]
            csv_cols = [c for c in feature_list if c in df_csv.columns]
        else:
            # Use intersection of CSV and generated columns
            present = [c for c in gen_row.columns if c != 'timestamp' and c in df_csv.columns]
            csv_cols = present

        # Extract CSV row for ts
        match = df_csv[df_csv['timestamp'] == ts]
        if match.empty:
            print(f"{ts}: SKIP (timestamp not present in CSV)")
            continue
        csv_series = match.iloc[0][csv_cols].apply(pd.to_numeric, errors='coerce')
        gen_series = _series_from_row(gen_row, present).apply(pd.to_numeric, errors='coerce')

        a = {k: float(csv_series.get(k, np.nan)) for k in csv_cols}
        b = {k: float(gen_series.get(k, np.nan)) for k in present}
        miss_b, miss_a, diffs = _compare_maps(a, b, tol=float(args.tol))
        if not miss_b and not miss_a and not diffs:
            print(f"{ts}: OK (match within tol={args.tol}) cols={len(present)}")
            ok += 1
        else:
            print(f"{ts}: MISMATCH cols={len(present)} miss_in_generated={len(miss_b)} miss_in_csv={len(miss_a)} diffs={len(diffs)}")
            if miss_b:
                print(f"  Missing in generated (sample): {miss_b[:10]}")
            if miss_a:
                print(f"  Missing in CSV (sample): {miss_a[:10]}")
            if diffs:
                for name, va, vb, d in diffs[:10]:
                    print(f"  {name}: csv={va} gen={vb} diff={d}")

    print(f"\nSummary: compared={total} ok={ok} failed={total-ok}")
    return 0 if ok == total else 2


if __name__ == "__main__":
    raise SystemExit(main())

