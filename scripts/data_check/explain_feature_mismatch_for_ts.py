#!/usr/bin/env python3
"""
Explain feature mismatches for a specific timestamp by comparing:
  - Features computed from OHLCV DuckDB (current pipeline)
  - Features computed from stored lookbacks PKLs (should match if lookbacks match)
  - Stored features in DuckDB features table (JSON) [optional]
  - Historical features CSV (features_all_tf.csv) [optional]

Outputs per-source comparison with counts of missing columns and value diffs,
and lists top differing feature names and values to aid diagnosis.

Example:
  python run/data_check/explain_feature_mismatch_for_ts.py \
    --ts "2025-08-14 21:00:00" \
    --lookbacks-dir "/Volumes/Extreme SSD/trading_data/cex/lookbacks/binance_btcusdt_perp_1h" \
    --duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb" \
    --table ohlcv_btcusdt_1h \
    --stored-features-db "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb" \
    --features-csv "/Volumes/Extreme SSD/trading_data/cex/lookbacks/binance_btcusdt_perp_1h/features_all_tf.csv" \
    --tol 1e-7 --top 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import sys as _sys
import types as _types

try:
    from run.data_loader import load_ohlcv_duckdb, validate_hourly_continuity
    from run.lookbacks_builder import build_latest_lookbacks, trim_lookbacks_to_base_window
    from run.features_builder import compute_latest_features_from_lookbacks
except ModuleNotFoundError:
    import sys as _sys
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_ROOT))
    from run.data_loader import load_ohlcv_duckdb, validate_hourly_continuity
    from run.lookbacks_builder import build_latest_lookbacks, trim_lookbacks_to_base_window
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


def _ts_key(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d_%H%M%S")


def _read_pickle_compat(path: Path):
    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError as e:
        # Handle numpy private module renames across versions by aliasing
        msg = str(e)
        if "numpy._core" in msg:
            # Create shim modules so old pickles resolve
            if 'numpy._core' not in _sys.modules:
                _sys.modules['numpy._core'] = _types.ModuleType('numpy._core')
            try:
                _sys.modules['numpy._core.numeric'] = np.core.numeric  # type: ignore[attr-defined]
            except Exception:
                pass
            return pd.read_pickle(path)
        raise


def _load_pkl_lookbacks(lookbacks_dir: Path, timeframes: List[str], ts: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    key = _ts_key(ts)
    for tf in timeframes:
        pkl = lookbacks_dir / f"lookbacks_{tf}.pkl"
        store = _read_pickle_compat(pkl)
        row = store.get("rows", {}).get(key)
        if row is None:
            raise KeyError(f"PKL row missing for {tf} at {key}")
        # Normalize index to naive UTC
        idx = pd.to_datetime(row.index, errors="coerce", utc=True)
        row = row.copy()
        row.index = idx.tz_convert("UTC").tz_localize(None)
        out[tf] = row
    return out


def _compute_from_duckdb(duckdb_path: Path, table: str, ts: pd.Timestamp, base_hours: int, buffer_hours: int, timeframes: List[str]) -> pd.DataFrame:
    required = int(base_hours + max(0, buffer_hours))
    start = ts - pd.Timedelta(hours=required - 1)
    df_all = load_ohlcv_duckdb(str(duckdb_path), table=table, start=start, end=ts)
    validate_hourly_continuity(df_all, end_ts=ts, required_hours=required)
    lookbacks = build_latest_lookbacks(df_all[df_all["timestamp"] <= ts], window_hours=required, timeframes=timeframes)
    lookbacks = trim_lookbacks_to_base_window(lookbacks, base_hours=base_hours)
    return compute_latest_features_from_lookbacks(lookbacks)


def _fetch_stored_features_row(db_path: Path, ts: pd.Timestamp) -> Dict[str, float] | None:
    try:
        import duckdb
    except Exception:
        return None
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        con.execute("SET TimeZone='UTC';")
        row = con.execute(
            """
            SELECT features FROM features
            WHERE ts = ? ORDER BY created_at DESC LIMIT 1
            """,
            [pd.Timestamp(ts).to_pydatetime()],
        ).fetchone()
        if not row:
            return None
        val = row[0]
        if isinstance(val, dict):
            return {str(k): float(v) if v is not None else float("nan") for k, v in val.items()}
        s = con.execute(
            "SELECT CAST(features AS VARCHAR) FROM features WHERE ts = ? ORDER BY created_at DESC LIMIT 1",
            [pd.Timestamp(ts).to_pydatetime()],
        ).fetchone()[0]
        return {str(k): float(v) if v is not None else float("nan") for k, v in json.loads(s).items()}
    finally:
        con.close()


def _load_csv_row(csv_path: Path, ts: pd.Timestamp) -> Dict[str, float] | None:
    if not csv_path or not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if 'timestamp' not in df.columns:
        return None
    ts_col = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['timestamp'] = ts_col.dt.tz_convert('UTC').dt.tz_localize(None)
    row = df[df['timestamp'] == ts]
    if row.empty:
        return None
    series = row.iloc[0].drop(labels=['timestamp'])
    series = pd.to_numeric(series, errors='coerce')
    return {str(k): float(series[k]) if pd.notna(series[k]) else float('nan') for k in series.index}


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
    ap = argparse.ArgumentParser(description="Explain feature mismatches for a given timestamp")
    ap.add_argument("--ts", type=str, required=True, help="Timestamp to analyze (UTC)")
    ap.add_argument("--lookbacks-dir", type=Path, required=True, help="Directory containing lookbacks_*.pkl")
    ap.add_argument("--duckdb", type=Path, required=True, help="OHLCV DuckDB path")
    ap.add_argument("--table", type=str, default=DEFAULT_TABLE, help="OHLCV table name")
    ap.add_argument("--timeframes", nargs="+", default=["1H","4H","12H","1D"], help="Timeframes to use")
    ap.add_argument("--base-hours", type=int, default=720, help="Base window hours")
    ap.add_argument("--buffer-hours", type=int, default=6, help="Buffer hours")
    ap.add_argument("--feature-list-json", type=Path, default=None, help="Filter to this feature list (defaults to production list)")
    ap.add_argument("--stored-features-db", type=Path, default=None, help="DuckDB feature store to compare (optional)")
    ap.add_argument("--features-csv", type=Path, default=None, help="Historical features CSV (optional)")
    ap.add_argument("--tol", type=float, default=1e-9, help="Numeric tolerance")
    ap.add_argument("--top", type=int, default=20, help="Max differing features to print per comparison")
    args = ap.parse_args(argv)

    ts = pd.to_datetime(args.ts, utc=True).tz_convert('UTC').tz_localize(None)
    feat_list = _load_feature_list(args.feature_list_json)

    # Compute from DuckDB
    gen_row = _compute_from_duckdb(args.duckdb, args.table, ts, args.base_hours, args.buffer_hours, list(args.timeframes))
    gen_map = {str(k): float(v) if pd.notna(v) else float('nan') for k, v in gen_row.drop(columns=['timestamp']).iloc[0].items()}

    # Compute from PKLs
    pkl_lbs = _load_pkl_lookbacks(args.lookbacks_dir, list(args.timeframes), ts)
    pkl_row = compute_latest_features_from_lookbacks(pkl_lbs)
    pkl_map = {str(k): float(v) if pd.notna(v) else float('nan') for k, v in pkl_row.drop(columns=['timestamp']).iloc[0].items()}

    def _filter(m: Dict[str, float]) -> Dict[str, float]:
        if not feat_list:
            return m
        return {k: v for k, v in m.items() if k in feat_list}

    gen_map_f = _filter(gen_map)
    pkl_map_f = _filter(pkl_map)

    # Compare gen vs pkl (should be equal if lookbacks match and functions stable)
    miss_b, miss_a, diffs = _compare_maps(gen_map_f, pkl_map_f, tol=float(args.tol))
    print(f"gen_vs_pkl: miss_in_pkl={len(miss_b)} miss_in_gen={len(miss_a)} diffs={len(diffs)}")
    for name, a, b, d in diffs[: args.top]:
        print(f"  {name}: gen={a} pkl={b} diff={d}")

    # Compare against stored DuckDB row (if provided)
    if args.stored_features_db is not None:
        stored = _fetch_stored_features_row(args.stored_features_db, ts)
        if stored is None:
            print("stored_vs_gen: no row found in stored DB at ts")
        else:
            stored_f = _filter(stored)
            miss_b2, miss_a2, diffs2 = _compare_maps(stored_f, gen_map_f, tol=float(args.tol))
            print(f"stored_vs_gen: miss_in_gen={len(miss_b2)} miss_in_stored={len(miss_a2)} diffs={len(diffs2)}")
            for name, a2, b2, d2 in diffs2[: args.top]:
                print(f"  {name}: stored={a2} gen={b2} diff={d2}")

    # Compare against CSV row (if provided)
    if args.features_csv is not None:
        csv_map = _load_csv_row(args.features_csv, ts)
        if csv_map is None:
            print("csv_vs_gen: no row found in CSV at ts")
        else:
            csv_f = _filter(csv_map)
            miss_b3, miss_a3, diffs3 = _compare_maps(csv_f, gen_map_f, tol=float(args.tol))
            print(f"csv_vs_gen: miss_in_gen={len(miss_b3)} miss_in_csv={len(miss_a3)} diffs={len(diffs3)}")
            for name, a3, b3, d3 in diffs3[: args.top]:
                print(f"  {name}: csv={a3} gen={b3} diff={d3}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
