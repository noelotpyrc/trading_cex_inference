#!/usr/bin/env python3
"""
Compare stored lookbacks PKLs against lookbacks generated from OHLCV DuckDB
for the same timestamps.

This helps isolate whether discrepancies originate at the lookback-building
stage (resampling/alignment/windowing) vs. later feature computation.

For each selected timestamp:
  - Load per-timeframe lookback from PKL store (rows[ts_key])
  - Regenerate lookbacks from OHLCV using the same window length (in 1H bars)
  - Compare shape, index alignment, and OHLCV columns with tolerance

Example:
  python run/data_check/compare_lookbacks_pkl_vs_duckdb.py \
    --lookbacks-dir "/Volumes/Extreme SSD/trading_data/cex/lookbacks/binance_btcusdt_perp_1h" \
    --duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb" \
    --table ohlcv_btcusdt_1h --limit 3 --timeframes 1H 4H 12H 1D --tol 1e-9
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import sys as _sys
import types as _types

try:
    from run.data_loader import load_ohlcv_duckdb, validate_hourly_continuity
    from run.lookbacks_builder import build_latest_lookbacks
except ModuleNotFoundError:
    # Allow running as a script: add repo root to sys.path
    import sys as _sys
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_ROOT))
    from run.data_loader import load_ohlcv_duckdb, validate_hourly_continuity
    from run.lookbacks_builder import build_latest_lookbacks


OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _read_store(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Lookback store not found: {path}")
    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError as e:
        # Handle numpy private module renames across versions by aliasing
        msg = str(e)
        if "numpy._core" in msg:
            try:
                # Create a shim so pickles referencing 'numpy._core.numeric' resolve
                if 'numpy._core' not in _sys.modules:
                    _sys.modules['numpy._core'] = _types.ModuleType('numpy._core')
                # Map common submodules
                try:
                    _sys.modules['numpy._core.numeric'] = np.core.numeric  # type: ignore[attr-defined]
                except Exception:
                    pass
                # Retry after aliasing
                return pd.read_pickle(path)
            except Exception:
                raise
        raise


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out.index, errors="coerce", utc=True)
    out.index = idx.tz_convert("UTC").tz_localize(None)
    return out


def _select_timestamps(store_1h: Dict, limit: int, explicit_ts: List[str] | None) -> List[pd.Timestamp]:
    base_index = store_1h.get("base_index")
    if base_index is None or len(base_index) == 0:
        raise ValueError("1H store missing 'base_index' or it is empty")
    ts_idx = pd.to_datetime(pd.Index(base_index), errors="coerce", utc=True)
    ts_idx = ts_idx.tz_convert("UTC").tz_localize(None)
    if explicit_ts:
        want = [pd.to_datetime(t, utc=True).tz_convert("UTC").tz_localize(None) for t in explicit_ts]
        selected = [t for t in want if t in set(ts_idx)]
        return selected[:limit]
    # Default: most recent N
    return list(ts_idx[-limit:])


def _ts_key(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d_%H%M%S")


def _compare_frames(a: pd.DataFrame, b: pd.DataFrame, tol: float) -> Tuple[bool, Dict[str, int], Dict[str, float]]:
    ok = True
    # Ensure same columns subset
    cols = [c for c in OHLCV_COLS if c in a.columns and c in b.columns]
    stats_counts: Dict[str, int] = {}
    stats_maxdiff: Dict[str, float] = {}
    # Index alignment check
    if len(a) != len(b) or not a.index.equals(b.index):
        ok = False
        # Align for diff reporting
        common = a.index.intersection(b.index)
        a = a.loc[common]
        b = b.loc[common]
    for c in cols:
        av = pd.to_numeric(a[c], errors="coerce").astype(float)
        bv = pd.to_numeric(b[c], errors="coerce").astype(float)
        diff = (av - bv).abs()
        mism = int((~(np.isclose(av, bv, atol=tol, rtol=0, equal_nan=True))).sum())
        stats_counts[c] = mism
        stats_maxdiff[c] = float(diff.max() if len(diff) else 0.0)
        if mism > 0:
            ok = False
    return ok, stats_counts, stats_maxdiff


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compare lookbacks PKL vs OHLCV-generated lookbacks")
    ap.add_argument("--lookbacks-dir", type=Path, required=True, help="Directory containing lookbacks_*.pkl")
    ap.add_argument("--duckdb", type=Path, required=True, help="OHLCV DuckDB path")
    ap.add_argument("--table", type=str, default="ohlcv_btcusdt_1h", help="OHLCV table name")
    ap.add_argument("--timeframes", nargs="+", default=["1H", "4H", "12H", "1D"], help="Timeframes to compare")
    ap.add_argument("--limit", type=int, default=3, help="Number of recent timestamps to test (ignored if --ts given)")
    ap.add_argument("--ts", nargs="*", default=[], help="Explicit timestamps to test (UTC)")
    ap.add_argument("--tol", type=float, default=1e-9, help="Tolerance for numeric equality")
    args = ap.parse_args(argv)

    # Load stores per timeframe
    stores: Dict[str, Dict] = {}
    for tf in args.timeframes:
        pkl = Path(args.lookbacks_dir) / f"lookbacks_{tf}.pkl"
        stores[tf] = _read_store(pkl)

    # Select timestamps from 1H store base_index
    ts_list = _select_timestamps(stores[args.timeframes[0]], int(args.limit), list(args.ts or []))
    if not ts_list:
        print("No timestamps selected; exiting")
        return 0

    # Window length (in base 1H bars) used by PKL building
    lookback_base_rows = int(stores[args.timeframes[0]].get("lookback_base_rows") or 168)
    print(f"Using window_hours={lookback_base_rows} (from 1H store lookback_base_rows)")

    all_ok = True
    for ts in ts_list:
        ts_key = _ts_key(ts)
        # Load minimal OHLCV slice
        earliest_needed = ts - pd.Timedelta(hours=lookback_base_rows - 1)
        df_all = load_ohlcv_duckdb(str(args.duckdb), table=str(args.table), start=earliest_needed, end=ts)
        try:
            validate_hourly_continuity(df_all, end_ts=ts, required_hours=lookback_base_rows)
        except Exception as e:
            print(f"{ts}: SKIP (continuity failed: {e})")
            all_ok = False
            continue
        lb_gen = build_latest_lookbacks(df_all[df_all["timestamp"] <= ts], window_hours=lookback_base_rows, timeframes=args.timeframes)

        print(f"\nTimestamp: {ts} ({ts_key})")
        tf_all_ok = True
        for tf in args.timeframes:
            lb_pkl = stores[tf].get("rows", {}).get(ts_key)
            if lb_pkl is None:
                print(f"  {tf}: MISSING in PKL for {ts_key}")
                tf_all_ok = False
                continue
            a = _normalize_index(lb_pkl)
            b = _normalize_index(lb_gen[tf])
            # Restrict to expected columns
            a = a[[c for c in OHLCV_COLS if c in a.columns]]
            b = b[[c for c in OHLCV_COLS if c in b.columns]]
            ok, counts, maxdiff = _compare_frames(a, b, tol=float(args.tol))
            status = "OK" if ok else "DIFF"
            print(f"  {tf}: {status} | len(pkl)={len(a)} len(gen)={len(b)}")
            if not ok:
                print(f"    mismatches per col: {counts}")
                print(f"    max abs diff per col: {maxdiff}")
            tf_all_ok = tf_all_ok and ok
        all_ok = all_ok and tf_all_ok

    print(f"\nSummary: all_ok={all_ok}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
