#!/usr/bin/env python3
"""
E2E test: generate features for a recent timestamp and compare against
stored features in a reference DuckDB features database.

Steps:
  1) Determine a target timestamp and feature_key from the reference DB
     (use the most recent ts for the chosen key by default).
  2) Run feature backfill for that ts using OHLCV DuckDB into a temp features DB.
  3) Read both feature JSON maps and compare numerically with tolerance.

Exit codes:
  0: PASS (no significant diffs)
  2: FAIL (missing keys or numeric diffs exceed tolerance)
  3: ERROR (unexpected exception)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb  # type: ignore
import pandas as pd

try:
    from run.backfill_features import BackfillFeaturesConfig, backfill_features
except ModuleNotFoundError:
    # Allow running as `python run/e2e_backfill_features_compare.py` by adding repo root to sys.path
    import sys as _sys
    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in _sys.path:
        _sys.path.insert(0, str(_repo_root))
    from run.backfill_features import BackfillFeaturesConfig, backfill_features


DEFAULT_OHLCV_DB = "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb"
DEFAULT_STORED_FEATURES_DB = "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb"
DEFAULT_TABLE = "ohlcv_btcusdt_1h"


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _resolve_target_ts(stored_db: Path, explicit_ts: Optional[str]) -> pd.Timestamp:
    """Return a target timestamp: use explicit value, else the most recent ts in stored DB."""
    if explicit_ts:
        return pd.Timestamp(pd.to_datetime(explicit_ts, utc=True).tz_convert("UTC").tz_localize(None))
    con = duckdb.connect(str(stored_db), read_only=True)
    try:
        con.execute("SET TimeZone='UTC';")
        row = con.execute("SELECT MAX(ts) FROM features").fetchone()
        if not row or row[0] is None:
            raise SystemExit(f"No features found in stored DB: {stored_db}")
        return pd.Timestamp(row[0])
    finally:
        con.close()


def _fetch_features_row_by_ts(db_path: Path, ts: pd.Timestamp) -> Dict[str, float]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        con.execute("SET TimeZone='UTC';")
        # If multiple feature_key entries exist at the same ts, pick the latest created_at
        row = con.execute(
            """
            SELECT features
            FROM features
            WHERE ts = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [pd.Timestamp(ts).to_pydatetime()],
        ).fetchone()
        if not row:
            raise KeyError(f"Feature row missing at ts={ts} in {db_path}")
        val = row[0]
        if isinstance(val, dict):
            return {str(k): float(v) if v is not None else float("nan") for k, v in val.items()}
        # Fallback: try casting to JSON then parse keys
        s = con.execute(
            """
            SELECT CAST(features AS VARCHAR)
            FROM features
            WHERE ts = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [pd.Timestamp(ts).to_pydatetime()],
        ).fetchone()[0]
        return {str(k): float(v) if v is not None else float("nan") for k, v in json.loads(s).items()}
    finally:
        con.close()


def _compare_feature_maps(
    left: Dict[str, float], right: Dict[str, float], tol: float
) -> Tuple[List[str], List[str], List[Tuple[str, float, float, float]]]:
    """Return (missing_in_right, missing_in_left, diffs_over_tol).

    diffs_over_tol elements are tuples of (feature, left_val, right_val, abs_diff)
    where abs(left_val - right_val) > tol and both are finite numbers.
    """
    left_keys = set(left.keys())
    right_keys = set(right.keys())
    miss_r = sorted(list(left_keys - right_keys))
    miss_l = sorted(list(right_keys - left_keys))

    diffs: List[Tuple[str, float, float, float]] = []
    for k in sorted(left_keys & right_keys):
        lv = left[k]
        rv = right[k]
        try:
            lvf = float(lv)
            rvf = float(rv)
        except Exception:
            continue
        if pd.isna(lvf) and pd.isna(rvf):
            continue
        if pd.isna(lvf) or pd.isna(rvf):
            diffs.append((k, lvf, rvf, float("nan")))
            continue
        d = abs(lvf - rvf)
        if d > tol:
            diffs.append((k, lvf, rvf, d))
    return miss_r, miss_l, diffs


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="E2E: compare generated features against stored features")
    ap.add_argument("--duckdb", type=Path, default=Path(DEFAULT_OHLCV_DB), help="OHLCV DuckDB path")
    ap.add_argument("--table", type=str, default=DEFAULT_TABLE, help="OHLCV table name")
    ap.add_argument(
        "--stored-features-db",
        type=Path,
        default=Path(DEFAULT_STORED_FEATURES_DB),
        help="Reference features DuckDB path to compare against",
    )
    ap.add_argument(
        "--temp-features-db",
        type=Path,
        default=Path("tmp/e2e_generated_features.duckdb"),
        help="Temporary DuckDB path where generated features will be written",
    )
    ap.add_argument("--ts", type=str, default=None, help="Explicit timestamp to test. If omitted, pick most recent from stored DB")
    ap.add_argument("--base-hours", type=int, default=30 * 24, help="Base window in hours (e.g., 720)")
    ap.add_argument("--buffer-hours", type=int, default=6, help="Extra hours on top of base window")
    ap.add_argument("--timeframes", nargs="+", default=["1H", "4H", "12H", "1D"], help="Timeframes for lookbacks")
    ap.add_argument("--tol", type=float, default=1e-9, help="Absolute tolerance for value comparisons")
    args = ap.parse_args(argv)

    # 1) Decide target ts from stored DB
    target_ts = _resolve_target_ts(args.stored_features_db, args.ts)
    print(f"Using ts={target_ts} from stored DB")

    # 2) Generate features for that ts into a temp features DB
    temp_db: Path = args.temp_features_db
    _ensure_parent(temp_db)
    # Use a unique feature_key derived from the timestamp to avoid collisions in the temp DB
    derived_key = f"e2e_ts_{pd.Timestamp(target_ts).strftime('%Y%m%d_%H%M%S')}"
    cfg = BackfillFeaturesConfig(
        duckdb_path=args.duckdb,
        feat_duckdb_path=temp_db,
        table=args.table,
        feature_key=derived_key,
        feature_list_json=None,
        mode="ts_list",
        start=None,
        end=None,
        ts=[pd.Timestamp(target_ts).strftime("%Y-%m-%d %H:%M:%S")],
        ts_file=None,
        buffer_hours=int(args.buffer_hours),
        base_hours=int(args.base_hours),
        at_most=1,
        dry_run=False,
        overwrite=True,
        timeframes=list(args.timeframes),
    )
    print(f"Generating features for ts={target_ts} into {temp_db} ...")
    rc = backfill_features(cfg)
    if rc != 0:
        print(f"[ERROR] backfill_features returned non-zero: {rc}")
        return 3

    # 3) Fetch and compare
    stored_map = _fetch_features_row_by_ts(args.stored_features_db, target_ts)
    gen_map = _fetch_features_row_by_ts(temp_db, target_ts)

    miss_in_generated, miss_in_stored, diffs = _compare_feature_maps(stored_map, gen_map, tol=float(args.tol))

    ok = True
    if miss_in_generated:
        ok = False
        print(f"Missing in generated ({len(miss_in_generated)}): sample {miss_in_generated[:20]}")
    if miss_in_stored:
        ok = False
        print(f"Missing in stored ({len(miss_in_stored)}): sample {miss_in_stored[:20]}")
    if diffs:
        ok = False
        print(f"Value diffs > tol={args.tol} ({len(diffs)}). Showing up to 20:")
        for name, lv, rv, d in diffs[:20]:
            print(f"  {name}: stored={lv} generated={rv} abs_diff={d}")

    if ok:
        print("\nPASS: Generated features match stored features within tolerance.")
        return 0
    else:
        print("\nFAIL: Differences detected beyond tolerance.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
