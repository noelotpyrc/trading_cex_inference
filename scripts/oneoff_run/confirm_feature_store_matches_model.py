#!/usr/bin/env python3
"""
Confirm whether a DuckDB features table stores only the features used by a given model.

Checks recent timestamps and compares the JSON keys stored in `features.features`
against the LightGBM model's `feature_name()` list.

Usage examples:
  python run/confirm_feature_store_matches_model.py \
    --stored-features-db "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb" \
    --model-dir "/Volumes/Extreme SSD/trading_data/cex/models/run/binance_btcusdt_perp_1h/y_logret_168h" \
    --limit-ts 5

  python run/confirm_feature_store_matches_model.py \
    --stored-features-db /path/features.duckdb \
    --model-file /path/to/model.txt --ts "2025-08-07 12:00:00"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import duckdb  # type: ignore
import lightgbm as lgb  # type: ignore
import pandas as pd


def _load_model_features(model_file: Path) -> List[str]:
    booster = lgb.Booster(model_file=str(model_file))
    return list(booster.feature_name())


def _resolve_model_file(model_file: Optional[Path], model_dir: Optional[Path]) -> Path:
    if model_file is not None:
        mf = model_file
        if not mf.exists():
            raise FileNotFoundError(f"Model file not found: {mf}")
        return mf
    if model_dir is not None:
        candidate = Path(model_dir) / "model.txt"
        if not candidate.exists():
            raise FileNotFoundError(f"model.txt not found under model_dir: {model_dir}")
        return candidate
    raise SystemExit("Provide --model-file or --model-dir")


def _json_map_from_row(val) -> dict:
    if isinstance(val, dict):
        return val
    try:
        return json.loads(val)
    except Exception:
        return {}


def _fetch_recent_feature_keys(
    db_path: Path,
    *,
    ts: Optional[pd.Timestamp] = None,
    limit_ts: int = 3,
) -> List[Tuple[pd.Timestamp, Set[str]]]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        con.execute("SET TimeZone='UTC';")
        rows: List[Tuple[pd.Timestamp, Set[str]]] = []
        if ts is not None:
            df = con.execute(
                """
                SELECT ts, CAST(features AS VARCHAR) AS js
                FROM features
                WHERE ts = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                [pd.Timestamp(ts).to_pydatetime()],
            ).fetch_df()
        else:
            df = con.execute(
                """
                WITH latest AS (
                  SELECT ts, MAX(created_at) AS created_at
                  FROM features
                  GROUP BY ts
                )
                SELECT f.ts, CAST(f.features AS VARCHAR) AS js
                FROM features f
                JOIN latest l USING(ts, created_at)
                ORDER BY ts DESC
                LIMIT ?
                """,
                [int(limit_ts)],
            ).fetch_df()
        for _, r in df.iterrows():
            t = pd.Timestamp(r["ts"])
            mp = _json_map_from_row(r["js"])
            rows.append((t, set(map(str, mp.keys()))))
        return rows
    finally:
        con.close()


def _load_config_features(config_json: Path) -> List[str]:
    data = json.loads(Path(config_json).read_text())
    if isinstance(data, dict) and "features" in data:
        seq = data["features"]
    else:
        seq = data
    if not isinstance(seq, list):
        raise ValueError("Config JSON must be a list of feature names or a dict with key 'features'")
    return [str(x) for x in seq]


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Confirm if stored features match a model's feature set")
    ap.add_argument("--stored-features-db", type=Path, required=True, help="Path to features DuckDB")
    ap.add_argument("--model-file", type=Path, default=None, help="Path to LightGBM model.txt")
    ap.add_argument("--model-dir", type=Path, default=None, help="Directory containing model.txt (alternative to --model-file)")
    ap.add_argument("--ts", type=str, default=None, help="Specific timestamp to check (default: check recent N)")
    ap.add_argument("--limit-ts", type=int, default=3, help="Number of recent timestamps to check if --ts not provided")
    ap.add_argument("--config-json", type=Path, default=None, help="Path to a feature list JSON to compare against the model feature set")
    args = ap.parse_args(argv)

    model_path = _resolve_model_file(args.model_file, args.model_dir)
    model_features = set(_load_model_features(model_path))
    print(f"Model features: {len(model_features)} columns from {model_path}")

    target_ts = pd.Timestamp(pd.to_datetime(args.ts, utc=True).tz_convert("UTC").tz_localize(None)) if args.ts else None
    rows = _fetch_recent_feature_keys(args.stored_features_db, ts=target_ts, limit_ts=int(args.limit_ts))
    if not rows:
        print("No feature rows found to compare.")
        return 2

    all_match = True
    for ts, stored_keys in rows:
        missing_in_model = sorted(list(stored_keys - model_features))
        missing_in_store = sorted(list(model_features - stored_keys))
        is_subset = stored_keys.issubset(model_features)
        is_equal = stored_keys == model_features
        print(f"\nTimestamp: {ts}")
        print(f"  stored_keys: {len(stored_keys)}; model_features: {len(model_features)}")
        print(f"  stored ⊆ model: {is_subset}; stored == model: {is_equal}")
        if missing_in_model:
            print(f"  Extra in store (not in model): {len(missing_in_model)}; sample: {missing_in_model[:10]}")
        if missing_in_store:
            print(f"  Missing in store (in model): {len(missing_in_store)}; sample: {missing_in_store[:10]}")
        all_match = all_match and is_subset

    if all_match:
        print("\nConclusion: Stored feature keys are a subset of the model features for checked timestamps.")
    else:
        print("\nConclusion: Stored feature keys are NOT a subset for at least one timestamp.")

    # Optional: also compare a config JSON feature list to the model
    exit_code = 0 if all_match else 2
    if args.config_json is not None:
        cfg_feats = set(_load_config_features(args.config_json))
        missing_in_model = sorted(list(cfg_feats - model_features))
        missing_in_config = sorted(list(model_features - cfg_feats))
        print(f"\nConfig vs Model:")
        print(f"  config_json: {args.config_json}")
        print(f"  config count: {len(cfg_feats)}; model count: {len(model_features)}")
        print(f"  config ⊆ model: {cfg_feats.issubset(model_features)}; config == model: {cfg_feats == model_features}")
        if missing_in_model:
            print(f"  In config but not in model ({len(missing_in_model)}): sample {missing_in_model[:10]}")
        if missing_in_config:
            print(f"  In model but not in config ({len(missing_in_config)}): sample {missing_in_config[:10]}")
        if cfg_feats != model_features:
            exit_code = 2

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
