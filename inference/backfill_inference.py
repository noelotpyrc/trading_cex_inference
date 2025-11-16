#!/usr/bin/env python3
"""
Backfill predictions using pre-computed features from DuckDB.

This script loads pre-computed features from a features table and runs model
inference to generate predictions. It assumes features have already been
backfilled using backfill_features.py.

Key differences from legacy backfill_inference_missing.py:
  - No OHLCV loading or processing (20-50x faster)
  - No feature computation (features must exist in DB)
  - Validates OHLCV ↔ Features consistency before inference
  - Smart feature loading with dedup logic (most recent by default)

Selection modes for target timestamps:
  - window: explicit --start/--end date range
  - last_from_predictions: continue from last prediction timestamp
  - ts_list: explicit timestamps via --ts or --ts-file

Example:
  python scripts/backfill_inference.py \
    --duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb" \
    --feat-duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb" \
    --pred-duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_prediction.duckdb" \
    --table ohlcv_btcusdt_1h \
    --model-path "/Volumes/Extreme SSD/trading_data/cex/models/run/binance_btcusdt_perp_1h/y_logret_168h" \
    --mode last_from_predictions \
    --dataset "binance_btcusdt_perp_1h"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb  # type: ignore
import pandas as pd


DEFAULT_TABLE = "ohlcv_btcusdt_1h"


def _now_floor_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).floor("h").tz_convert(None)


def _is_table_present(con, table_name: str) -> bool:
    try:
        res = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ? LIMIT 1",
            [table_name]
        ).fetchone()
        return bool(res)
    except Exception:
        return False


def _query_ohlcv_timestamps(
    con: duckdb.DuckDBPyConnection, table: str, target_timestamps: List[pd.Timestamp]
) -> List[pd.Timestamp]:
    """Query OHLCV DB for which target timestamps have data."""
    if not target_timestamps:
        return []

    min_ts = min(target_timestamps)
    max_ts = max(target_timestamps)

    q = f"SELECT timestamp FROM {table} WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp"
    df = con.execute(q, [min_ts.to_pydatetime(), max_ts.to_pydatetime()]).fetch_df()

    if df.empty:
        return []

    ohlcv_ts = set(pd.to_datetime(df["timestamp"]).dt.tz_localize(None))
    return [ts for ts in target_timestamps if ts in ohlcv_ts]


def _query_feature_timestamps(
    con: duckdb.DuckDBPyConnection,
    target_timestamps: List[pd.Timestamp],
    feature_key: Optional[str] = None
) -> List[pd.Timestamp]:
    """Query features DB for which target timestamps have features."""
    if not target_timestamps:
        return []

    min_ts = min(target_timestamps)
    max_ts = max(target_timestamps)

    if feature_key:
        q = "SELECT DISTINCT ts FROM features WHERE feature_key = ? AND ts BETWEEN ? AND ? ORDER BY ts"
        df = con.execute(q, [feature_key, min_ts.to_pydatetime(), max_ts.to_pydatetime()]).fetch_df()
    else:
        q = "SELECT DISTINCT ts FROM features WHERE ts BETWEEN ? AND ? ORDER BY ts"
        df = con.execute(q, [min_ts.to_pydatetime(), max_ts.to_pydatetime()]).fetch_df()

    if df.empty:
        return []

    feat_ts = set(pd.to_datetime(df["ts"]).dt.tz_localize(None))
    return [ts for ts in target_timestamps if ts in feat_ts]


def validate_data_consistency(
    ohlcv_con: duckdb.DuckDBPyConnection,
    feat_con: duckdb.DuckDBPyConnection,
    table: str,
    target_timestamps: List[pd.Timestamp],
    feature_key: Optional[str] = None
) -> List[pd.Timestamp]:
    """Validate OHLCV and features are in sync for target timestamps.

    Returns: List of valid timestamps (both OHLCV and features present)
    Raises: ValueError if critical discrepancies found
    """
    # Get OHLCV timestamps
    ohlcv_ts_set = set(_query_ohlcv_timestamps(ohlcv_con, table, target_timestamps))

    # Get feature timestamps
    feat_ts_set = set(_query_feature_timestamps(feat_con, target_timestamps, feature_key))

    # Find discrepancies
    missing_features = ohlcv_ts_set - feat_ts_set  # OHLCV exists but no features
    missing_ohlcv = feat_ts_set - ohlcv_ts_set     # Features exist but no OHLCV

    if missing_features:
        print(f"ERROR: {len(missing_features)} timestamps have OHLCV but missing features:")
        for ts in sorted(missing_features)[:10]:
            print(f"  - {ts}")
        if len(missing_features) > 10:
            print(f"  ... and {len(missing_features) - 10} more")

        suggestion = f"--feature-key {feature_key}" if feature_key else ""
        raise ValueError(
            f"Feature data is incomplete. "
            f"Please run: backfill_features.py {suggestion} --mode last_from_features"
        )

    if missing_ohlcv:
        print(f"ERROR: {len(missing_ohlcv)} timestamps have features but missing OHLCV:")
        for ts in sorted(missing_ohlcv)[:10]:
            print(f"  - {ts}")
        if len(missing_ohlcv) > 10:
            print(f"  ... and {len(missing_ohlcv) - 10} more")
        raise ValueError(
            f"OHLCV data is incomplete or features are ahead of OHLCV. "
            f"This should not happen - check data integrity."
        )

    # Return intersection (both present)
    valid = sorted(ohlcv_ts_set & feat_ts_set)
    if not valid:
        print("WARNING: No timestamps have both OHLCV and features present")

    return valid


def _load_features_for_timestamp(
    con: duckdb.DuckDBPyConnection,
    ts: pd.Timestamp,
    feature_key: Optional[str] = None
) -> tuple[Optional[dict], Optional[str]]:
    """Load pre-computed features for a timestamp.

    If feature_key is specified, loads most recent features for that key.
    Otherwise, loads most recent features across all keys (dedup by created_at DESC).

    Returns:
        (features_dict, feature_key_used)
    """
    try:
        if feature_key:
            # Get features for specific feature_key
            q = """
                SELECT features, feature_key
                FROM features
                WHERE feature_key = ?
                  AND ts = ?
                ORDER BY created_at DESC
                LIMIT 1
            """
            row = con.execute(q, [feature_key, ts.to_pydatetime()]).fetchone()
        else:
            # Get most recent features regardless of key (return the key used)
            q = """
                SELECT features, feature_key
                FROM features
                WHERE ts = ?
                ORDER BY created_at DESC
                LIMIT 1
            """
            row = con.execute(q, [ts.to_pydatetime()]).fetchone()

        if not row:
            return None, None

        features_json = row[0]
        used_feature_key = row[1]

        if isinstance(features_json, str):
            return json.loads(features_json), used_feature_key
        return features_json, used_feature_key
    except Exception as e:
        print(f"ERROR loading features for {ts}: {e}")
        return None, None


def _list_closed_bars_in_window(
    con: duckdb.DuckDBPyConnection, table: str, start: pd.Timestamp, end: pd.Timestamp
) -> List[pd.Timestamp]:
    """List closed bars (timestamp <= now - 1h) in OHLCV table."""
    now_floor = _now_floor_utc()
    cutoff = now_floor - pd.Timedelta(hours=1)
    start = pd.to_datetime(start).tz_localize(None)
    end = pd.to_datetime(end).tz_localize(None)
    q = f"""
        SELECT timestamp FROM {table}
        WHERE timestamp BETWEEN ? AND ? AND timestamp <= ?
        ORDER BY timestamp
    """
    df = con.execute(q, [start.to_pydatetime(), end.to_pydatetime(), cutoff.to_pydatetime()]).fetch_df()
    return [pd.Timestamp(t) for t in df["timestamp"]] if not df.empty else []


def _last_prediction_ts(con, model_path: str, dataset: Optional[str] = None) -> Optional[pd.Timestamp]:
    """Get last prediction timestamp for a model (and optional dataset filter)."""
    try:
        if dataset:
            q = "SELECT MAX(ts) FROM predictions WHERE model_path = ? AND dataset = ?"
            row = con.execute(q, [model_path, dataset]).fetchone()
        else:
            q = "SELECT MAX(ts) FROM predictions WHERE model_path = ?"
            row = con.execute(q, [model_path]).fetchone()

        if row and row[0] is not None:
            return pd.Timestamp(row[0])
    except Exception:
        pass
    return None


def _prediction_exists(con, ts: pd.Timestamp, model_path: str) -> bool:
    """Check if prediction exists for (ts, model_path)."""
    try:
        row = con.execute(
            "SELECT 1 FROM predictions WHERE ts = ? AND model_path = ? LIMIT 1",
            [ts.to_pydatetime(), model_path]
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _parse_ts_list(ts_list) -> List[pd.Timestamp]:
    """Parse list of timestamp strings."""
    out: List[pd.Timestamp] = []
    for s in ts_list:
        t = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(t):
            continue
        out.append(t.tz_convert("UTC").tz_localize(None))
    return sorted(set(out))


def _load_ts_file(path: Path) -> List[pd.Timestamp]:
    """Load timestamps from file (one per line)."""
    if not path or not path.exists():
        return []
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return _parse_ts_list(lines)


def ensure_predictions_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create predictions table if it doesn't exist."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            ts TIMESTAMP NOT NULL,
            model_path TEXT NOT NULL,
            y_pred DOUBLE NOT NULL,
            feature_key TEXT,
            dataset TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ts, model_path)
        )
    """)


def _save_prediction(
    con: duckdb.DuckDBPyConnection,
    ts: pd.Timestamp,
    y_pred: float,
    model_path: str,
    feature_key: Optional[str] = None,
    dataset: Optional[str] = None
) -> None:
    """Save prediction to predictions table."""
    # Delete existing if present
    con.execute(
        "DELETE FROM predictions WHERE ts = ? AND model_path = ?",
        [ts.to_pydatetime(), model_path]
    )

    # Check if dataset column exists
    schema = con.execute("DESCRIBE predictions").fetchall()
    has_dataset = any(col[0] == 'dataset' for col in schema)

    # Insert new prediction (with or without dataset column)
    if has_dataset:
        con.execute(
            "INSERT INTO predictions (ts, model_path, y_pred, feature_key, dataset) VALUES (?, ?, ?, ?, ?)",
            [ts.to_pydatetime(), model_path, float(y_pred), feature_key, dataset]
        )
    else:
        con.execute(
            "INSERT INTO predictions (ts, model_path, y_pred, feature_key) VALUES (?, ?, ?, ?)",
            [ts.to_pydatetime(), model_path, float(y_pred), feature_key]
        )


def _resolve_model_path(model_path: Path) -> Path:
    """Resolve model path to model.txt file path (matching legacy behavior).

    Args:
        model_path: Path to model directory or file

    Returns:
        Path to model.txt file

    Examples:
        /path/to/y_logret_168h -> /path/to/y_logret_168h/model.txt
        /path/to/y_logret_168h/model.txt -> /path/to/y_logret_168h/model.txt
    """
    model_path = Path(model_path)

    if model_path.is_file():
        # If it's already a file, return as-is
        return model_path
    elif model_path.is_dir():
        # If it's a directory, append model.txt
        return model_path / "model.txt"
    else:
        # Path doesn't exist, assume it's a directory and append model.txt
        return model_path / "model.txt"


def _load_model(model_path: Path):
    """Load LightGBM model from path."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError(
            "lightgbm is required for inference. "
            "Install with: pip install lightgbm>=3.3.0"
        )

    model_path = Path(model_path)

    # Try loading as booster file
    if model_path.is_file():
        return lgb.Booster(model_file=str(model_path))

    # Try loading from directory (look for model.txt or model.lgb)
    if model_path.is_dir():
        for fname in ["model.txt", "model.lgb", "lgbm_model.txt"]:
            model_file = model_path / fname
            if model_file.exists():
                return lgb.Booster(model_file=str(model_file))

    raise FileNotFoundError(f"Could not find LightGBM model at {model_path}")


@dataclass
class BackfillInferenceConfig:
    duckdb_path: Path  # OHLCV source DB (for validation)
    feat_duckdb_path: Path  # Features source DB
    pred_duckdb_path: Path  # Predictions target DB
    table: str
    model_path: Path
    feature_key: Optional[str]  # Optional: use most recent if not specified
    dataset: Optional[str]
    mode: str
    start: Optional[str]
    end: Optional[str]
    ts: List[str]
    ts_file: Optional[Path]
    at_most: Optional[int]
    dry_run: bool


def _select_target_timestamps(cfg: BackfillInferenceConfig, resolved_model_path: str) -> List[pd.Timestamp]:
    """Select target timestamps based on mode."""
    con_ohlcv = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        con_ohlcv.execute("SET TimeZone='UTC';")
        now_floor = _now_floor_utc()
        cutoff = now_floor - pd.Timedelta(hours=1)

        if cfg.mode == "ts_list":
            ts_list = _parse_ts_list(cfg.ts)
            if cfg.ts_file:
                ts_list = sorted(set(ts_list) | set(_load_ts_file(cfg.ts_file)))
            if not ts_list:
                return []
            lo, hi = min(ts_list), max(ts_list)
            in_db = _list_closed_bars_in_window(con_ohlcv, cfg.table, lo, hi)
            ts = sorted(set(ts_list).intersection(set(in_db)))
            return [t for t in ts if t <= cutoff]

        if cfg.mode == "last_from_predictions":
            # Get last prediction timestamp
            try:
                con_pred = duckdb.connect(str(cfg.pred_duckdb_path), read_only=True)
                con_pred.execute("SET TimeZone='UTC';")
                last_pred = _last_prediction_ts(con_pred, resolved_model_path, cfg.dataset)
            finally:
                try:
                    con_pred.close()
                except Exception:
                    pass

            start_ts = (last_pred + pd.Timedelta(hours=1)) if last_pred is not None else None
            end_ts = pd.to_datetime(cfg.end).tz_localize(None) if cfg.end else cutoff
            if start_ts is None:
                start_ts = end_ts - pd.Timedelta(hours=48)
            return _list_closed_bars_in_window(con_ohlcv, cfg.table, start_ts, end_ts)

        # Default: window mode
        if not cfg.start and not cfg.end:
            end_ts = cutoff
            start_ts = end_ts - pd.Timedelta(hours=48)
        else:
            start_ts = pd.to_datetime(cfg.start).tz_localize(None) if cfg.start else None
            end_ts = pd.to_datetime(cfg.end).tz_localize(None) if cfg.end else cutoff
            if start_ts is None:
                start_ts = end_ts - pd.Timedelta(hours=48)
        return _list_closed_bars_in_window(con_ohlcv, cfg.table, start_ts, end_ts)
    finally:
        con_ohlcv.close()


def backfill_inference(cfg: BackfillInferenceConfig) -> int:
    # Load model
    print(f"Loading model from {cfg.model_path}...")
    try:
        model = _load_model(cfg.model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return 2

    # Resolve model path to model.txt file path (matching legacy behavior)
    resolved_model_path = _resolve_model_path(cfg.model_path)
    print(f"Resolved model path: {resolved_model_path}")

    # Determine target timestamps
    print(f"Selecting target timestamps (mode={cfg.mode})...")
    targets = _select_target_timestamps(cfg, str(resolved_model_path))
    if cfg.at_most is not None and cfg.at_most > 0:
        targets = targets[: cfg.at_most]

    if not targets:
        print("No target timestamps found to run inference (after filtering/limits)")
        return 0

    print(f"Found {len(targets)} target timestamps: [{targets[0]} .. {targets[-1]}]")

    # Validate OHLCV ↔ Features consistency
    print(f"Validating OHLCV ↔ Features consistency...")
    con_ohlcv = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    con_feat = duckdb.connect(str(cfg.feat_duckdb_path), read_only=True)

    try:
        con_ohlcv.execute("SET TimeZone='UTC';")
        con_feat.execute("SET TimeZone='UTC';")

        valid_timestamps = validate_data_consistency(
            con_ohlcv, con_feat, cfg.table, targets, cfg.feature_key
        )
        print(f"✓ Validation passed: {len(valid_timestamps)} timestamps have both OHLCV and features")
    except ValueError as e:
        print(f"[ERROR] Data validation failed: {e}")
        return 2
    finally:
        con_ohlcv.close()
        con_feat.close()

    if not valid_timestamps:
        print("No valid timestamps to process after validation")
        return 0

    targets = valid_timestamps

    print(f"Backfill inference plan: bars={len(targets)} model={resolved_model_path.name} key={cfg.feature_key or 'most_recent'}")
    if cfg.dry_run:
        for t in targets:
            print("  -", t)
        return 0

    # Open connections for inference loop
    con_feat = duckdb.connect(str(cfg.feat_duckdb_path), read_only=True)
    con_pred = duckdb.connect(str(cfg.pred_duckdb_path))

    try:
        con_feat.execute("SET TimeZone='UTC';")
        con_pred.execute("SET TimeZone='UTC';")
        ensure_predictions_table(con_pred)

        processed = 0
        for ts in targets:
            # Skip if prediction exists
            if _prediction_exists(con_pred, ts, str(resolved_model_path)):
                print(f"SKIP {ts}: prediction already exists")
                continue

            # Load features (returns features dict and actual feature_key used)
            features_dict, used_feature_key = _load_features_for_timestamp(con_feat, ts, cfg.feature_key)
            if not features_dict:
                print(f"SKIP {ts}: no features found")
                continue

            # Convert to DataFrame for model prediction
            features_df = pd.DataFrame([features_dict])

            # Validate features align with model
            model_features = model.feature_name()
            missing = [f for f in model_features if f not in features_dict]
            if missing:
                print(f"SKIP {ts}: missing {len(missing)} required features (first 10): {missing[:10]}")
                continue

            # Reorder to match model features
            features_df = features_df[model_features]

            # Predict
            try:
                y_pred = model.predict(features_df)[0]
            except Exception as e:
                print(f"SKIP {ts}: prediction failed ({e})")
                continue

            # Save prediction (use the actual feature_key that was used)
            try:
                _save_prediction(
                    con_pred, ts, y_pred, str(resolved_model_path),
                    used_feature_key, cfg.dataset
                )
                processed += 1
                print(f"OK {ts}: prediction={y_pred:.6f}")
            except Exception as e:
                print(f"SKIP {ts}: failed to save prediction ({e})")
                continue
    finally:
        try:
            con_feat.close()
        except Exception:
            pass
        try:
            con_pred.close()
        except Exception:
            pass

    print(f"Backfill inference complete: wrote={processed} predictions out of {len(targets)} planned")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> BackfillInferenceConfig:
    p = argparse.ArgumentParser(description="Backfill predictions using pre-computed features")
    p.add_argument("--duckdb", type=Path, required=True, help="DuckDB database path for OHLCV (validation only)")
    p.add_argument("--feat-duckdb", type=Path, required=True, help="DuckDB path for features table")
    p.add_argument("--pred-duckdb", type=Path, required=True, help="DuckDB path for predictions table")
    p.add_argument("--table", type=str, default=DEFAULT_TABLE, help="OHLCV table name in DuckDB")
    p.add_argument("--model-path", type=Path, required=True, help="Path to LightGBM model file or directory")
    p.add_argument("--feature-key", default=None, help="Optional: feature key to use (default: most recent)")
    p.add_argument("--dataset", default=None, help="Dataset label for predictions table")
    p.add_argument("--mode", choices=["window", "last_from_predictions", "ts_list"], default="last_from_predictions")
    p.add_argument("--start", default=None, help="Start timestamp (inclusive) for window mode")
    p.add_argument("--end", default=None, help="End timestamp (inclusive) for window/last_from_predictions")
    p.add_argument("--ts", nargs="*", default=[], help="Explicit timestamps for ts_list mode")
    p.add_argument("--ts-file", type=Path, default=None, help="File with one timestamp per line for ts_list mode")
    p.add_argument("--at-most", type=int, default=None, help="Cap the number of bars to process")
    p.add_argument("--dry-run", action="store_true", help="Plan only; do not run inference or write predictions")
    args = p.parse_args(argv)

    return BackfillInferenceConfig(
        duckdb_path=args.duckdb,
        feat_duckdb_path=args.feat_duckdb,
        pred_duckdb_path=args.pred_duckdb,
        table=str(args.table),
        model_path=args.model_path,
        feature_key=args.feature_key,
        dataset=args.dataset,
        mode=str(args.mode),
        start=args.start,
        end=args.end,
        ts=list(args.ts or []),
        ts_file=args.ts_file,
        at_most=(int(args.at_most) if args.at_most is not None else None),
        dry_run=bool(args.dry_run),
    )


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    try:
        return backfill_inference(cfg)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
