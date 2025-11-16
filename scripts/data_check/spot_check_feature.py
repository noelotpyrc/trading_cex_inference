#!/usr/bin/env python3
"""Spot-check a feature by recomputing it from lookback data and comparing stored CSV values."""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure repository root is importable when script is run from arbitrary CWD
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_engineering.build_multi_timeframe_features import compute_features_one

TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
KEY_FMT = "%Y%m%d_%H%M%S"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timestamp", help="Timestamp to inspect, e.g. '2024-02-05 17:00:00'")
    parser.add_argument("timeframe", choices=["1H", "4H", "12H", "1D"], help="Timeframe suffix to inspect")
    parser.add_argument("feature", help="Base feature name (without timeframe suffix), e.g. close_obv")
    parser.add_argument("features_a", help="Path to first features CSV (e.g., features_all_tf.csv)")
    parser.add_argument("features_b", help="Path to second features CSV (e.g., merged_features_targets.csv)")
    parser.add_argument("lookbacks_a", help="Lookbacks directory corresponding to features_a")
    parser.add_argument("lookbacks_b", help="Lookbacks directory corresponding to features_b")
    return parser.parse_args()


def load_features_value(path: Path, ts: pd.Timestamp, column: str) -> Optional[float]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if 'timestamp' not in df.columns:
        if 'time' in df.columns:
            df = df.rename(columns={'time': 'timestamp'})
    if 'timestamp' not in df.columns:
        raise ValueError(f"timestamp column not found in {path}")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
    match = df[df['timestamp'] == ts]
    if match.empty or column not in df.columns:
        return None
    value = match.iloc[0][column]
    return None if pd.isna(value) else float(value)


def load_lookback_row(directory: Path, timeframe: str, key: str) -> pd.DataFrame:
    path = directory / f"lookbacks_{timeframe}.pkl"
    with path.open('rb') as handle:
        payload = pickle.load(handle)
    rows = payload['rows']
    df = rows.get(key)
    if df is None:
        raise KeyError(f"Key {key} missing in {path}")
    # Normalize index to naive timestamps for consistency
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert(None)
    return df


def recompute_feature(df: pd.DataFrame, timeframe: str, full_name: str) -> Optional[float]:
    out = compute_features_one(df, tf=timeframe, skip_slow=False)
    value = out.get(full_name)
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return None
    return float(value)


def main() -> None:
    args = parse_args()
    ts = pd.to_datetime(args.timestamp, utc=True).tz_convert('UTC').tz_localize(None)
    key = ts.strftime(KEY_FMT)
    tf = args.timeframe.upper()
    column = f"{args.feature}_{tf}"

    features_a_path = Path(args.features_a)
    features_b_path = Path(args.features_b)

    value_a = load_features_value(features_a_path, ts, column)
    value_b = load_features_value(features_b_path, ts, column)

    lookback_a = load_lookback_row(Path(args.lookbacks_a), tf, key)
    lookback_b = load_lookback_row(Path(args.lookbacks_b), tf, key)

    recomputed_a = recompute_feature(lookback_a, tf, column)
    recomputed_b = recompute_feature(lookback_b, tf, column)

    print(f"Timestamp: {args.timestamp} ({key})")
    print(f"Feature column: {column}")
    print("--- Stored CSV values ---")
    print(f"features_a ({features_a_path}): {value_a}")
    print(f"features_b ({features_b_path}): {value_b}")
    print("--- Recomputed from lookbacks ---")
    print(f"lookbacks_a ({args.lookbacks_a}): {recomputed_a}")
    print(f"lookbacks_b ({args.lookbacks_b}): {recomputed_b}")

    if recomputed_a is not None and value_a is not None:
        print(f"diff_a (stored - recomputed): {value_a - recomputed_a}")
    if recomputed_b is not None and value_b is not None:
        print(f"diff_b (stored - recomputed): {value_b - recomputed_b}")


if __name__ == "__main__":
    main()
