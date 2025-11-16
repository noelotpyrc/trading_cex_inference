#!/usr/bin/env python3
"""Detailed comparison of feature values between two CSV sources over a time range."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

TIMESTAMP_COL = "timestamp"
DEFAULT_TOL = 1e-9


def _load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"timestamp column missing in {path}")
    ts = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce", utc=True)
    df[TIMESTAMP_COL] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def _filter_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (df[TIMESTAMP_COL] >= start) & (df[TIMESTAMP_COL] <= end)
    return df.loc[mask].copy()


def _numeric_or_object(series: pd.Series) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    return converted


def _format_stat(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.10g}"


def compare_features(
    features_a: pd.DataFrame,
    features_b: pd.DataFrame,
    *,
    tolerance: float,
) -> List[str]:
    a = features_a.set_index(TIMESTAMP_COL)
    b = features_b.set_index(TIMESTAMP_COL)

    # Align on common timestamps
    common_ts = a.index.intersection(b.index)
    a = a.loc[common_ts].sort_index()
    b = b.loc[common_ts].sort_index()

    lines: List[str] = []
    lines.append(f"Common timestamps: {len(common_ts):,}")
    if len(common_ts) == 0:
        return lines

    # Determine shared feature columns (exclude timestamp and y-targets)
    columns = [
        col
        for col in a.columns
        if col in b.columns and not col.lower().startswith("y")
    ]

    lines.append(f"Compared feature columns: {len(columns):,}")

    for col in columns:
        series_a = a[col]
        series_b = b[col]

        num_a = _numeric_or_object(series_a)
        num_b = _numeric_or_object(series_b)

        both_nan = num_a.isna() & num_b.isna()
        diff = num_a - num_b
        diff[both_nan] = 0.0

        mismatch_mask = (~both_nan) & (diff.abs() > tolerance)
        mismatch_count = int(mismatch_mask.sum())
        if mismatch_count == 0:
            continue

        lines.append(f"\nColumn {col}: mismatches={mismatch_count}")
        mismatches = diff[mismatch_mask]
        stats = {
            "mean": mismatches.mean(),
            "median": mismatches.median(),
            "min": mismatches.min(),
            "max": mismatches.max(),
            "std": mismatches.std(ddof=0),
        }
        for name, value in stats.items():
            lines.append(f"  {name}: {_format_stat(value)}")

        largest = mismatches.abs().nlargest(5).index
        lines.append("  sample mismatches:")
        for ts in largest:
            lines.append(
                f"    {ts} -> A={_format_stat(num_a.loc[ts])} B={_format_stat(num_b.loc[ts])} diff={_format_stat(diff.loc[ts])}"
            )

    if len(lines) == 2:
        lines.append("No mismatches detected within the specified range.")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features_a", help="Path to features_all_tf.csv")
    parser.add_argument("features_b", help="Path to merged_features_targets.csv")
    parser.add_argument("--start", required=True, help="Start timestamp (inclusive)")
    parser.add_argument("--end", required=True, help="End timestamp (inclusive)")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOL, help="Numeric tolerance for equality")
    args = parser.parse_args()

    features_path_a = Path(args.features_a)
    features_path_b = Path(args.features_b)

    df_a = _load_features(features_path_a)
    df_b = _load_features(features_path_b)

    start_ts = pd.to_datetime(args.start, utc=True).tz_convert("UTC").tz_localize(None)
    end_ts = pd.to_datetime(args.end, utc=True).tz_convert("UTC").tz_localize(None)

    df_a = _filter_range(df_a, start_ts, end_ts)
    df_b = _filter_range(df_b, start_ts, end_ts)

    print(f"Source A rows in range: {len(df_a):,}")
    print(f"Source B rows in range: {len(df_b):,}")

    for line in compare_features(df_a, df_b, tolerance=args.tolerance):
        print(line)


if __name__ == "__main__":
    main()
