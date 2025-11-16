#!/usr/bin/env python3
"""Compare O/H/L/C/V columns between two OHLCV CSV sources on common timestamps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    elif "time" in df.columns:
        ts = pd.to_datetime(df["time"], errors="coerce", utc=True)
    elif "open_time" in df.columns:
        ts = pd.to_datetime(df["open_time"], unit="ms", errors="coerce", utc=True)
    else:
        raise ValueError(f"Could not find timestamp column in {path}")

    df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")

    missing = [col for col in COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected OHLCV columns {missing} in {path}")

    df[COLUMNS] = df[COLUMNS].apply(pd.to_numeric, errors="coerce")
    return df[["timestamp", *COLUMNS]]


def describe_differences(df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[str]:
    merged = df_a.merge(df_b, on="timestamp", suffixes=("_a", "_b"), how="inner")
    if merged.empty:
        return ["No overlapping timestamps"]

    lines: List[str] = [
        f"Total overlapping rows: {len(merged):,}",
    ]

    for column in COLUMNS:
        col_a = f"{column}_a"
        col_b = f"{column}_b"
        diff = merged[col_a] - merged[col_b]
        nonzero = diff[diff != 0]
        lines.append(f"\nColumn {column.upper()}:")
        lines.append(f"  differing rows: {len(nonzero):,}")
        if nonzero.empty:
            continue
        lines.append(f"  mean diff: {nonzero.mean():.6g}")
        lines.append(f"  median diff: {nonzero.median():.6g}")
        lines.append(f"  min diff: {nonzero.min():.6g}")
        lines.append(f"  max diff: {nonzero.max():.6g}")
        lines.append(f"  std diff: {nonzero.std(ddof=0):.6g}")

        largest = nonzero.abs().nlargest(5)
        lines.append("  largest absolute differences:")
        for ts, value in largest.items():
            timestamp = merged.loc[ts, "timestamp"]
            lines.append(
                f"    {timestamp}: diff={value:.6g} A={merged.loc[ts, col_a]:.6g} B={merged.loc[ts, col_b]:.6g}"
            )

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_a", help="First OHLCV CSV path (e.g., perp feed)")
    parser.add_argument("csv_b", help="Second OHLCV CSV path")
    args = parser.parse_args()

    path_a = Path(args.csv_a)
    path_b = Path(args.csv_b)

    df_a = load_ohlcv(path_a)
    df_b = load_ohlcv(path_b)

    print(f"Source A: {path_a} rows={len(df_a):,} range={df_a['timestamp'].min()} -> {df_a['timestamp'].max()}")
    print(f"Source B: {path_b} rows={len(df_b):,} range={df_b['timestamp'].min()} -> {df_b['timestamp'].max()}")

    for line in describe_differences(df_a, df_b):
        print(line)


if __name__ == "__main__":
    main()
