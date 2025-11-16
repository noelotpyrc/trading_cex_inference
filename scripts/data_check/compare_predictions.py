#!/usr/bin/env python3
"""Compare predictions from two CSV files over their overlapping timestamps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

TIMESTAMP_COL = "timestamp"
VALUE_COLS = ["y_pred", "prediction", "pred", "y"]


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"timestamp column missing in {path}")
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce", utc=True)
    df[TIMESTAMP_COL] = df[TIMESTAMP_COL].dt.tz_convert("UTC").dt.tz_localize(None)
    value_col = None
    for candidate in VALUE_COLS:
        if candidate in df.columns:
            value_col = candidate
            break
    if value_col is None:
        raise ValueError(f"No prediction column found in {path}; looked for {VALUE_COLS}")
    df = df[[TIMESTAMP_COL, value_col]].rename(columns={value_col: "y_pred"})
    return df.dropna(subset=[TIMESTAMP_COL, "y_pred"]).sort_values(TIMESTAMP_COL)


def compare_predictions(df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[str]:
    merged = df_a.merge(df_b, on=TIMESTAMP_COL, suffixes=("_a", "_b"), how="inner")
    if merged.empty:
        return ["No overlapping timestamps"]

    diff = merged["y_pred_a"] - merged["y_pred_b"]
    lines: List[str] = [
        f"Overlapping rows: {len(merged):,}",
        f"Mean diff: {diff.mean():.10g}",
        f"Median diff: {diff.median():.10g}",
        f"Std diff: {diff.std(ddof=0):.10g}",
        f"Min diff: {diff.min():.10g}",
        f"Max diff: {diff.max():.10g}",
    ]

    mismatches = diff[diff != 0]
    lines.append(f"Non-zero differences: {len(mismatches):,}")

    if not mismatches.empty:
        lines.append("Sample largest absolute differences:")
        largest = mismatches.abs().nlargest(5).index
        for idx in largest:
            ts = merged.loc[idx, TIMESTAMP_COL]
            lines.append(
                f"  {ts} -> A={merged.loc[idx, 'y_pred_a']:.10g} B={merged.loc[idx, 'y_pred_b']:.10g} diff={diff.loc[idx]:.10g}"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare predictions from two CSV files over their overlapping timestamps. "
            "Accepts positional arguments or --csv-a/--csv-b flags."
        )
    )
    parser.add_argument("csv_a", nargs="?", help="Path to first predictions CSV")
    parser.add_argument("csv_b", nargs="?", help="Path to second predictions CSV")
    parser.add_argument("--csv-a", dest="csv_a_opt", help="Optional flag-form path for source A")
    parser.add_argument("--csv-b", dest="csv_b_opt", help="Optional flag-form path for source B")
    args = parser.parse_args()

    csv_a_path = args.csv_a_opt or args.csv_a
    csv_b_path = args.csv_b_opt or args.csv_b
    if not csv_a_path or not csv_b_path:
        parser.error("Please provide paths for both CSV inputs (either positional args or --csv-a/--csv-b)")

    df_a = load_predictions(Path(csv_a_path))
    df_b = load_predictions(Path(csv_b_path))

    print(f"Source A rows: {len(df_a):,}")
    print(f"Source B rows: {len(df_b):,}")
    for line in compare_predictions(df_a, df_b):
        print(line)


if __name__ == "__main__":
    main()
