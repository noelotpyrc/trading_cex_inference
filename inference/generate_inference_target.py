#!/usr/bin/env python3
"""Compute ground-truth targets for inference timestamps and persist to DuckDB."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from feature_engineering.targets import TargetGenerationConfig, extract_forward_window, generate_targets_for_row
from run.data_loader import load_ohlcv_duckdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate forward-looking targets for inference rows")
    parser.add_argument("--duckdb", required=True, type=Path, help="Path to DuckDB database containing OHLCV data")
    parser.add_argument("--ohlcv-table", default="ohlcv_btcusdt_1h", help="DuckDB table name with OHLCV data")
    parser.add_argument("--timestamp", help="Single timestamp (UTC ISO) of the entry bar to label")
    parser.add_argument("--start", help="Start timestamp (inclusive, UTC ISO) for bulk generation")
    parser.add_argument("--end", help="End timestamp (inclusive, UTC ISO) for bulk generation")
    parser.add_argument("--horizon", type=int, default=168, help="Forward horizon in bars")
    parser.add_argument(
        "--target-label",
        type=str,
        help="Human-readable horizon label (default: '<horizon>h')",
    )
    parser.add_argument(
        "--target-prefix",
        choices=["y_logret", "y_ret"],
        default="y_logret",
        help="Return type to compute (log vs. simple)",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        help="Override target name stored in the table (defaults to '<prefix>_<label>')",
    )
    parser.add_argument("--feature-key", type=str, help="Optional feature key label to include in the output")
    parser.add_argument("--min-forward-rows", type=int, default=None, help="Require at least this many forward rows (defaults to horizon)")
    parser.add_argument("--output-csv", required=True, type=Path, help="Path to write the generated targets CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.timestamp and not args.start:
        raise ValueError("Provide either --timestamp or --start/--end for bulk generation")
    if args.timestamp and (args.start or args.end):
        raise ValueError("Specify either --timestamp or --start/--end, not both")

    horizon = int(args.horizon)
    if horizon <= 0:
        raise ValueError("--horizon must be positive")

    target_label = args.target_label or f"{horizon}h"
    target_name = args.target_name or f"{args.target_prefix}_{target_label}"
    log_returns = args.target_prefix == "y_logret"
    min_forward = args.min_forward_rows or horizon

    if args.timestamp:
        entry_ts = pd.to_datetime(args.timestamp, utc=True)
        if pd.isna(entry_ts):
            raise ValueError(f"Unable to parse timestamp: {args.timestamp}")
        entry_ts = entry_ts.tz_convert('UTC').tz_localize(None)
        start_ts = entry_ts
        end_ts = entry_ts
    else:
        start_ts = pd.to_datetime(args.start, utc=True)
        if pd.isna(start_ts):
            raise ValueError(f"Unable to parse --start timestamp: {args.start}")
        start_ts = start_ts.tz_convert('UTC').tz_localize(None)
        if args.end:
            end_ts = pd.to_datetime(args.end, utc=True)
            if pd.isna(end_ts):
                raise ValueError(f"Unable to parse --end timestamp: {args.end}")
            end_ts = end_ts.tz_convert('UTC').tz_localize(None)
        else:
            # Use latest complete timestamp available in the OHLCV table
            df_latest = load_ohlcv_duckdb(
                db_path=args.duckdb,
                table=args.ohlcv_table,
                start=start_ts,
                end=None,
            )
            if df_latest.empty:
                raise ValueError("Unable to determine end timestamp automatically; OHLCV query returned no rows")
            end_ts = pd.Timestamp(df_latest["timestamp"].iloc[-1])
        if end_ts < start_ts:
            raise ValueError("--end must be >= --start")

    # Fetch the OHLCV slice covering requested entry bars plus forward window
    forward_end = end_ts + pd.Timedelta(hours=horizon)
    ohlcv = load_ohlcv_duckdb(
        db_path=args.duckdb,
        table=args.ohlcv_table,
        start=start_ts,
        end=forward_end,
    )
    if ohlcv.empty:
        raise ValueError("OHLCV query returned no rows; check table name and timestamp range")

    # Identify entry indices within range
    candidate_mask = (ohlcv["timestamp"] >= start_ts) & (ohlcv["timestamp"] <= end_ts)
    entry_indices = ohlcv.index[candidate_mask].tolist()
    if not entry_indices:
        raise ValueError("No OHLCV rows found within the requested timestamp range")

    config = TargetGenerationConfig(
        horizons_bars=[horizon],
        horizon_labels={horizon: target_label},
        include_returns=True,
        include_mfe_mae=False,
        include_barriers=False,
        log_returns=log_returns,
    )

    records = []
    for idx in entry_indices:
        entry_ts_row = ohlcv.at[idx, "timestamp"]
        entry_price = float(ohlcv.at[idx, "close"])
        if not pd.notna(entry_price):
            logging.warning("Skipping %s due to NaN close", entry_ts_row)
            continue

        forward_window = extract_forward_window(ohlcv, idx, horizon)
        if len(forward_window) < min_forward:
            logging.warning(
                "Skipping %s: insufficient forward data (%d < %d)",
                entry_ts_row,
                len(forward_window),
                min_forward,
            )
            continue

        targets = generate_targets_for_row(forward_window=forward_window, entry_price=entry_price, config=config)
        value = targets.get(target_name)
        if value is None:
            logging.warning("Target %s not produced for %s", target_name, entry_ts_row)
            continue
        records.append(
            {
                "timestamp": entry_ts_row,
                "target_name": target_name,
                "y_true": float(value),
                "horizon_bars": horizon,
                "target_label": target_label,
                "feature_key": args.feature_key,
            }
        )

    if not records:
        raise RuntimeError("No targets were generated; check data coverage and parameters")

    output_df = pd.DataFrame(records)
    output_df.sort_values("timestamp", inplace=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    print(f"Wrote {len(output_df)} target rows to {args.output_csv}")


if __name__ == "__main__":
    main()
