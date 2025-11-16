#!/usr/bin/env python3
"""Inspect lookback pickle snapshots for a specific timestamp across two folders."""

import argparse
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:  # the lookback pickles contain pandas DataFrames
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover - simple runtime guard
    raise SystemExit(
        "pandas is required; install it or run the script with the project virtualenv"
    ) from exc

import numpy as np

TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
KEY_FMT = "%Y%m%d_%H%M%S"
DEFAULT_TIMEFRAMES = ["1H", "4H", "12H", "1D"]
PRIMARY_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class Snapshot:
    path: Path
    found: bool
    dataframe: Optional[pd.DataFrame]
    available_keys: Sequence[str]
    nearest_key: Optional[str]

    @property
    def shape(self) -> Optional[Tuple[int, int]]:
        return None if self.dataframe is None else self.dataframe.shape

    @property
    def columns(self) -> List[str]:
        return [] if self.dataframe is None else list(self.dataframe.columns)

    @property
    def index_range(self) -> Optional[Tuple[str, str]]:
        if self.dataframe is None or self.dataframe.empty:
            return None
        first = _format_timestamp_repr(self.dataframe.index[0])
        last = _format_timestamp_repr(self.dataframe.index[-1])
        return first, last

    @property
    def index_name(self) -> Optional[str]:
        return None if self.dataframe is None else self.dataframe.index.name


def _format_timestamp_repr(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        if value.tz is not None:
            return value.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z")
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def parse_timestamp(raw: str) -> datetime:
    try:
        return datetime.strptime(raw, TIMESTAMP_FMT)
    except ValueError as exc:
        raise SystemExit(f"Timestamp '{raw}' does not match format {TIMESTAMP_FMT}") from exc


def format_key(dt: datetime) -> str:
    return dt.strftime(KEY_FMT)


def load_snapshot(directory: Path, timeframe: str, key: str, dt: datetime) -> Snapshot:
    path = directory / f"lookbacks_{timeframe}.pkl"
    if not path.exists():
        return Snapshot(path=path, found=False, dataframe=None, available_keys=[], nearest_key=None)

    with path.open("rb") as handle:
        payload = pickle.load(handle)

    rows = payload.get("rows", {})
    found_df = rows.get(key)
    nearest_key = key if key in rows else _locate_nearest_key(rows.keys(), dt)
    available_keys = list(rows.keys())
    return Snapshot(
        path=path,
        found=found_df is not None,
        dataframe=found_df,
        available_keys=available_keys,
        nearest_key=nearest_key,
    )


def _locate_nearest_key(keys: Iterable[str], target: datetime) -> Optional[str]:
    best: Optional[Tuple[float, str]] = None
    for key in keys:
        try:
            key_dt = datetime.strptime(key, KEY_FMT)
        except ValueError:
            continue
        delta = abs((key_dt - target).total_seconds())
        if best is None or delta < best[0]:
            best = (delta, key)
    return None if best is None else best[1]


def summarize_snapshot(label: str, snapshot: Snapshot, dt: datetime) -> List[str]:
    lines: List[str] = [f"{label} -> {snapshot.path}"]
    if not snapshot.path.exists():
        lines.append("  file missing")
        return lines
    if not snapshot.found:
        lines.append("  timestamp not present")
        if snapshot.nearest_key:
            nearest_dt = datetime.strptime(snapshot.nearest_key, KEY_FMT)
            lines.append(
                f"  nearest available key: {snapshot.nearest_key} ("
                f"{nearest_dt.strftime(TIMESTAMP_FMT)})"
            )
        return lines

    df = snapshot.dataframe
    assert df is not None
    shape_str = f"{df.shape[0]} x {df.shape[1]}"
    lines.append(f"  shape: {shape_str}")
    columns = snapshot.columns
    if columns:
        preview = ", ".join(columns[:8])
        if len(columns) > 8:
            preview += f", ... ({len(columns)} total)"
        lines.append(f"  columns: {preview}")
    if snapshot.index_range:
        start, end = snapshot.index_range
        lines.append(
            f"  index ({snapshot.index_name or 'index'}): {start} -> {end}"
        )
    summary_cols = _select_summary_columns(df.columns)
    last_rows = df.iloc[[-1]][summary_cols]
    rendered = last_rows.to_string(index=True)
    lines.append("  last row (selected columns):")
    for line in rendered.splitlines():
        lines.append("    " + line)
    return lines


def _select_summary_columns(columns: Sequence[str]) -> List[str]:
    primary = [col for col in PRIMARY_COLUMNS if col in columns]
    if primary:
        return primary
    limit = min(len(columns), 5)
    return list(columns[:limit])


def render_common_difference(snapshot_a: Snapshot, snapshot_b: Snapshot) -> List[str]:
    if snapshot_a.dataframe is None or snapshot_b.dataframe is None:
        return []
    df_a = snapshot_a.dataframe
    df_b = snapshot_b.dataframe
    columns = [
        col for col in PRIMARY_COLUMNS if col in df_a.columns and col in df_b.columns
    ]
    if not columns:
        return []
    last_a = df_a.iloc[-1][columns]
    last_b = df_b.iloc[-1][columns]
    deltas = last_a - last_b
    output = ["  common column deltas (A - B) on final row:"]
    for column in columns:
        output.append(f"    {column}: {deltas[column]:.10g}")
    return output


def _standardize_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    result = df.copy()
    result.index = idx
    return result


def _compute_obv_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if not {"close", "volume"}.issubset(df.columns):
        return pd.Series(dtype=float)
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    if len(close) < 1:
        return pd.Series(dtype=float)
    obv = np.zeros(len(close), dtype=float)
    for i in range(1, len(close)):
        if pd.isna(close.iloc[i]) or pd.isna(close.iloc[i - 1]):
            obv[i] = obv[i - 1]
            continue
        if close.iloc[i] > close.iloc[i - 1]:
            obv[i] = obv[i - 1] + float(volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv[i] = obv[i - 1] - float(volume.iloc[i])
        else:
            obv[i] = obv[i - 1]
    return pd.Series(obv, index=df.index, name="obv")


def render_obv_divergence(snapshot_a: Snapshot, snapshot_b: Snapshot) -> List[str]:
    if snapshot_a.dataframe is None or snapshot_b.dataframe is None:
        return []
    needed = {"close", "volume"}
    if not needed.issubset(snapshot_a.dataframe.columns) or not needed.issubset(snapshot_b.dataframe.columns):
        return []

    df_a = _standardize_index(snapshot_a.dataframe)
    df_b = _standardize_index(snapshot_b.dataframe)

    obv_a = _compute_obv_series(df_a)
    obv_b = _compute_obv_series(df_b)
    if obv_a.empty or obv_b.empty:
        return []

    aligned = pd.DataFrame({"A": obv_a, "B": obv_b}).dropna()
    if aligned.empty:
        return []

    diff_mask = (aligned["A"] - aligned["B"]).abs() > 1e-6
    if not bool(diff_mask.any()):
        return ["  OBV sequences match across the lookback window."]

    first_idx = aligned.index[diff_mask][0]
    delta = aligned.loc[first_idx, "A"] - aligned.loc[first_idx, "B"]

    lines = ["  OBV divergence detected:"]
    lines.append(
        f"    first mismatch at {first_idx}: A={aligned.loc[first_idx, 'A']:.10g} "
        f"B={aligned.loc[first_idx, 'B']:.10g} delta={delta:.10g}"
    )

    pos = aligned.index.get_indexer_for([first_idx])[0]
    prev_idx = aligned.index[pos - 1] if pos > 0 else None
    if prev_idx is not None:
        prev_delta = aligned.loc[prev_idx, "A"] - aligned.loc[prev_idx, "B"]
        lines.append(
            f"    previous {prev_idx}: A={aligned.loc[prev_idx, 'A']:.10g} "
            f"B={aligned.loc[prev_idx, 'B']:.10g} delta={prev_delta:.10g}"
        )

    def _format_row(df: pd.DataFrame, idx: pd.Timestamp) -> str:
        row = df.loc[idx]
        return f"close={row['close']:.10g} volume={row['volume']:.10g}"

    lines.append(f"    A row: {_format_row(df_a, first_idx)}")
    lines.append(f"    B row: {_format_row(df_b, first_idx)}")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "timestamp",
        help="Timestamp to inspect, e.g. '2024-02-05 17:00:00'",
    )
    parser.add_argument(
        "dir_a",
        help="First lookbacks folder (often the raw feature build output)",
    )
    parser.add_argument(
        "dir_b",
        help="Second lookbacks folder (often the training pipeline source)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=DEFAULT_TIMEFRAMES,
        help="Timeframes to inspect (default: 1H 4H 12H 1D)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dt = parse_timestamp(args.timestamp)
    key = format_key(dt)
    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)

    print(f"Inspecting timestamp: {args.timestamp} -> key {key}")
    for timeframe in args.timeframes:
        print(f"\n=== Timeframe {timeframe} ===")
        snapshot_a = load_snapshot(dir_a, timeframe, key, dt)
        snapshot_b = load_snapshot(dir_b, timeframe, key, dt)

        for line in summarize_snapshot("A", snapshot_a, dt):
            print(line)
        for line in summarize_snapshot("B", snapshot_b, dt):
            print(line)

        diff_lines = render_common_difference(snapshot_a, snapshot_b)
        for line in diff_lines:
            print(line)

        obv_lines = render_obv_divergence(snapshot_a, snapshot_b)
        for line in obv_lines:
            print(line)


if __name__ == "__main__":
    main()
