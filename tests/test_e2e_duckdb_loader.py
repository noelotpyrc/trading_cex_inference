#!/usr/bin/env python3
from __future__ import annotations

"""
E2E test for load_ohlcv_duckdb using the same CSV data as the DB-vs-CSV
comparison. Loads OHLCV from DuckDB over the CSV's time range and prints
randomly sampled rows from both sources for visual comparison.

Skips gracefully if the DuckDB path is not available or no overlapping rows
exist between CSV and DB.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _proj_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = _proj_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from run.data_loader import load_ohlcv_csv, load_ohlcv_duckdb

    parser = argparse.ArgumentParser(description='E2E: load OHLCV from DuckDB and compare with CSV rows')
    parser.add_argument('--csv-path', default=str(root / 'data' / 'BINANCE_BTCUSDT.P, 60.csv'), help='Path to 1H OHLCV CSV')
    parser.add_argument('--duckdb', default='/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb', help='Path to DuckDB file')
    parser.add_argument('--table', default='ohlcv_btcusdt_1h', help='DuckDB table name for OHLCV')
    parser.add_argument('--count', type=int, default=2, help='How many timestamps to sample for printing')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()

    db_path = Path(args.duckdb)
    if not db_path.exists():
        print(f'SKIP: DuckDB path not found: {db_path}')
        return

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f'SKIP: CSV path not found: {csv_path}')
        return

    # Load CSV and DB within CSV range
    df_csv = load_ohlcv_csv(str(csv_path))
    assert {'timestamp','open','high','low','close','volume'}.issubset(df_csv.columns)
    if df_csv.empty:
        print('SKIP: CSV is empty')
        return
    csv_min, csv_max = df_csv['timestamp'].min(), df_csv['timestamp'].max()

    df_db = load_ohlcv_duckdb(str(db_path), table=args.table, start=csv_min, end=csv_max)
    # Basic validations for the loader contract when data exists
    if not df_db.empty:
        assert list(df_db.columns) == ['timestamp','open','high','low','close','volume']
        assert df_db['timestamp'].is_monotonic_increasing
        assert pd.api.types.is_datetime64_any_dtype(df_db['timestamp'])

    print(f'CSV rows={len(df_csv)} range=[{csv_min}..{csv_max}]')
    print(f'DB  rows={len(df_db)} (filtered to CSV range)')

    # Intersection of timestamps for fair comparison
    ts_csv = set(df_csv['timestamp'])
    ts_db = set(df_db['timestamp']) if not df_db.empty else set()
    common_ts = sorted(ts_csv.intersection(ts_db))

    rng = np.random.default_rng(int(args.seed))
    k = max(1, int(args.count))
    if common_ts:
        idxs = rng.choice(len(common_ts), size=min(k, len(common_ts)), replace=False)
        sample_ts = [common_ts[i] for i in sorted(idxs.tolist())]
    else:
        # No overlap: sample from CSV only to show missing in DB
        idxs = rng.choice(len(df_csv), size=min(k, len(df_csv)), replace=False)
        sample_ts = [pd.Timestamp(df_csv.iloc[i]['timestamp']) for i in sorted(idxs.tolist())]

    csv_map = df_csv.set_index('timestamp')[['open','high','low','close','volume']]
    db_map = df_db.set_index('timestamp')[['open','high','low','close','volume']] if not df_db.empty else pd.DataFrame()

    for ts in sample_ts:
        print(f'==== {ts} ====')
        csv_row = csv_map.loc[ts] if ts in csv_map.index else None
        if ts in (db_map.index if not db_map.empty else []):
            db_row = db_map.loc[ts]
            diffs = (db_row.astype(float) - csv_row.astype(float)).abs() if csv_row is not None else None
            print('DB :', f"open={db_row['open']:.8f}", f"high={db_row['high']:.8f}", f"low={db_row['low']:.8f}", f"close={db_row['close']:.8f}", f"volume={db_row['volume']:.6f}")
        else:
            db_row = None
            print('DB : NOT FOUND')
        if csv_row is not None:
            print('CSV:', f"open={csv_row['open']:.8f}", f"high={csv_row['high']:.8f}", f"low={csv_row['low']:.8f}", f"close={csv_row['close']:.8f}", f"volume={csv_row['volume']:.6f}")
        else:
            print('CSV: NOT FOUND')
        if db_row is not None and csv_row is not None:
            print('DIFF:', f"open={diffs['open']:.10f}", f"high={diffs['high']:.10f}", f"low={diffs['low']:.10f}", f"close={diffs['close']:.10f}", f"volume={diffs['volume']:.10f}")

    print('e2e duckdb loader test OK')


if __name__ == '__main__':
    main()

