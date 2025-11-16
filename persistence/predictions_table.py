#!/usr/bin/env python3
"""Utilities for creating and inspecting the DuckDB predictions table."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import duckdb  # type: ignore
import pandas as pd

TABLE_NAME = "predictions"
CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    ts TIMESTAMP NOT NULL,
    y_pred DOUBLE NOT NULL,
    model_path VARCHAR NOT NULL,
    feature_key VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


@contextmanager
def connect(db_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    con = duckdb.connect(str(db_path))
    try:
        con.execute("SET TimeZone='UTC';")
        yield con
    finally:
        con.close()


def ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(CREATE_TABLE_SQL)
    cols = con.execute(f"PRAGMA table_info('{TABLE_NAME}')").fetchall()
    column_names = {row[1] for row in cols}
    if 'feature_key' not in column_names:
        raise RuntimeError(
            "predictions table missing 'feature_key' column; please drop/recreate the table to upgrade schema"
        )
    # Enforce uniqueness for (ts, model_path, feature_key) via a unique index.
    # If duplicates already exist, index creation will fail; surface a helpful error.
    try:
        con.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_pred_unique ON {TABLE_NAME}(ts, model_path, feature_key)"
        )
    except Exception:
        # Check for duplicates and raise a clearer message
        dup_q = f"""
            SELECT ts, model_path, feature_key, COUNT(*) AS c
            FROM {TABLE_NAME}
            GROUP BY 1,2,3
            HAVING COUNT(*) > 1
            ORDER BY c DESC
            LIMIT 5
        """
        try:
            dups = con.execute(dup_q).fetch_df()
        except Exception:
            dups = None
        if dups is not None and not dups.empty:
            raise RuntimeError(
                "Cannot enforce (ts, model_path, feature_key) uniqueness: duplicates exist. "
                f"Sample duplicates:\n{dups.to_string(index=False)}"
            )
        raise


@dataclass
class PredictionRow:
    ts: pd.Timestamp
    y_pred: float
    model_path: str
    feature_key: str

    @classmethod
    def from_payload(cls, payload: dict) -> "PredictionRow":
        ts = payload.get("ts") or payload.get("timestamp")
        if ts is None:
            raise ValueError("timestamp/ts required in payload")
        ts_parsed = pd.Timestamp(ts).tz_localize(None)
        y_pred = float(payload.get("y_pred"))
        model_path = payload.get("model_path") or payload.get("model_run")
        if not model_path:
            raise ValueError("model_path/model_run required in payload")
        feature_key = payload.get("feature_key")
        if not feature_key:
            raise ValueError("feature_key required in payload")
        return cls(ts=ts_parsed, y_pred=y_pred, model_path=str(model_path), feature_key=str(feature_key))


def insert_predictions(con: duckdb.DuckDBPyConnection, rows: Iterable[PredictionRow]) -> int:
    ensure_table(con)
    data = [(row.ts.to_pydatetime(), row.y_pred, row.model_path, row.feature_key) for row in rows]
    if not data:
        return 0
    con.executemany(
        f"INSERT INTO {TABLE_NAME} (ts, y_pred, model_path, feature_key) VALUES (?, ?, ?, ?)",
        data,
    )
    return len(data)


def fetch_recent(con: duckdb.DuckDBPyConnection, limit: int) -> pd.DataFrame:
    ensure_table(con)
    query = f"SELECT ts, y_pred, model_path, feature_key, created_at FROM {TABLE_NAME} ORDER BY ts DESC LIMIT ?"
    return con.execute(query, [limit]).fetch_df()


def table_stats(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    ensure_table(con)
    query = f"""
        SELECT COUNT(*) AS n_rows,
               MIN(ts) AS ts_min,
               MAX(ts) AS ts_max,
               MIN(created_at) AS created_at_min,
               MAX(created_at) AS created_at_max
        FROM {TABLE_NAME}
    """
    return con.execute(query).fetch_df()


def describe_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    ensure_table(con)
    return con.execute(f"DESCRIBE {TABLE_NAME}").fetch_df()


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Manage the predictions table")
    parser.add_argument("--duckdb", required=True, type=Path, help="Path to DuckDB file")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create the predictions table if it is missing")

    stats_p = sub.add_parser("stats", help="Show row counts and timestamp range")
    stats_p.add_argument("--limit", type=int, default=5, help="Show last N rows after stats")

    head_p = sub.add_parser("head", help="Print recent prediction rows")
    head_p.add_argument("--limit", type=int, default=10, help="Number of rows to display")

    sub.add_parser("schema", help="Show table schema")

    args = parser.parse_args(argv)

    with connect(args.duckdb) as con:
        if args.command == "init":
            ensure_table(con)
            print(f"ensured {TABLE_NAME} exists at {args.duckdb}")
        elif args.command == "schema":
            df = describe_table(con)
            print(df.to_string(index=False))
        elif args.command == "stats":
            stats_df = table_stats(con)
            print(stats_df.to_string(index=False))
            head_df = fetch_recent(con, limit=args.limit)
            if not head_df.empty:
                print("\nRecent predictions:")
                print(head_df.to_string(index=False))
        elif args.command == "head":
            df = fetch_recent(con, limit=args.limit)
            if df.empty:
                print("No rows in predictions table")
            else:
                print(df.to_string(index=False))
        else:
            parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
