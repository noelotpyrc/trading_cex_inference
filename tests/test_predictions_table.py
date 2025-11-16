#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def _proj_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = _proj_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        import duckdb  # noqa: F401  # Ensure dependency available for test
    except ModuleNotFoundError:
        print('SKIP: duckdb module not installed')
        return

    try:
        import pandas as pd
    except ModuleNotFoundError:
        print('SKIP: pandas module not installed')
        return

    from run.predictions_table import (
        PredictionRow,
        connect as predictions_connect,
        describe_table,
        fetch_recent,
        ensure_table,
        insert_predictions,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'predictions_test.duckdb'

        # ensure_table should be idempotent and create expected schema
        with predictions_connect(db_path) as con:
            ensure_table(con)
            ensure_table(con)
            schema_df = describe_table(con)
            assert list(schema_df['column_name']) == ['ts', 'y_pred', 'model_path', 'feature_key', 'created_at']

        # insert_predictions should accept alias payload keys and persist rows
        rows = [
            PredictionRow.from_payload({
                'timestamp': '2025-01-01 00:00:00',
                'y_pred': 0.123,
                'model_path': 'models/run/modelA/model.txt',
                'feature_key': 'snapshot_A',
            }),
            PredictionRow.from_payload({
                'ts': pd.Timestamp('2025-01-01 01:00:00'),
                'y_pred': 0.456,
                'model_run': 'models/run/modelA/model.txt',
                'feature_key': 'snapshot_A',
            }),
        ]
        assert rows[0].ts.tzinfo is None
        assert rows[1].ts.tzinfo is None

        with predictions_connect(db_path) as con:
            inserted = insert_predictions(con, rows)
            assert inserted == len(rows)
            recent = fetch_recent(con, limit=5)

        assert len(recent) == 2
        # Fetch is ordered by timestamp descending
        assert pd.Timestamp(recent.iloc[0]['ts']) >= pd.Timestamp(recent.iloc[1]['ts'])
        assert set(recent['model_path']) == {'models/run/modelA/model.txt'}
        assert set(recent['feature_key']) == {'snapshot_A'}
        assert set(recent['y_pred'].round(3)) == {0.123, 0.456}

    print('predictions_table tests OK')


if __name__ == '__main__':
    main()
