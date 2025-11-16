#!/usr/bin/env python3
from __future__ import annotations

import json
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
        import duckdb  # noqa: F401
    except ModuleNotFoundError:
        print('SKIP: duckdb module not installed')
        return

    try:
        import pandas as pd
    except ModuleNotFoundError:
        print('SKIP: pandas module not installed')
        return

    from run.features_table import (
        FeatureRow,
        connect as features_connect,
        describe_table,
        ensure_table,
        fetch_recent,
        upsert_feature_rows,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'features_test.duckdb'
        feature_key = 'binance_btcusdt_perp_1h__test'

        df = pd.DataFrame([
            {
                'timestamp': pd.Timestamp('2025-03-21 05:00:00', tz='UTC'),
                'feat_a': 0.1,
                'feat_b': 1.5,
            }
        ])

        row = FeatureRow.from_dataframe(feature_key, df, feature_columns=['feat_a', 'feat_b'])
        assert row.ts.tzinfo is None
        assert set(row.features.keys()) == {'feat_a', 'feat_b'}

        with features_connect(db_path) as con:
            ensure_table(con)
            ensure_table(con)
            upsert_feature_rows(con, [row])
            schema_df = describe_table(con)
            assert list(schema_df['column_name']) == ['feature_key', 'ts', 'features', 'created_at']
            recent = fetch_recent(con, feature_key, limit=5)

        assert len(recent) == 1
        assert recent.iloc[0]['feature_key'] == feature_key
        stored_json = recent.iloc[0]['features']
        stored_map = json.loads(stored_json)
        assert stored_map['feat_a'] == 0.1
        assert stored_map['feat_b'] == 1.5

    print('features_table tests OK')


if __name__ == '__main__':
    main()
