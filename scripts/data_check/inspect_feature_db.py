#!/usr/bin/env python3
"""
One-off script to inspect DuckDB features table.
Checks:
1. If features table has all columns from the JSON config
2. Time window of the features table
"""

import json
import duckdb
from pathlib import Path

# Paths
DB_PATH = "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb"
JSON_PATH = "configs/feature_lists/binance_btcusdt_p60_selected_trial1.json"

def main():
    # Load expected features from JSON
    print(f"Loading expected features from: {JSON_PATH}")
    with open(JSON_PATH, 'r') as f:
        expected_features = json.load(f)
    print(f"Expected features count: {len(expected_features)}")

    # Connect to DuckDB
    print(f"\nConnecting to DuckDB: {DB_PATH}")
    con = duckdb.connect(DB_PATH, read_only=True)

    # Get table columns
    print("\nQuerying features table schema...")
    result = con.execute("DESCRIBE features").fetchall()
    db_columns = [row[0] for row in result]

    print(f"Database columns count: {len(db_columns)}")
    print(f"Table columns: {db_columns}")

    # The features are stored in a nested structure, let's inspect it
    print("\nChecking if features are in nested column...")
    sample_row = con.execute("SELECT features FROM features LIMIT 1").fetchone()

    if sample_row:
        features_col = sample_row[0]
        print(f"Features column type: {type(features_col)}")

        # Extract keys from the features column
        if isinstance(features_col, dict):
            feature_keys = list(features_col.keys())
        else:
            # Try to get keys from the struct/JSON
            try:
                keys_query = """
                SELECT DISTINCT json_keys(features) as keys
                FROM features
                LIMIT 1
                """
                keys_result = con.execute(keys_query).fetchone()
                if keys_result:
                    feature_keys = keys_result[0]
                else:
                    feature_keys = []
            except:
                # Alternative: try to extract the first row as dict
                try:
                    dict_query = "SELECT features::VARCHAR FROM features LIMIT 1"
                    result = con.execute(dict_query).fetchone()
                    import json as json_lib
                    feature_keys = list(json_lib.loads(result[0]).keys())
                except:
                    feature_keys = []

        print(f"Number of features in nested column: {len(feature_keys)}")
        print(f"First few feature keys: {feature_keys[:10]}")
    else:
        feature_keys = []
        print("No data in features table")

    # Check which expected features are present
    missing_features = []
    present_features = []

    for feature in expected_features:
        if feature in feature_keys:
            present_features.append(feature)
        else:
            missing_features.append(feature)

    # Report results
    print("\n" + "="*60)
    print("FEATURE COMPARISON RESULTS")
    print("="*60)
    print(f"Total expected features: {len(expected_features)}")
    print(f"Features present in DB: {len(present_features)}")
    print(f"Missing features: {len(missing_features)}")

    if missing_features:
        print("\nMissing features:")
        for feat in missing_features:
            print(f"  - {feat}")
    else:
        print("\n✓ All features from JSON are present in the database!")

    # Get time window
    print("\n" + "="*60)
    print("TIME WINDOW")
    print("="*60)

    time_query = """
    SELECT
        MIN(ts) as min_time,
        MAX(ts) as max_time,
        COUNT(*) as row_count
    FROM features
    """

    time_result = con.execute(time_query).fetchone()

    if time_result:
        min_time, max_time, row_count = time_result
        print(f"Start time: {min_time}")
        print(f"End time:   {max_time}")
        print(f"Total rows: {row_count:,}")

        # Calculate duration
        if min_time and max_time:
            duration = max_time - min_time
            print(f"Duration:   {duration}")

    # Additional info: get sample row
    print("\n" + "="*60)
    print("SAMPLE DATA (first row)")
    print("="*60)

    sample = con.execute("SELECT * FROM features LIMIT 1").fetchone()
    sample_columns = [desc[0] for desc in con.description]

    print(f"First few columns with values:")
    for i in range(min(10, len(sample_columns))):
        print(f"  {sample_columns[i]}: {sample[i]}")

    con.close()
    print("\n✓ Inspection complete!")

if __name__ == "__main__":
    main()
