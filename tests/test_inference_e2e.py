#!/usr/bin/env python3
"""
End-to-end test for backfill_inference.py using production databases.

Tests running inference on pre-computed features for a small date range.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd


def run_e2e_test():
    """Run end-to-end test for inference backfilling."""

    # Test configuration
    OHLCV_DB = "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb"
    FEATURE_DB = "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb"
    PRED_DB_TEST = "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_prediction_test.duckdb"
    TABLE = "ohlcv_btcusdt_1h"
    MODEL_PATH = "/Volumes/Extreme SSD/trading_data/cex/models/run/binance_btcusdt_perp_1h/y_logret_168h"
    DATASET = "e2e_test_inference"
    START = "2025-11-13 00:00:00"
    END = "2025-11-14 00:00:00"

    print("=" * 80)
    print("E2E Test: backfill_inference.py")
    print("=" * 80)
    print(f"OHLCV DB:     {OHLCV_DB}")
    print(f"Feature DB:   {FEATURE_DB}")
    print(f"Pred DB:      {PRED_DB_TEST}")
    print(f"Table:        {TABLE}")
    print(f"Model path:   {MODEL_PATH}")
    print(f"Dataset:      {DATASET}")
    print(f"Date range:   {START} to {END}")
    print()

    # Step 1: Check model exists
    print("Step 1: Checking model exists...")
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"❌ FAILED: Model not found at {MODEL_PATH}")
        return False

    # Check for model file in directory
    if model_path.is_dir():
        model_files = list(model_path.glob("*.txt")) + list(model_path.glob("*.lgb"))
        if not model_files:
            print(f"❌ FAILED: No model files found in {MODEL_PATH}")
            return False
        print(f"✓ Found model file: {model_files[0].name}")
    else:
        print(f"✓ Found model file: {model_path.name}")

    print()

    # Step 2: Check features exist for target range
    print("Step 2: Checking features exist for target range...")
    try:
        con_feat = duckdb.connect(FEATURE_DB, read_only=True)
        con_feat.execute("SET TimeZone='UTC';")

        result = con_feat.execute("""
            SELECT COUNT(DISTINCT ts), MIN(ts), MAX(ts)
            FROM features
            WHERE ts BETWEEN ? AND ?
        """, [START, END]).fetchone()

        if result[0] == 0:
            print(f"❌ FAILED: No features found in range {START} to {END}")
            print("   Please run backfill_features.py first")
            con_feat.close()
            return False

        print(f"✓ Found features for {result[0]} unique timestamps from {result[1]} to {result[2]}")

        # Check how many feature keys exist
        keys = con_feat.execute("""
            SELECT DISTINCT feature_key, COUNT(*) as cnt
            FROM features
            WHERE ts BETWEEN ? AND ?
            GROUP BY feature_key
            ORDER BY feature_key
        """, [START, END]).fetchall()

        print(f"✓ Feature keys in range: {len(keys)}")
        for key, cnt in keys[:5]:
            print(f"  - {key}: {cnt} rows")
        if len(keys) > 5:
            print(f"  ... and {len(keys) - 5} more")

        con_feat.close()
    except Exception as e:
        print(f"❌ FAILED: Error checking features: {e}")
        return False

    print()

    # Step 3: Check OHLCV data exists
    print("Step 3: Checking OHLCV data exists...")
    try:
        con_ohlcv = duckdb.connect(OHLCV_DB, read_only=True)
        con_ohlcv.execute("SET TimeZone='UTC';")

        result = con_ohlcv.execute(f"""
            SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM {TABLE}
            WHERE timestamp BETWEEN ? AND ?
        """, [START, END]).fetchone()

        if result[0] == 0:
            print(f"❌ FAILED: No OHLCV data found in range {START} to {END}")
            con_ohlcv.close()
            return False

        print(f"✓ Found {result[0]} OHLCV rows from {result[1]} to {result[2]}")
        con_ohlcv.close()
    except Exception as e:
        print(f"❌ FAILED: Error checking OHLCV data: {e}")
        return False

    print()

    # Step 4: Clean up any existing test predictions
    print("Step 4: Cleaning up any existing test predictions...")
    try:
        con_pred = duckdb.connect(PRED_DB_TEST)
        con_pred.execute("SET TimeZone='UTC';")

        # Check if predictions table exists
        table_exists = con_pred.execute("""
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'predictions' LIMIT 1
        """).fetchone()

        if table_exists:
            count_before = con_pred.execute(
                "SELECT COUNT(*) FROM predictions WHERE dataset = ?",
                [DATASET]
            ).fetchone()[0]

            if count_before > 0:
                con_pred.execute(
                    "DELETE FROM predictions WHERE dataset = ?",
                    [DATASET]
                )
                print(f"✓ Deleted {count_before} existing test predictions")
            else:
                print("✓ No existing test predictions to clean up")
        else:
            print("✓ Predictions table doesn't exist yet (will be created)")

        con_pred.close()
    except Exception as e:
        print(f"❌ FAILED: Error cleaning up test predictions: {e}")
        return False

    print()

    # Step 5: Run backfill_inference.py
    print("Step 5: Running backfill_inference.py...")
    print()

    import subprocess
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "backfill_inference.py"),
        "--duckdb", OHLCV_DB,
        "--feat-duckdb", FEATURE_DB,
        "--pred-duckdb", PRED_DB_TEST,
        "--table", TABLE,
        "--model-path", MODEL_PATH,
        "--mode", "window",
        "--start", START,
        "--end", END,
        "--dataset", DATASET,
    ]

    print("Command:", " ".join(cmd))
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print(f"❌ FAILED: backfill_inference.py exited with code {result.returncode}")
        return False

    print()

    # Step 6: Verify predictions were written
    print("Step 6: Verifying predictions were written...")
    try:
        con_pred = duckdb.connect(PRED_DB_TEST, read_only=True)
        con_pred.execute("SET TimeZone='UTC';")

        # Count predictions written
        result = con_pred.execute("""
            SELECT COUNT(*), MIN(ts), MAX(ts)
            FROM predictions
            WHERE dataset = ?
        """, [DATASET]).fetchone()

        if result[0] == 0:
            print(f"❌ FAILED: No predictions were written for dataset={DATASET}")
            con_pred.close()
            return False

        print(f"✓ Found {result[0]} prediction rows from {result[1]} to {result[2]}")

        # Sample one prediction row to check structure
        sample = con_pred.execute("""
            SELECT ts, y_pred, model_path, feature_key, dataset, created_at
            FROM predictions
            WHERE dataset = ?
            ORDER BY ts
            LIMIT 1
        """, [DATASET]).fetchone()

        if sample:
            ts, y_pred, model_path_str, feature_key, dataset, created_at = sample
            print(f"✓ Sample timestamp: {ts}")
            print(f"✓ Sample prediction: {y_pred:.6f}")
            print(f"✓ Model path: {model_path_str}")
            print(f"✓ Feature key: {feature_key or 'most_recent'}")
            print(f"✓ Dataset: {dataset}")
            print(f"✓ Created at: {created_at}")

            # Check prediction is numeric
            if not isinstance(y_pred, (int, float)):
                print(f"⚠️  WARNING: Prediction is not numeric: {type(y_pred)}")
            elif pd.isna(y_pred):
                print(f"⚠️  WARNING: Prediction is NaN")
            else:
                print(f"✓ Prediction is valid numeric value")

        con_pred.close()
    except Exception as e:
        print(f"❌ FAILED: Error verifying predictions: {e}")
        return False

    print()

    # Step 7: Compare with production predictions (if they exist)
    print("Step 7: Comparing with production predictions (if available)...")
    PROD_PRED_DB = "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_prediction.duckdb"

    # Resolve model path to model.txt (matching backfill_inference.py behavior)
    model_path_resolved = Path(MODEL_PATH)
    if model_path_resolved.is_dir():
        model_path_resolved = model_path_resolved / "model.txt"

    try:
        con_prod = duckdb.connect(PROD_PRED_DB, read_only=True)
        con_prod.execute("SET TimeZone='UTC';")

        # Check if predictions exist for same model and time range (using resolved path)
        prod_preds = con_prod.execute("""
            SELECT ts, y_pred
            FROM predictions
            WHERE model_path = ?
            AND ts BETWEEN ? AND ?
            ORDER BY ts
        """, [str(model_path_resolved), START, END]).fetchall()

        if prod_preds:
            print(f"✓ Found {len(prod_preds)} production predictions for comparison")

            # Load test predictions
            con_test = duckdb.connect(PRED_DB_TEST, read_only=True)
            con_test.execute("SET TimeZone='UTC';")

            test_preds = con_test.execute("""
                SELECT ts, y_pred
                FROM predictions
                WHERE dataset = ?
                ORDER BY ts
            """, [DATASET]).fetchall()

            # Compare predictions
            if len(test_preds) != len(prod_preds):
                print(f"⚠️  WARNING: Prediction count mismatch - test:{len(test_preds)} vs prod:{len(prod_preds)}")

            # Compare values for matching timestamps
            prod_dict = {pd.Timestamp(ts): pred for ts, pred in prod_preds}
            test_dict = {pd.Timestamp(ts): pred for ts, pred in test_preds}

            common_ts = set(prod_dict.keys()) & set(test_dict.keys())
            if common_ts:
                print(f"✓ Comparing {len(common_ts)} common timestamps...")
                max_abs_diff = 0.0
                max_rel_diff = 0.0
                mismatches = []

                for ts in sorted(common_ts):
                    prod_val = prod_dict[ts]
                    test_val = test_dict[ts]

                    diff = abs(float(test_val) - float(prod_val))
                    max_abs_diff = max(max_abs_diff, diff)

                    if abs(float(prod_val)) > 1e-10:
                        rel_diff = diff / abs(float(prod_val))
                        max_rel_diff = max(max_rel_diff, rel_diff)

                        # Flag significant differences (>0.1% relative or >1e-6 absolute)
                        if rel_diff > 0.001 or diff > 1e-6:
                            mismatches.append(f"{ts}: test={test_val:.8f} prod={prod_val:.8f} (rel_diff={rel_diff:.2e})")

                if mismatches:
                    print(f"⚠️  WARNING: {len(mismatches)} predictions have value differences:")
                    for msg in mismatches[:10]:
                        print(f"    {msg}")
                    if len(mismatches) > 10:
                        print(f"    ... and {len(mismatches) - 10} more")
                else:
                    print(f"✓ All prediction values match (max_abs_diff={max_abs_diff:.2e}, max_rel_diff={max_rel_diff:.2e})")

            con_test.close()
        else:
            print("ℹ️  No production predictions found for same model and time range")

        con_prod.close()
    except Exception as e:
        print(f"⚠️  WARNING: Error comparing with production: {e}")

    print()

    # Step 8: Cleanup
    print("Step 8: Cleanup...")
    try:
        response = input(f"Delete test predictions with dataset='{DATASET}'? [y/N]: ")
        if response.lower() == 'y':
            try:
                con_pred = duckdb.connect(PRED_DB_TEST)
                con_pred.execute("SET TimeZone='UTC';")
                con_pred.execute(
                    "DELETE FROM predictions WHERE dataset = ?",
                    [DATASET]
                )
                con_pred.close()
                print(f"✓ Deleted test predictions")
            except Exception as e:
                print(f"⚠️  WARNING: Error deleting test predictions: {e}")
        else:
            print(f"ℹ️  Test predictions kept (dataset='{DATASET}')")
    except (EOFError, KeyboardInterrupt):
        # Non-interactive mode or user interrupted
        print(f"ℹ️  Skipping cleanup (non-interactive mode)")
        print(f"ℹ️  Test predictions kept with dataset='{DATASET}'")

    print()
    print("=" * 80)
    print("✓ E2E Test PASSED")
    print("=" * 80)
    return True


if __name__ == "__main__":
    try:
        success = run_e2e_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ FAILED: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
