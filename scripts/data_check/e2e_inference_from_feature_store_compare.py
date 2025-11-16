#!/usr/bin/env python3
"""
E2E: Compare predictions from the new feature-store inference flow with
existing production predictions for the same model, and verify multi-model
coexistence (same ts + feature_key, different model_path) is supported.

Tests:
  1) Compare y_pred for a small set of timestamps between:
     - production predictions in an existing predictions table (read-only)
     - predictions produced by run/backfill_inference_from_feature_store.py
        into a temporary predictions DB
  2) Insert two rows with the same (ts, feature_key) but different model_path
     into a temporary predictions DB and confirm both coexist.

Example:
  python run/e2e_inference_from_feature_store_compare.py \
    --feat-duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb" \
    --prod-pred-duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_prediction.duckdb" \
    --temp-pred-duckdb "tmp/e2e_pred.duckdb" \
    --feature-key "prod_default" \
    --model-path "/Volumes/Extreme SSD/trading_data/cex/models/run/binance_btcusdt_perp_1h/y_logret_168h" \
    --limit 5 --tol 1e-9
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
import random
import uuid

import duckdb  # type: ignore
import pandas as pd

from run.backfill_inference_from_feature_store import InferenceConfig, run_inference
from run.model_io_lgbm import resolve_model_file
from run.predictions_table import ensure_table as ensure_predictions_table, insert_predictions, PredictionRow


def _resolve_model_path_str(model_path: str) -> str:
    mf, _ = resolve_model_file(model_path=model_path)
    return str(mf)


def _fetch_prod_preds(prod_db: Path, feature_key: str, model_file_str: str, limit: int) -> List[Tuple[pd.Timestamp, float]]:
    con = duckdb.connect(str(prod_db), read_only=True)
    try:
        con.execute("SET TimeZone='UTC';")
        q = """
            SELECT ts, y_pred
            FROM predictions
            WHERE feature_key = ? AND model_path = ?
            ORDER BY ts DESC
            LIMIT ?
        """
        df = con.execute(q, [feature_key, model_file_str, int(limit)]).fetch_df()
        out: List[Tuple[pd.Timestamp, float]] = []
        for _, r in df.iterrows():
            out.append((pd.Timestamp(r["ts"]), float(r["y_pred"])))
        return out
    finally:
        con.close()


def _fetch_all_prod_ts(prod_db: Path, feature_key: str, model_file_str: str) -> List[pd.Timestamp]:
    con = duckdb.connect(str(prod_db), read_only=True)
    try:
        con.execute("SET TimeZone='UTC';")
        q = """
            SELECT ts FROM predictions
            WHERE feature_key = ? AND model_path = ?
            ORDER BY ts DESC
        """
        df = con.execute(q, [feature_key, model_file_str]).fetch_df()
        return [pd.Timestamp(t) for t in df['ts']] if not df.empty else []
    finally:
        con.close()


def _fetch_preds(db: Path, feature_key: str, model_file_str: str, ts_list: List[pd.Timestamp]) -> List[Tuple[pd.Timestamp, float]]:
    if not ts_list:
        return []
    con = duckdb.connect(str(db), read_only=True)
    try:
        con.execute("SET TimeZone='UTC';")
        q = """
            SELECT ts, y_pred
            FROM predictions
            WHERE feature_key = ? AND model_path = ? AND ts IN ({})
            ORDER BY ts DESC
        """.format(
            ",".join(["?" for _ in ts_list])
        )
        params = [feature_key, model_file_str] + [pd.Timestamp(t).to_pydatetime() for t in ts_list]
        df = con.execute(q, params).fetch_df()
        out: List[Tuple[pd.Timestamp, float]] = []
        for _, r in df.iterrows():
            out.append((pd.Timestamp(r["ts"]), float(r["y_pred"])))
        return out
    finally:
        con.close()


def _compare_preds(prod: List[Tuple[pd.Timestamp, float]], new: List[Tuple[pd.Timestamp, float]], tol: float) -> Tuple[int, int, List[str]]:
    prod_map = {pd.Timestamp(ts): val for ts, val in prod}
    new_map = {pd.Timestamp(ts): val for ts, val in new}
    common = sorted(set(prod_map.keys()) & set(new_map.keys()))
    mismatches = []
    for ts in common:
        a = prod_map[ts]
        b = new_map[ts]
        if not pd.isna(a) and not pd.isna(b) and abs(a - b) > tol:
            mismatches.append(f"{ts}: prod={a} new={b} diff={abs(a-b)}")
    return len(common), len(mismatches), mismatches


def _insert_multi_models_from_results(tmp_db: Path, feature_key: str, ts_preds: List[Tuple[pd.Timestamp, float]]) -> Tuple[bool, str]:
    """Insert two additional model_path rows for each ts in ts_preds into the same DB.

    Ensures (ts, model_path, feature_key) are unique by deleting any prior rows for
    the chosen mock model_paths before inserting.
    """
    if not ts_preds:
        return False, "No timestamps provided for multi-model insert test"
    with duckdb.connect(str(tmp_db)) as con:
        con.execute("SET TimeZone='UTC';")
        ensure_predictions_table(con)
        # Collect existing model_path values to avoid collisions
        try:
            mp_df = con.execute("SELECT DISTINCT model_path FROM predictions").fetch_df()
            existing_mps = set(mp_df["model_path"].astype(str).tolist()) if not mp_df.empty else set()
        except Exception:
            existing_mps = set()
        # Generate two unique mock model paths that are guaranteed not to exist yet
        def _new_mock_path() -> str:
            while True:
                cand = f"/mock/e2e_model_{uuid.uuid4().hex}/model.txt"
                if cand not in existing_mps:
                    existing_mps.add(cand)
                    return cand
        mp_a = _new_mock_path()
        mp_b = _new_mock_path()
        rows = []
        for ts, y in ts_preds:
            rows.append(PredictionRow.from_payload({
                'timestamp': ts,
                'y_pred': y,
                'model_path': mp_a,
                'feature_key': feature_key,
            }))
            rows.append(PredictionRow.from_payload({
                'timestamp': ts,
                'y_pred': y,
                'model_path': mp_b,
                'feature_key': feature_key,
            }))
        insert_predictions(con, rows)
        # Verify that for each ts we now have 3 rows (original + 2 mock models)
        ok_all = True
        for ts, _ in ts_preds:
            df = con.execute(
                "SELECT COUNT(*) AS n FROM predictions WHERE ts = ? AND feature_key = ?",
                [pd.Timestamp(ts).to_pydatetime(), feature_key],
            ).fetch_df()
            n = int(df.iloc[0]['n']) if not df.empty else 0
            if n < 3:
                ok_all = False
                break
        if ok_all:
            return True, "PASS: multiple model_path rows co-exist for each ts+feature_key in temp DB (added 2 unique mock model_paths)"
        return False, "FAIL: expected at least 3 rows per ts (original + 2 mocks)"


def _test_duplicate_reject(tmp_db: Path, feature_key: str, model_path: str, ts_preds: List[Tuple[pd.Timestamp, float]]) -> Tuple[bool, str]:
    """Attempt to insert exact duplicates (ts, model_path, feature_key) and expect failure.

    Returns (ok, message) where ok=True means duplicates were rejected as expected.
    """
    if not ts_preds:
        return False, "No timestamps provided for duplicate-reject test"
    with duckdb.connect(str(tmp_db)) as con:
        con.execute("SET TimeZone='UTC';")
        ensure_predictions_table(con)
        try:
            rows = [
                PredictionRow.from_payload({
                    'timestamp': ts,
                    'y_pred': y,
                    'model_path': model_path,
                    'feature_key': feature_key,
                })
                for ts, y in ts_preds
            ]
            insert_predictions(con, rows)
        except Exception as e:
            # Any constraint/duplicate error is a PASS for this test
            return True, "PASS: duplicate (ts, model_path, feature_key) insert was rejected"
        # If we get here, duplicates were accepted, which is a failure
        return False, "FAIL: duplicate insert unexpectedly succeeded"


def main() -> int:
    ap = argparse.ArgumentParser(description="E2E comparison for FS inference vs production predictions")
    ap.add_argument("--feat-duckdb", required=True, type=Path)
    ap.add_argument("--prod-pred-duckdb", required=True, type=Path)
    ap.add_argument("--temp-pred-duckdb", required=True, type=Path)
    ap.add_argument("--feature-key", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--sample", choices=["recent", "random"], default="recent", help="Pick most recent N or random N from all production predictions")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for --sample random")
    args = ap.parse_args()

    model_file_str = _resolve_model_path_str(args.model_path)

    # 1) Select timestamps from production for this model + feature_key
    if args.sample == "recent":
        prod_rows = _fetch_prod_preds(args.prod_pred_duckdb, args.feature_key, model_file_str, args.limit)
        if not prod_rows:
            print("No production predictions found for the provided model_path and feature_key.")
            return 2
        ts_list = [ts for ts, _ in prod_rows]
    else:
        all_ts = _fetch_all_prod_ts(args.prod_pred_duckdb, args.feature_key, model_file_str)
        if not all_ts:
            print("No production predictions found for the provided model_path and feature_key.")
            return 2
        k = min(int(args.limit), len(all_ts))
        rnd = random.Random(int(args.seed))
        ts_list = rnd.sample(all_ts, k)
        # fetch production predictions only for sampled timestamps
        prod_rows = _fetch_preds(args.prod_pred_duckdb, args.feature_key, model_file_str, ts_list)
    print(f"Comparing on {len(ts_list)} timestamps; latest={max(ts_list)} earliest={min(ts_list)} (sample={args.sample})")

    # 2) Run new inference into a temporary predictions DB
    tmp_db = args.temp_pred_duckdb
    tmp_db.parent.mkdir(parents=True, exist_ok=True)
    cfg = InferenceConfig(
        feat_db=args.feat_duckdb,
        pred_db=tmp_db,
        feature_key=str(args.feature_key),
        dataset="e2e",
        model_root=None,
        model_path=str(args.model_path),
        mode="ts_list",
        start=None,
        end=None,
        ts=[t.strftime("%Y-%m-%d %H:%M:%S") for t in ts_list],
        ts_file=None,
        at_most=len(ts_list),
        overwrite=True,
        dry_run=False,
    )
    rc = run_inference(cfg)
    if rc != 0:
        print(f"[ERROR] FS inference returned code {rc}")
        return 3

    # 3) Fetch predictions from temp DB and compare
    new_rows = _fetch_preds(tmp_db, args.feature_key, model_file_str, ts_list)
    common, mism_count, mismatches = _compare_preds(prod_rows, new_rows, tol=float(args.tol))
    if mism_count == 0 and common == len(ts_list):
        print("PASS: FS inference matches production predictions within tolerance.")
    else:
        print(f"FAIL: mismatches={mism_count}/{common}")
        for s in mismatches[:10]:
            print("  ", s)

    # 4) Multi-model coexistence test using the same tmp predictions DB
    # Reuse the y_pred we just wrote; add two additional model_paths for the same ts set
    ok_multi, msg_multi = _insert_multi_models_from_results(tmp_db, args.feature_key, new_rows)
    print(msg_multi)

    # 5) Duplicate reject test: try inserting the same 5 rows again for the real model_path
    ok_dup, msg_dup = _test_duplicate_reject(tmp_db, args.feature_key, model_file_str, new_rows)
    print(msg_dup)

    return 0 if (mism_count == 0 and ok_multi and ok_dup) else 2


if __name__ == "__main__":
    raise SystemExit(main())
