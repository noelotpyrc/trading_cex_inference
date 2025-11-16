#!/usr/bin/env python3
"""
LightGBM inference utilities and simple CLI.

This module provides reusable functions to:
- Locate and load a trained LightGBM model (model.txt) from a models root
- Align incoming feature frames to the model's expected feature schema
- Run predictions for a pandas DataFrame or a CSV file

Reference training pipeline: see `model/run_lgbm_pipeline.py` for how models
are produced and what artifacts exist inside each run directory.

Typical layout expected under the model root (one or more run_* folders):
  <MODEL_ROOT>/run_YYYYMMDD_HHMMSS_lgbm_<target>_<objective>/
    - model.txt
    - metrics.json
    - feature_importance.csv
    - pred_train.csv / pred_val.csv / pred_test.csv
    - run_metadata.json
    - pipeline_config.json

By default, the CLI will pick the most recently modified model.txt under the
given model root.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import lightgbm as lgb


# Default model root provided by the user
DEFAULT_MODEL_ROOT = Path("/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60")


def _ensure_stdout_logging(level: str = "INFO") -> None:
    """Ensure logging outputs to stdout at the given level without duplicating handlers."""
    root = logging.getLogger()
    desired_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(desired_level)
    has_stdout = False
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler):
            has_stdout = True
            # Keep the existing stream; just ensure level is not higher than desired
            if h.level > desired_level:
                h.setLevel(desired_level)
    if not has_stdout:
        sh = logging.StreamHandler()
        sh.setLevel(desired_level)
        sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root.addHandler(sh)


def _all_model_files(model_root: Path) -> List[Path]:
    """Return all `model.txt` files under the given root (non-recursive across files, but searches subfolders).

    We look for `model.txt` one directory deep because training outputs are in run_* subfolders.
    """
    if not model_root.exists():
        return []
    # Search one level down (run_* dirs). If users have deeper nesting, rglob would work too.
    model_files: List[Path] = []
    for child in model_root.iterdir():
        if child.is_dir():
            candidate = child / "model.txt"
            if candidate.exists():
                model_files.append(candidate)
    # Fallback to recursive search if nothing found at first level
    if not model_files:
        model_files = list(model_root.rglob("model.txt"))
    return sorted(model_files)


def resolve_model_file(
    model_root: Union[str, Path] | None = DEFAULT_MODEL_ROOT,
    model_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Resolve the model.txt path and its run directory.

    Args:
        model_root: Base directory containing one or more trained run directories
        model_path: Optional explicit path pointing either directly to a model.txt file
                    or a run directory containing model.txt

    Returns:
        (model_file_path, run_dir_path)

    Raises:
        FileNotFoundError: When no model file can be found
    """
    if model_path is not None:
        mp = Path(model_path)
        if mp.is_dir():
            model_file = mp / "model.txt"
            if not model_file.exists():
                raise FileNotFoundError(f"model.txt not found under provided directory: {mp}")
            return model_file, mp
        if mp.is_file():
            # If they passed metrics.json or something else, require model.txt
            if mp.name != "model.txt":
                raise FileNotFoundError(f"Provided file is not model.txt: {mp}")
            return mp, mp.parent
        raise FileNotFoundError(f"Provided model_path does not exist: {mp}")

    # Only use model_root when model_path is not provided
    root = Path(model_root) if model_root is not None else DEFAULT_MODEL_ROOT
    # Discover all model.txt files and pick the most recently modified one
    candidates = _all_model_files(root)
    if not candidates:
        raise FileNotFoundError(f"No model.txt files found under model_root: {root}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest, latest.parent


def load_booster(model_file: Union[str, Path]) -> lgb.Booster:
    """Load a LightGBM Booster from a model.txt file."""
    model_path = Path(model_file)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    booster = lgb.Booster(model_file=str(model_path))
    return booster


def load_run_metadata(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load helpful metadata from a run directory when available.

    Attempts to read `run_metadata.json`, `pipeline_config.json`, and `metrics.json`.
    Missing files are ignored.
    """
    run = Path(run_dir)
    meta: Dict[str, Any] = {}
    for name in ("run_metadata.json", "pipeline_config.json", "metrics.json"):
        fp = run / name
        if fp.exists():
            try:
                meta[name] = json.loads(Path(fp).read_text())
            except Exception:
                # Keep best-effort behavior
                pass
    return meta


def align_features_for_booster(
    input_frame: pd.DataFrame,
    booster: lgb.Booster,
    *,
    fill_value: float | None = np.nan,
    drop_extra: bool = True,
) -> pd.DataFrame:
    """Align a DataFrame to the Booster's expected feature schema.

    - Ensures all expected features exist (adds missing cols with `fill_value`)
    - Orders columns exactly as the model expects
    - Drops extra columns by default
    - Coerces dtypes to float where possible
    """
    expected: List[str] = list(booster.feature_name())
    frame = input_frame.copy()

    missing = [c for c in expected if c not in frame.columns]
    extra = [c for c in frame.columns if c not in expected]

    if missing:
        logging.warning("%d missing feature(s) not in input: %s", len(missing), ", ".join(missing[:20]))
        for col in missing:
            frame[col] = fill_value

    if drop_extra and extra:
        logging.info("Dropping %d extra column(s) not used by model", len(extra))
        frame = frame.drop(columns=extra)

    # Reorder to match training schema
    frame = frame[expected]

    # Coerce to numeric dtype (LightGBM expects floats), preserving NaNs
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')

    return frame


def predict_dataframe(
    booster: lgb.Booster,
    features: pd.DataFrame,
    *,
    num_iteration: Optional[int] = None,
) -> np.ndarray:
    """Run predictions on an already-aligned features DataFrame."""
    if num_iteration is None:
        try:
            num_iteration = int(getattr(booster, "best_iteration", 0) or 0) or None
        except Exception:
            num_iteration = None
    preds = booster.predict(features, num_iteration=num_iteration)
    return np.asarray(preds)


def predict_from_csv(
    *,
    input_csv: Union[str, Path],
    model_root: Union[str, Path] = DEFAULT_MODEL_ROOT,
    model_path: Optional[Union[str, Path]] = None,
    output_csv: Optional[Union[str, Path]] = None,
    timestamp_column: Optional[str] = None,
    merge_input: bool = False,
    num_iteration: Optional[int] = None,
) -> pd.DataFrame:
    """Predict from a CSV file using a model chosen from a root or explicit path.

    Returns the resulting DataFrame (predictions and optionally merged inputs). When
    `output_csv` is provided, writes the result to disk as well.
    """
    model_file, run_dir = resolve_model_file(model_root=model_root, model_path=model_path)
    logging.info("Using model: %s", model_file)

    booster = load_booster(model_file)
    df = pd.read_csv(input_csv)

    # Preserve an identifier/timestamp column if requested
    id_series = None
    if timestamp_column and timestamp_column in df.columns:
        id_series = df[timestamp_column].copy()

    features = align_features_for_booster(df, booster)
    preds = predict_dataframe(booster, features, num_iteration=num_iteration)

    result = pd.DataFrame({"y_pred": preds})
    if id_series is not None:
        result.insert(0, timestamp_column, id_series.values)
    if merge_input:
        # Reattach full input after predictions for debugging/analysis convenience
        result = pd.concat([result, df.reset_index(drop=True)], axis=1)

    if output_csv is not None:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        logging.info("Predictions saved to: %s", out_path)

    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run LightGBM inference using a trained model.")
    p.add_argument("--input-csv", required=True, help="Path to CSV containing features (may contain extra columns)")
    p.add_argument("--output-csv", default=None, help="Optional path to save predictions CSV")
    p.add_argument(
        "--model-root",
        default=str(DEFAULT_MODEL_ROOT),
        help="Directory containing trained run_* folders with model.txt (default: user's models root)",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Explicit path to model.txt or a run directory containing model.txt. Overrides --model-root.",
    )
    p.add_argument(
        "--timestamp-column",
        default=None,
        help="Optional column name in input CSV to carry through to outputs.",
    )
    p.add_argument(
        "--merge-input",
        action="store_true",
        help="Include original input columns alongside predictions in the output.",
    )
    p.add_argument(
        "--num-iteration",
        type=int,
        default=None,
        help="Override number of boosting iterations to use for prediction (default uses model.best_iteration).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    _ensure_stdout_logging(args.log_level)

    predict_from_csv(
        input_csv=Path(args.input_csv),
        model_root=Path(args.model_root) if args.model_root else DEFAULT_MODEL_ROOT,
        model_path=Path(args.model_path) if args.model_path else None,
        output_csv=Path(args.output_csv) if args.output_csv else None,
        timestamp_column=args.timestamp_column,
        merge_input=bool(args.merge_input),
        num_iteration=args.num_iteration,
    )


if __name__ == "__main__":
    main()


