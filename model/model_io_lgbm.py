#!/usr/bin/env python3
"""
LightGBM model I/O wrappers for inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import lightgbm as lgb
import pandas as pd

from model.lgbm_inference import resolve_model_file, load_booster, align_features_for_booster, predict_dataframe


def load_lgbm_model(model_root: Optional[str] = None, model_path: Optional[str] = None) -> Tuple[lgb.Booster, Path]:
    mf, run_dir = resolve_model_file(model_root=model_root or None, model_path=model_path or None)
    booster = load_booster(mf)
    return booster, run_dir


def predict_latest_row(booster: lgb.Booster, features_row_df: pd.DataFrame) -> float:
    aligned = align_features_for_booster(features_row_df.drop(columns=['timestamp']), booster)
    preds = predict_dataframe(booster, aligned)
    return float(preds[0])













