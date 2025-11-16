# Trading CEX Inference

Lightweight inference engine for cryptocurrency trading models. Loads pre-trained models and pre-computed features to generate predictions.

## Architecture

This repository is part of the separated CEX trading pipeline:
- **trading_cex_data_processing** - OHLCV loading, feature engineering, feature backfilling
- **trading_cex_inference** - Model inference using pre-computed features
- **trading_cex_ml_training** - Model training and MLflow tracking

## Structure

```
trading_cex_inference/
├── inference/         # Main inference scripts
│   ├── backfill_inference.py          # Infer from pre-computed features (production)
│   └── generate_inference_target.py   # Target generation for evaluation
├── model/             # Model I/O
│   ├── model_io_lgbm.py       # Model loading wrapper
│   └── lgbm_inference.py      # LightGBM inference utilities
├── persistence/       # DuckDB persistence
│   └── predictions_table.py   # Predictions storage
├── utils/             # Utilities
│   └── binning.py             # Prediction binning
├── scripts/           # Additional scripts
│   ├── oneoff_run/            # One-off utility scripts (3 files)
│   └── data_check/            # Data validation scripts (8 files)
└── tests/             # Unit tests
```

## Setup

```bash
cd /Users/noel/projects/trading_cex_inference
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Production Workflow

**Step 1: Backfill Features** (from trading_cex_data_processing repo)

```bash
cd /Users/noel/projects/trading_cex_data_processing

python scripts/backfill_features.py \
  --duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb" \
  --feat-duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb" \
  --table ohlcv_btcusdt_1h \
  --feature-key "manual_backfill" \
  --mode last_from_features \
  --base-hours 720 \
  --feature-list "/Users/noel/projects/trading_cex/configs/feature_lists/binance_btcusdt_p60_default.json"
```

**Step 2: Run Inference** (from this repo)

```bash
cd /Users/noel/projects/trading_cex_inference

python inference/backfill_inference.py \
  --duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb" \
  --feat-duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb" \
  --pred-duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_prediction.duckdb" \
  --table ohlcv_btcusdt_1h \
  --model-path "/Volumes/Extreme SSD/trading_data/cex/models/run/binance_btcusdt_perp_1h/y_logret_168h" \
  --mode last_from_predictions \
  --dataset "binance_btcusdt_perp_1h"
```

### Inference Modes

The `backfill_inference.py` script supports three timestamp selection modes:

**1. Window Mode** - Explicit date range:
```bash
python inference/backfill_inference.py \
  --mode window \
  --start "2025-11-13 00:00:00" \
  --end "2025-11-14 00:00:00" \
  [other args...]
```

**2. Last from Predictions** - Continue from last prediction (incremental):
```bash
python inference/backfill_inference.py \
  --mode last_from_predictions \
  [other args...]
```

**3. Timestamp List** - Explicit timestamps:
```bash
python inference/backfill_inference.py \
  --mode ts_list \
  --ts "2025-11-13 00:00:00" "2025-11-13 01:00:00" \
  [other args...]
```

### Key Features

**Data Validation:**
- Validates OHLCV ↔ Features consistency before inference
- Aborts if features are missing - guides user to run backfill_features.py first

**Smart Feature Loading:**
- Default: Uses most recent features (dedup by created_at DESC)
- Optional: Specify `--feature-key` for reproducible feature version

**Performance:**
- 20-50x faster than legacy scripts (no OHLCV loading or feature computation)
- Inference only: ~3-10s per 1000 bars

## Output

All predictions are stored in DuckDB `predictions` table:

```sql
CREATE TABLE predictions (
    ts TIMESTAMP NOT NULL,
    model_path TEXT NOT NULL,
    y_pred DOUBLE NOT NULL,
    feature_key TEXT,
    dataset TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts, model_path)
)
```

## Dependencies

- **lightgbm** - Model inference
- **pandas, numpy** - Data processing
- **duckdb** - Data persistence
- **trading_cex_ml_training** - Pre-trained models (external)
- **trading_cex_data_processing** - Feature engineering (external, for features)

## Testing

E2E test for inference (requires pre-computed features):

```bash
cd /Users/noel/projects/trading_cex_data_processing
python scripts/test_inference_e2e.py
```

## Migration from Legacy

The new separated architecture replaces these legacy scripts:

| Legacy Script | New Workflow |
|--------------|--------------|
| `backfill_inference_missing.py` | `backfill_features.py` + `backfill_inference.py` |
| `backfill_inference_from_feature_store.py` | `backfill_inference.py` |
| `run_inference_lgbm.py` | Use two-step process (not needed) |

**Benefits:**
- 20-50x faster inference (features pre-computed)
- Clear separation of concerns
- Data validation before inference
- Reuse features for multiple models
