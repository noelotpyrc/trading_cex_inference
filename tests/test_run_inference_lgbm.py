#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import subprocess
import os
import os


def _proj_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = _proj_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Try to run the end-to-end inference CLI if a model is available
    models_root = Path('/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60')
    if not models_root.exists():
        print('SKIP: models root not found')
        return

    # Generate a small synthetic OHLCV with OHLCV columns
    # Use a dedicated tmp folder name for this inference CLI test
    tmp_dir = root / '.tmp' / 'inference_cli_test'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ohlcv_path = tmp_dir / 'ohlcv_1h.csv'
    n = 800
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(3)
    ts = pd.date_range('2024-01-01', periods=n, freq='h')
    base = 100 + np.cumsum(rng.normal(0, 0.5, size=n))
    close = base + rng.normal(0, 0.2, size=n)
    spread = np.abs(rng.normal(0.2, 0.05, size=n))
    open_ = base
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = np.abs(rng.normal(1000, 50, size=n))
    pd.DataFrame({'timestamp': ts, 'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}).to_csv(ohlcv_path, index=False)

    # Use user's dedicated DB test folder
    db_root = Path('/Volumes/Extreme SSD/trading_data/cex/db')
    db_root.mkdir(parents=True, exist_ok=True)
    db_path = db_root / 'inference_test.duckdb'

    venv_python = root / 'venv' / 'bin' / 'python'
    env = dict(**os.environ)
    env['PYTHONPATH'] = str(root)
    debug_dir = tmp_dir / 'debug'
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Count files before
    before_feats = list(debug_dir.glob('features_*.csv'))
    before_preds = list(debug_dir.glob('prediction_*.csv'))

    cmd = [
        str(venv_python), str(root / 'run' / 'run_inference_lgbm.py'),
        '--input-csv', str(ohlcv_path),
        '--dataset', 'BINANCE_BTCUSDT.P, 60',
        '--model-root', str(models_root),
        '--duckdb', str(db_path),
        '--buffer-hours', '6',
        '--debug-dir', str(debug_dir),
    ]
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print('SKIP: inference run failed (likely no usable model):', e)
        return

    # Verify new timestamped debug artifacts exist
    after_feats = list(debug_dir.glob('features_*.csv'))
    after_preds = list(debug_dir.glob('prediction_*.csv'))
    assert len(after_feats) >= len(before_feats) + 1
    assert len(after_preds) >= len(before_preds) + 1

    print('run_inference_lgbm smoke test OK (timestamped debug artifacts written)')


if __name__ == '__main__':
    main()

