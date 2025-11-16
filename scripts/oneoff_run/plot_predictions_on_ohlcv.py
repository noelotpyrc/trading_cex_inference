"""
Plot predictions as signals (y_pred > 0) over OHLCV candlesticks and color
them by realized y_true sign based on open_{t+1} vs close_{t+24}.

Inputs:
- Predictions CSV from run/merge_and_score_current_norm.py (must have 'timestamp' and 'y_pred')
- OHLCV file (csv/parquet/pkl) with columns: open, high, low, close, volume (case-insensitive)

Output:
- An interactive Plotly HTML saved next to the predictions file unless --output is provided.

Example:
  python run/plot_predictions_on_ohlcv.py \
    --pred-csv "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60/diagnosis/y_logret_24h_huber_all/predictions_from_merged_current_norm.csv" \
    --ohlcv "/Volumes/Extreme SSD/trading_data/cex/lookbacks/binance_btcusdt_perp_1h/ohlcv_1h.csv" \
    --output "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60/diagnosis/y_logret_24h_huber_all/predictions_overlay.html"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _read_any(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    # Fallback to CSV
    return pd.read_csv(path)


def _ensure_dt_index(df: pd.DataFrame, candidates=("timestamp", "time", "datetime", "date")) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    ts_col: Optional[str] = None
    for c in candidates:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        ts_col = df.columns[0]
    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    if hasattr(ts, "dt"):
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        ts = ts.tz_convert("UTC").tz_localize(None)
    df = df.drop(columns=[ts_col])
    df.index = pd.DatetimeIndex(ts, name="timestamp")
    return df.sort_index()


def _load_predictions(path: Path) -> pd.DataFrame:
    df = _read_any(path)
    df = _ensure_dt_index(df)
    # Ensure y_pred exists
    if "y_pred" not in df.columns:
        # If merged outputs contain a different name, try to find it
        raise ValueError("Predictions CSV must contain 'y_pred' column")
    return df[["y_pred"]]


def _load_ohlcv(path: Path) -> pd.DataFrame:
    df = _read_any(path)
    df = _ensure_dt_index(df)
    # Standardize column names
    rename_map = {c: str(c).lower() for c in df.columns}
    df = df.rename(columns=rename_map)
    needed = ["open", "high", "low", "close"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV is missing required columns: {missing}")
    return df


def plot_predictions_on_ohlcv(pred_csv: Path, ohlcv_path: Path, output: Optional[Path] = None) -> Path:
    preds = _load_predictions(pred_csv)
    ohlcv = _load_ohlcv(ohlcv_path)

    # Restrict OHLCV to predictions time range
    start, end = preds.index.min(), preds.index.max()
    ohlcv_win = ohlcv[(ohlcv.index >= start) & (ohlcv.index <= end)]

    # Align signal timestamps to OHLCV index (only y_pred > 0)
    signals = preds[preds["y_pred"] > 0.0]
    ts = signals.index
    # Require presence of t, t+1h (open), and t+24h (close) in OHLCV
    mask_t = ts.isin(ohlcv.index)
    mask_t1 = (ts + pd.Timedelta(hours=1)).isin(ohlcv.index)
    mask_t24 = (ts + pd.Timedelta(hours=24)).isin(ohlcv.index)
    valid_ts = ts[mask_t & mask_t1 & mask_t24 & (ts >= start) & (ts <= end)]

    # Compute realized y_true sign at these valid timestamps
    # Use reindex to preserve 1:1 positional alignment with base timestamps t
    open_tp1_vals = ohlcv["open"].reindex(valid_ts + pd.Timedelta(hours=1)).astype(float).to_numpy()
    close_tp24_vals = ohlcv["close"].reindex(valid_ts + pd.Timedelta(hours=24)).astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ytrue_logret = np.log(close_tp24_vals / open_tp1_vals)
    ytrue_sign = pd.Series(np.sign(ytrue_logret), index=valid_ts)

    ts_up = ytrue_sign[ytrue_sign > 0].index
    ts_dn = ytrue_sign[ytrue_sign <= 0].index

    # Print overall win rate over evaluated signals (y_pred>0 with valid outcomes)
    total_signals = int(((signals.index >= start) & (signals.index <= end)).sum())
    evaluated = int(len(ytrue_sign))
    wins = int((ytrue_sign > 0).sum())
    losses = evaluated - wins
    win_rate = (wins / evaluated * 100.0) if evaluated > 0 else float('nan')
    print(f"Signals (y_pred>0) in window: {total_signals}")
    print(f"Evaluated (with open(t+1) & close(t+24)): {evaluated}")
    print(f"Wins (y_true>0): {wins} | Losses: {losses} | Win rate: {win_rate:.2f}%")

    y_up = ohlcv_win.loc[ts_up, "high"] * 1.002 if len(ts_up) else pd.Series(dtype=float)
    y_dn = ohlcv_win.loc[ts_dn, "low"] * 0.998 if len(ts_dn) else pd.Series(dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=ohlcv_win.index,
            open=ohlcv_win["open"], high=ohlcv_win["high"], low=ohlcv_win["low"], close=ohlcv_win["close"],
            name="OHLCV",
        )
    )
    if len(ts_up):
        fig.add_trace(
            go.Scatter(
                x=ts_up, y=y_up,
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="limegreen", line=dict(width=1, color="#004d00")),
                name="y_pred>0 & y_true>0",
            )
        )
    if len(ts_dn):
        fig.add_trace(
            go.Scatter(
                x=ts_dn, y=y_dn,
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#cc0000", line=dict(width=1, color="#660000")),
                name="y_pred>0 & y_true<=0",
            )
        )
    fig.update_layout(
        title="Predictions overlay (y_pred>0; colored by y_true sign from open_{t+1} vs close_{t+24})",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    out_path = output or pred_csv.with_name("predictions_overlay.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Mark y_pred>0 signals on OHLCV candlesticks and save as HTML")
    ap.add_argument("--pred-csv", required=True, help="Predictions CSV with columns: timestamp,y_pred")
    ap.add_argument("--ohlcv", required=True, help="OHLCV file (csv/parquet/pkl) with open,high,low,close,volume")
    ap.add_argument("--output", default=None, help="Output HTML path (default: predictions_overlay.html next to pred-csv)")
    args = ap.parse_args()

    pred_csv = Path(args.pred_csv)
    ohlcv_path = Path(args.ohlcv)
    out = Path(args.output) if args.output else None
    out_path = plot_predictions_on_ohlcv(pred_csv, ohlcv_path, out)
    print(f"Saved overlay: {out_path}")


if __name__ == "__main__":
    main()
