import pandas as pd
from pathlib import Path

shap_path = '/Volumes/Extreme SSD/trading_data/cex/shap/shap_values_test.parquet'
pred_path = '/Volumes/Extreme SSD/trading_data/cex/models/run/binance_btcusdt_perp_1h/y_logret_168h/pred_test.csv'

shap_df = pd.read_parquet(shap_path)
pred_df = pd.read_csv(pred_path)

merged = shap_df[['timestamp','y_pred']].merge(pred_df[['timestamp','y_pred']], on='timestamp', suffixes=('_shap','_orig'))
print(merged.head(10).to_string(index=False))
print('\nTail:')
print(merged.tail(10).to_string(index=False))
print('\nMax abs diff:', (merged['y_pred_shap']-merged['y_pred_orig']).abs().max())

print('\nShap:')
print(shap_df[['timestamp','y_pred']].head(3).to_string(index=False))
print('\nPred:')
print(pred_df[['timestamp','y_pred']].head(3).to_string(index=False))