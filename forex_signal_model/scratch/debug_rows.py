import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.binance_loader import BinanceLoader
from src.data.features import add_features, make_horizon_labels
from app import _resolve_binance_sym, _patched_get_cols

loader = BinanceLoader()
df = loader.fetch_ohlcv("PAXGUSDT", interval="1h", period="30d")
print("Fetched rows:", len(df))

df_feat = add_features(df)
print("After add_features rows:", len(df_feat))
print("NaN counts per column:")
for col in df_feat.columns:
    n = df_feat[col].isna().sum()
    if n > 0:
        print(f"  {col}: {n} NaNs")

cols = _patched_get_cols(df_feat)
df_clean = df_feat.dropna(subset=cols)
print("After dropna(subset=valid_cols) rows:", len(df_clean))
