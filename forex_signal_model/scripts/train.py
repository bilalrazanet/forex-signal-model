from __future__ import annotations

import argparse
import os
import sys

# Ensure src is importable when running from scripts/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

from src.data.data_loader import create_data_loader
from src.data.features import add_features, make_horizon_labels
from src.data.utils import parse_interval_minutes
from src.models.train import TrainConfig, get_default_feature_columns, train_xgb_classifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", help="Yahoo FX symbol e.g. EURUSD=X")
    p.add_argument("--symbols", help="Comma-separated Yahoo FX symbols for multi-pair training")
    p.add_argument("--data_source", default="yahoo", choices=["yahoo", "alphavantage"], help="Data source for market data")
    p.add_argument("--api_key", help="API key for AlphaVantage (or set ALPHAVANTAGE_API_KEY)")
    p.add_argument("--interval", default="5m", help="Data interval (e.g. 1m,5m,15m)")
    p.add_argument("--period", default="10d", help="Data lookback period")
    p.add_argument("--horizon_mins", type=int, default=10, help="Forecast horizon minutes")
    p.add_argument("--tp_atr_mult", type=float, default=1.0)
    p.add_argument("--sl_atr_mult", type=float, default=1.0)
    p.add_argument("--fee_bps", type=float, default=1.0)
    p.add_argument("--out", default="model.joblib")
    args = p.parse_args()

    if not args.symbol and not args.symbols:
        raise ValueError("Either --symbol or --symbols must be provided.")

    symbols = [s.strip() for s in (args.symbols or "").split(",") if s.strip()]
    if args.symbol:
        symbols = [args.symbol] + [s for s in symbols if s != args.symbol]

    loader = create_data_loader(args.data_source, api_key=args.api_key)
    interval_minutes = parse_interval_minutes(args.interval)
    horizon_bars = max(1, args.horizon_mins // interval_minutes)

    frames = []
    for symbol in symbols:
        df = loader.fetch_ohlcv(symbol, interval=args.interval, period=args.period)
        if df.empty:
            raise RuntimeError(f"No data returned for symbol: {symbol}")

        df_feat = add_features(df)
        df_lab = make_horizon_labels(
            df_feat,
            horizon_bars=horizon_bars,
            tp_atr_mult=args.tp_atr_mult,
            sl_atr_mult=args.sl_atr_mult,
            fee_bps=args.fee_bps,
        )
        df_lab = df_lab.assign(symbol=symbol)
        frames.append(df_lab)

    df_lab = pd.concat(frames, axis=0)
    feature_cols = get_default_feature_columns(df_lab)

    artifacts = train_xgb_classifier(
        df_lab,
        feature_cols=feature_cols,
        model_out_path=args.out,
        config=TrainConfig(),
    )

    print("Training done.")
    print(artifacts)


if __name__ == "__main__":
    # Ensure src is importable when running from scripts/
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    main()

