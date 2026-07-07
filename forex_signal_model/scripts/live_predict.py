from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure src is importable when running from scripts/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import joblib
import pandas as pd

from src.broker.metatrader_bridge import BrokerConfig, FileSignalBroker
from src.data.data_loader import create_data_loader
from src.data.features import add_features
from src.signals.signals import SignalConfig, predict_signal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--data_source", default="yahoo", choices=["yahoo", "alphavantage"], help="Data source for market data")
    p.add_argument("--api_key", help="API key for AlphaVantage (or set ALPHAVANTAGE_API_KEY)")
    p.add_argument("--interval", default="5m")
    p.add_argument("--period", default="5d")
    p.add_argument("--model", default="model.joblib")
    p.add_argument("--buy_threshold", type=float, default=0.55)
    p.add_argument("--min_confidence", type=float, default=0.0)
    p.add_argument("--signals_file", default="signals_out.json")
    p.add_argument("--sleep_s", type=int, default=20)
    args = p.parse_args()

    model_bundle = joblib.load(args.model)
    feature_cols = model_bundle["artifacts"]["feature_cols"]

    loader = create_data_loader(args.data_source, api_key=args.api_key)
    broker = FileSignalBroker(BrokerConfig(signals_file=args.signals_file))
    sig_cfg = SignalConfig(buy_threshold=args.buy_threshold, min_confidence=args.min_confidence)

    print("Starting live prediction loop...")
    while True:
        df = loader.fetch_ohlcv(args.symbol, interval=args.interval, period=args.period)
        df_feat = add_features(df)
        df_feat = df_feat.dropna(subset=feature_cols)
        if df_feat.empty:
            time.sleep(args.sleep_s)
            continue

        latest_row = df_feat.iloc[-1]
        out = predict_signal(model_bundle, latest_row, feature_cols=feature_cols, cfg=sig_cfg)
        payload = {
            "symbol": args.symbol,
            "interval": args.interval,
            "prediction": out,
            "latest_time": str(df_feat.index[-1]),
        }
        broker.publish(payload)
        print(payload)
        time.sleep(args.sleep_s)


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    main()

