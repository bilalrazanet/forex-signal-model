from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

# Ensure src is importable when running from scripts/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import joblib

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.data.data_loader import create_data_loader
from src.data.features import add_features, make_horizon_labels
from src.data.utils import parse_interval_minutes
from src.signals.signals import SignalConfig, predict_signal


@dataclass(frozen=True)
class BacktestConfig:
    n_splits: int = 5
    min_train_size: int = 200


def backtest(
    symbol: str,
    interval: str,
    period: str,
    horizon_mins: int,
    model_bundle: dict,
    feature_cols: List[str],
    tp_atr_mult: float = 1.0,
    sl_atr_mult: float = 1.0,
    fee_bps: float = 1.0,
    signal_cfg: SignalConfig = SignalConfig(),
    cfg: BacktestConfig = BacktestConfig(),
    data_source: str = "yahoo",
    api_key: Optional[str] = None,
) -> Dict:
    loader = create_data_loader(data_source, api_key=api_key)
    df = loader.fetch_ohlcv(symbol, interval=interval, period=period)
    df_feat = add_features(df)

    interval_minutes = parse_interval_minutes(interval)
    horizon_bars = max(1, horizon_mins // interval_minutes)

    df_lab = make_horizon_labels(
        df_feat,
        horizon_bars=horizon_bars,
        tp_atr_mult=tp_atr_mult,
        sl_atr_mult=sl_atr_mult,
        fee_bps=fee_bps,
    )

    data = df_lab.dropna(subset=feature_cols + ["y"]).copy()
    if data.empty:
        raise RuntimeError("Backtest dataset is empty after feature/label prep.")

    X = data[feature_cols]
    y = data["y"].astype(int)

    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)

    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), start=1):
        if len(tr_idx) < cfg.min_train_size:
            continue

        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        model = model_bundle["model"]
        proba_buy = model.predict_proba(X_va)[:, 1]

        # Convert probabilities -> BUY/SELL/HOLD via the same logic as live
        # (this is a proxy evaluation; not a full TP/SL PnL simulation)
        pred = np.full(len(y_va), -1)
        for i in range(len(X_va)):
            out_sig = predict_signal(
                model_bundle,
                X_va.iloc[i],
                feature_cols=feature_cols,
                cfg=signal_cfg,
            )
            if out_sig["side"] == "BUY":
                pred[i] = 1
            elif out_sig["side"] == "SELL":
                pred[i] = 0

        mask = pred != -1
        if mask.any():
            acc = accuracy_score(y_va[mask], pred[mask])
            hold_rate = float((~mask).mean())
            try:
                auc = float(roc_auc_score(y_va[mask], proba_buy[mask]))
            except Exception:
                auc = None
        else:
            acc = None
            hold_rate = 1.0
            auc = None

        fold_metrics.append(
            {
                "fold": fold,
                "n_eval": int(mask.sum()),
                "accuracy_buy_sell_only": acc,
                "hold_rate": hold_rate,
                "roc_auc_buy_sell_only": auc,
            }
        )

    # Global report on all data
    proba_buy_all = model_bundle["model"].predict_proba(X)[:, 1]
    try:
        roc_auc_all = float(roc_auc_score(y, proba_buy_all))
    except Exception:
        roc_auc_all = None

    pred_all = np.full(len(y), -1)
    for i in range(len(X)):
        out_sig = predict_signal(model_bundle, X.iloc[i], feature_cols=feature_cols, cfg=signal_cfg)
        if out_sig["side"] == "BUY":
            pred_all[i] = 1
        elif out_sig["side"] == "SELL":
            pred_all[i] = 0

    mask_all = pred_all != -1
    if mask_all.any():
        acc_all = float(accuracy_score(y[mask_all], pred_all[mask_all]))
        cls_report = classification_report(
            y[mask_all],
            pred_all[mask_all],
            output_dict=True,
            zero_division=0,
        )
        hold_rate_all = float((~mask_all).mean())
    else:
        acc_all = None
        cls_report = {}
        hold_rate_all = 1.0

    return {
        "symbol": symbol,
        "interval": interval,
        "period": period,
        "horizon_mins": horizon_mins,
        "horizon_bars": horizon_bars,
        "signal_thresholds": {
            "buy_threshold": signal_cfg.buy_threshold,
            "sell_threshold": signal_cfg.sell_threshold,
            "min_confidence": signal_cfg.min_confidence,
        },
        "global": {
            "roc_auc": roc_auc_all,
            "accuracy_buy_sell_only": acc_all,
            "hold_rate": hold_rate_all,
            "classification_report": cls_report,
        },
        "folds": fold_metrics,
        "n_folds_used": len(fold_metrics),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--data_source", default="yahoo", choices=["yahoo", "alphavantage"], help="Data source for market data")
    p.add_argument("--api_key", help="API key for AlphaVantage (or set ALPHAVANTAGE_API_KEY)")
    p.add_argument("--interval", default="5m")
    p.add_argument("--period", default="20d")
    p.add_argument("--horizon_mins", type=int, default=10)
    p.add_argument("--model", required=True, help="Path to model.joblib")
    p.add_argument("--tp_atr_mult", type=float, default=1.0)
    p.add_argument("--sl_atr_mult", type=float, default=1.0)
    p.add_argument("--fee_bps", type=float, default=1.0)
    p.add_argument("--buy_threshold", type=float, default=0.55)
    p.add_argument("--min_confidence", type=float, default=0.0)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--out", default="backtest_report.json")
    args = p.parse_args()

    model_bundle = joblib.load(args.model)
    feature_cols = model_bundle["artifacts"]["feature_cols"]
    loader = create_data_loader(args.data_source, api_key=args.api_key)

    report = backtest(
        symbol=args.symbol,
        interval=args.interval,
        period=args.period,
        horizon_mins=args.horizon_mins,
        model_bundle=model_bundle,
        feature_cols=feature_cols,
        tp_atr_mult=args.tp_atr_mult,
        sl_atr_mult=args.sl_atr_mult,
        fee_bps=args.fee_bps,
        signal_cfg=SignalConfig(buy_threshold=args.buy_threshold, min_confidence=args.min_confidence),
        cfg=BacktestConfig(n_splits=args.n_splits),
        data_source=args.data_source,
        api_key=args.api_key,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Backtest saved to: {args.out}")


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    main()

