"""
app.py — ForexAI Signal Dashboard  (Binance-ONLY edition)

Data source  : 100% Binance REST API (no Yahoo Finance)
Gold (XAUUSD): PAX Gold token on Binance (PAXGUSDT) — tracks spot XAU/USD

APIs:
  POST /api/train          — train XGBoost model
  GET  /api/predict        — full trade entry recommendation
  GET  /api/chartdata      — OHLCV candles
  GET  /api/ticker         — live price + 24h stats
  GET  /api/status         — server health

Run:  python app.py  →  http://localhost:5000
"""
from __future__ import annotations

import os, sys, threading, traceback
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import numpy as np
import pandas as pd
import joblib

# ── ALL data comes from Binance ────────────────────────────────────────────
from src.data.binance_loader import BinanceLoader

# Map user-facing names → Binance trading symbols
BINANCE_SYMBOL_MAP: dict[str, str] = {
    "BTCUSD":   "BTCUSDT",
    "BTCUSDT":  "BTCUSDT",
    "ETHUSD":   "ETHUSDT",
    "ETHUSDT":  "ETHUSDT",
    "BNBUSD":   "BNBUSDT",
    "BNBUSDT":  "BNBUSDT",
    "SOLUSD":   "SOLUSDT",
    "SOLUSDT":  "SOLUSDT",
    "XRPUSD":   "XRPUSDT",
    "XRPUSDT":  "XRPUSDT",
    "ADAUSD":   "ADAUSDT",
    "DOTUSD":   "DOTUSDT",
    "AVAXUSD":  "AVAXUSDT",
    "AVAXUSDT": "AVAXUSDT",
    "LINKUSD":  "LINKUSDT",
    "LTCUSD":   "LTCUSDT",
    # Gold: PAX Gold token on Binance — 1 PAXG = 1 troy ounce of gold
    "XAUUSD":   "PAXGUSDT",
    "GOLD":     "PAXGUSDT",
    "PAXG":     "PAXGUSDT",
    "PAXGUSDT": "PAXGUSDT",
}

# New map for Yahoo Finance
YFINANCE_SYMBOL_MAP: dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "XAGUSD": "SI=F",  # Silver Futures
    "SILVER": "SI=F",
    "OIL": "CL=F",     # Crude Oil Futures
    "WTI": "CL=F",
    "SPX": "^GSPC",    # S&P 500
    "NDX": "^IXIC",    # Nasdaq
}

# Human-readable labels for display
SYMBOL_LABELS: dict[str, str] = {
    "BTCUSDT":  "BTC/USD",
    "ETHUSDT":  "ETH/USD",
    "BNBUSDT":  "BNB/USD",
    "SOLUSDT":  "SOL/USD",
    "XRPUSDT":  "XRP/USD",
    "ADAUSDT":  "ADA/USD",
    "DOTUSDT":  "DOT/USD",
    "AVAXUSDT": "AVAX/USD",
    "LINKUSDT": "LINK/USD",
    "LTCUSDT":  "LTC/USD",
    "PAXGUSDT": "XAU/USD (Gold)",
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "USDCHF=X": "USD/CHF",
    "AUDUSD=X": "AUD/USD",
    "USDCAD=X": "USD/CAD",
    "NZDUSD=X": "NZD/USD",
    "SI=F":     "XAG/USD (Silver)",
    "CL=F":     "WTI (Crude Oil)",
    "^GSPC":    "S&P 500",
    "^IXIC":    "Nasdaq",
}

# Volume-based features that break on some pairs
ALWAYS_DROP_COLS = {"vol_z", "vol_chg"}

# ── Remove Patch and Init Data Loaders ─────────────────────────────────────
from src.data import data_loader as _dl

_binance_global = BinanceLoader()
_yahoo_global   = _dl.YahooFinanceLoader()

def _resolve_symbol(raw: str) -> tuple[str, str]:
    raw = raw.upper().strip()
    if raw in BINANCE_SYMBOL_MAP or raw in BINANCE_SYMBOL_MAP.values():
        return "binance", BINANCE_SYMBOL_MAP.get(raw, raw)
    elif raw in YFINANCE_SYMBOL_MAP or raw in YFINANCE_SYMBOL_MAP.values():
        return "yahoo", YFINANCE_SYMBOL_MAP.get(raw, raw)
    else:
        # Fallback to yahoo
        return "yahoo", raw

# Patch feature columns to remove NaN volume cols
from src.models import train as _model_train
_orig_get_cols = _model_train.get_default_feature_columns


def _patched_get_cols(df: pd.DataFrame):
    cols = _orig_get_cols(df)
    return [c for c in cols if c not in ALWAYS_DROP_COLS]


_model_train.get_default_feature_columns = _patched_get_cols

# ── Import model pipeline ──────────────────────────────────────────────────
from src.data.features import add_features, make_horizon_labels
from src.data.utils import parse_interval_minutes
from src.models.train import TrainConfig, get_default_feature_columns, train_xgb_classifier
from src.signals.signals import SignalConfig, predict_signal

# ── Flask App ──────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="dashboard", static_url_path="")
CORS(app)

MODEL_PATH  = "model.joblib"
_model_bundle = None
_train_lock   = threading.Lock()
_binance      = BinanceLoader()

INTERVAL_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15,
    "30m": 30, "1h": 60, "4h": 240, "1d": 1440,
}


def _fetch(symbol: str, interval: str, period: str) -> tuple[str, str, pd.DataFrame]:
    source, resolved_sym = _resolve_symbol(symbol)
    if source == "binance":
        df = _binance.fetch_ohlcv(resolved_sym, interval=interval, period=period)
    else:
        # yfinance doesn't natively do 4h, it maps best to 1h or 1d depending on the library logic
        # we'll just pass it through
        df = _yahoo_global.fetch_ohlcv(resolved_sym, interval=interval, period=period)
    return source, resolved_sym, df


def _compute_trade_levels(row: pd.Series, signal: str, price: float) -> dict:
    """Compute SL/TP/RR from ATR for a complete trade entry recommendation."""
    atr = float(row.get("atr_14", price * 0.002))  # fallback 0.2% ATR

    # Risk:Reward = 1:2 for BUY, 1:2 for SELL
    sl_mult = 1.5
    tp_mult = 2.5

    if signal == "BUY":
        sl    = price - sl_mult * atr
        tp    = price + tp_mult * atr
        sl_dist = price - sl
        tp_dist = tp - price
    elif signal == "SELL":
        sl    = price + sl_mult * atr
        tp    = price - tp_mult * atr
        sl_dist = sl - price
        tp_dist = price - tp
    else:  # HOLD
        sl = tp = price
        sl_dist = tp_dist = 0.0

    rr = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0.0

    # Format helpers
    def fmt(v):
        return round(float(v), 6) if price < 10 else round(float(v), 2)

    return {
        "entry":    fmt(price),
        "sl":       fmt(sl),
        "tp":       fmt(tp),
        "sl_dist":  fmt(sl_dist),
        "tp_dist":  fmt(tp_dist),
        "sl_pct":   round(sl_dist / price * 100, 3),
        "tp_pct":   round(tp_dist / price * 100, 3),
        "rr_ratio": rr,
    }


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("dashboard", "index.html")


# ── GET /api/status ────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    binance_ok = False
    btc_price  = 0.0
    try:
        btc_price  = _binance.get_ticker_price("BTCUSDT")
        binance_ok = btc_price > 0
    except Exception:
        pass
    return jsonify({
        "server":            "online",
        "model_loaded":      _model_bundle is not None or os.path.exists(MODEL_PATH),
        "binance_connected": binance_ok,
        "btc_price":         btc_price,
        "data_source":       "Binance API only",
    })


# ── GET /api/ticker ────────────────────────────────────────────────────────
@app.route("/api/ticker")
def api_ticker():
    symbol = request.args.get("symbol", "BTCUSD")
    try:
        source, resolved_sym = _resolve_symbol(symbol)
        if source == "binance":
            stats = _binance.get_24hr_stats(resolved_sym)
            label = SYMBOL_LABELS.get(resolved_sym, resolved_sym)
            return jsonify({
                "symbol":       symbol,
                "resolved_sym": resolved_sym,
                "label":        label,
                "price":        float(stats["lastPrice"]),
                "change_pct":   float(stats["priceChangePercent"]),
                "change_abs":   float(stats["priceChange"]),
                "high_24h":     float(stats["highPrice"]),
                "low_24h":      float(stats["lowPrice"]),
                "volume_24h":   float(stats["volume"]),
                "source":       source,
            })
        else:
            # For Yahoo, fetch recent candle to approximate ticker
            df = _yahoo_global.fetch_ohlcv(resolved_sym, interval="1d", period="2d")
            if df.empty:
                raise ValueError("No data from Yahoo Finance")
            last_close = df["Close"].iloc[-1]
            prev_close = df["Close"].iloc[0] if len(df) > 1 else last_close
            change_abs = last_close - prev_close
            change_pct = (change_abs / prev_close) * 100 if prev_close else 0.0
            label = SYMBOL_LABELS.get(resolved_sym, resolved_sym)
            return jsonify({
                "symbol":       symbol,
                "resolved_sym": resolved_sym,
                "label":        label,
                "price":        float(last_close),
                "change_pct":   float(change_pct),
                "change_abs":   float(change_abs),
                "high_24h":     float(df["High"].iloc[-1]),
                "low_24h":      float(df["Low"].iloc[-1]),
                "volume_24h":   float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0.0,
                "source":       source,
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── POST /api/train ────────────────────────────────────────────────────────
@app.route("/api/train", methods=["POST"])
def api_train():
    global _model_bundle
    if _train_lock.locked():
        return jsonify({"error": "Training already in progress"}), 409

    data     = request.get_json(force=True)
    symbol   = data.get("symbol",   "BTCUSD")
    interval = data.get("interval", "5m")
    period   = data.get("period",   "30d")
    horizon_minutes = int(data.get("horizon_minutes", 15))

    with _train_lock:
        try:
            interval_mins    = INTERVAL_MINUTES.get(interval, 5)
            horizon_bars     = max(1, round(horizon_minutes / interval_mins))

            source, resolved_sym, df = _fetch(symbol, interval, period)
            if df.empty:
                return jsonify({"error": f"No data from {source} for {resolved_sym}"}), 400

            df_feat = add_features(df)
            df_lab  = make_horizon_labels(
                df_feat, horizon_bars=horizon_bars,
                tp_atr_mult=1.5, sl_atr_mult=1.0, fee_bps=1.0,
            )
            df_lab = df_lab.assign(symbol=resolved_sym)

            feature_cols = get_default_feature_columns(df_lab)
            artifacts    = train_xgb_classifier(
                df_lab,
                feature_cols=feature_cols,
                model_out_path=MODEL_PATH,
                config=TrainConfig(),
            )
            _model_bundle = joblib.load(MODEL_PATH)

            return jsonify({
                "ok":              True,
                "symbol":          symbol,
                "binance_sym":     resolved_sym,
                "interval":        interval,
                "period":          period,
                "horizon_minutes": horizon_minutes,
                "horizon_bars":    horizon_bars,
                "rows_fetched":    len(df),
                "rows_trained":    len(df_lab),
                "cv_auc":          artifacts.get("cv_auc_mean"),
                "feature_count":   len(feature_cols),
                "source":          source,
            })
        except Exception as e:
            return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ── GET /api/predict ───────────────────────────────────────────────────────
@app.route("/api/predict")
def api_predict():
    global _model_bundle
    if _model_bundle is None:
        if os.path.exists(MODEL_PATH):
            _model_bundle = joblib.load(MODEL_PATH)
        else:
            return jsonify({"error": "Model not trained yet. Please train first."}), 400

    symbol          = request.args.get("symbol",    "BTCUSD")
    interval        = request.args.get("interval",  "5m")
    threshold       = float(request.args.get("threshold", 0.52))
    horizon_minutes = int(request.args.get("horizon_minutes", 15))

    try:
        source, resolved_sym, df = _fetch(symbol, interval, "3d")
        if df.empty:
            return jsonify({"error": f"No data from {source}"}), 400

        df_feat      = add_features(df)
        feature_cols = _model_bundle["artifacts"]["feature_cols"]
        valid_cols   = [c for c in feature_cols if c in df_feat.columns]
        df_clean     = df_feat.dropna(subset=valid_cols)

        if df_clean.empty:
            return jsonify({"error": "No valid rows after feature engineering"}), 400

        latest_row = df_clean.iloc[-1]
        sig_cfg    = SignalConfig(buy_threshold=threshold, min_confidence=0.0)
        out        = predict_signal(_model_bundle, latest_row,
                                    feature_cols=feature_cols, cfg=sig_cfg)

        signal     = str(out.get("side", "HOLD"))
        confidence = float(out.get("confidence", 0.5))
        proba_buy  = float(out.get("proba_buy",  0.5))
        proba_sell = float(out.get("proba_sell", 0.5))
        price      = float(df.iloc[-1]["Close"])

        trade = _compute_trade_levels(latest_row, signal, price)

        # Valid until
        now        = datetime.now(timezone.utc)
        valid_till = (now + timedelta(minutes=horizon_minutes)).strftime("%H:%M UTC")

        # Human-readable message
        label = SYMBOL_LABELS.get(resolved_sym, resolved_sym)
        if signal == "BUY":
            msg = (f"LONG {label} @ {trade['entry']} | "
                   f"SL: {trade['sl']} (-{trade['sl_pct']}%) | "
                   f"TP: {trade['tp']} (+{trade['tp_pct']}%) | "
                   f"R:R 1:{trade['rr_ratio']}")
        elif signal == "SELL":
            msg = (f"SHORT {label} @ {trade['entry']} | "
                   f"SL: {trade['sl']} (+{trade['sl_pct']}%) | "
                   f"TP: {trade['tp']} (-{trade['tp_pct']}%) | "
                   f"R:R 1:{trade['rr_ratio']}")
        else:
            msg = f"WAIT — No clear setup for {label}. Stay flat."

        # Latest candle for chart
        latest_candle = {
            "time":  int(df_clean.index[-1].timestamp()),
            "open":  float(df.iloc[-1]["Open"]),
            "high":  float(df.iloc[-1]["High"]),
            "low":   float(df.iloc[-1]["Low"]),
            "close": price,
        }

        return jsonify({
            # Core signal
            "signal":           signal,
            "confidence":       round(confidence, 4),
            "proba_buy":        round(proba_buy, 4),
            "proba_sell":       round(proba_sell, 4),
            # Trade entry
            "entry":            trade["entry"],
            "stop_loss":        trade["sl"],
            "take_profit":      trade["tp"],
            "sl_distance":      trade["sl_dist"],
            "tp_distance":      trade["tp_dist"],
            "sl_pct":           trade["sl_pct"],
            "tp_pct":           trade["tp_pct"],
            "rr_ratio":         trade["rr_ratio"],
            # Meta
            "symbol":           symbol,
            "binance_sym":      resolved_sym,
            "label":            SYMBOL_LABELS.get(resolved_sym, resolved_sym),
            "price":            price,
            "horizon_minutes":  horizon_minutes,
            "valid_until":      valid_till,
            "trade_message":    msg,
            "trade_reason":     out.get("reason", "No reason generated."),
            "timestamp":        str(df_clean.index[-1]),
            "latest_candle":    latest_candle,
            "source":           source,
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ── GET /api/chartdata ─────────────────────────────────────────────────────
@app.route("/api/chartdata")
def api_chartdata():
    symbol   = request.args.get("symbol",   "BTCUSD")
    interval = request.args.get("interval", "5m")
    try:
        source, resolved_sym, df = _fetch(symbol, interval, "5d")
        if df.empty:
            return jsonify({"candles": []})
        candles = [
            {
                "time":  int(ts.timestamp()),
                "open":  round(float(r["Open"]),  6),
                "high":  round(float(r["High"]),  6),
                "low":   round(float(r["Low"]),   6),
                "close": round(float(r["Close"]), 6),
            }
            for ts, r in df.iterrows()
        ]
        return jsonify({
            "candles":   candles,
            "symbol":    symbol,
            "binance_sym": resolved_sym,
            "label":     SYMBOL_LABELS.get(resolved_sym, resolved_sym),
            "interval":  interval,
            "source":    source,
            "count":     len(candles),
        })
    except Exception as e:
        return jsonify({"error": str(e), "candles": []}), 500


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ForexAI Signal Dashboard  [Binance-Only Edition]")
    print("  http://localhost:5000")
    print("  Binance key:", "LOADED" if os.getenv("BINANCE_API_KEY") else "NOT SET")
    print("  Gold (XAU/USD) via PAXG token on Binance")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
