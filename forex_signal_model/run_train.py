"""
run_train.py — Wrapper that:
  1. Patches YahooFinanceLoader.fetch_ohlcv with a direct requests call that works
     (yfinance's internal session gets rate-blocked for Forex pairs on Yahoo Finance).
  2. Drops always-NaN volume features (vol_z, vol_chg) from the feature list
     before training, because Yahoo Finance returns Volume=0 for all FX bars.

Your model source files (src/, scripts/) are NOT modified.

Usage (same flags as scripts/train.py):
    python run_train.py --symbol EURUSD=X --interval 5m --period 10d --out model.joblib
    python run_train.py --symbols EURUSD=X,GBPUSD=X --interval 5m --period 10d
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import requests
import numpy as np

# ── Direct Yahoo Finance fetcher (bypasses broken yfinance session) ────────
YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com/",
}

# Always fetch the maximum range and let make_horizon_labels use what it needs.
# 60d covers the most common training periods for 5m/15m intervals.
FETCH_RANGE = "60d"

# Volume-based columns that are always NaN for FX because Yahoo returns Vol=0
ALWAYS_NAN_FX_COLS = {"vol_z", "vol_chg"}


def _direct_fetch_ohlcv(symbol: str, interval: str = "5m", period: str = "10d") -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance v8 chart API using a browser User-Agent."""
    from datetime import timedelta

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"interval": interval, "range": FETCH_RANGE}

    resp = requests.get(url, params=params, headers=YAHOO_HEADERS, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame(quote, index=pd.to_datetime(timestamps, unit="s", utc=True))
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
    df.index.name = "Datetime"

    # Filter to requested period (e.g. "10d", "5d")
    if period.endswith("d"):
        cutoff = df.index.max() - timedelta(days=int(period[:-1]))
        df = df[df.index >= cutoff]
    elif period.endswith("h"):
        cutoff = df.index.max() - timedelta(hours=int(period[:-1]))
        df = df[df.index >= cutoff]

    return df


# ── Patch YahooFinanceLoader before importing training code ────────────────
from src.data import data_loader as _dl  # noqa: E402


def _patched_fetch(self, symbol: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    print(f"[run_train] Fetching {symbol} ({interval}, {period}) via direct Yahoo API...")
    df = _direct_fetch_ohlcv(symbol, interval=interval, period=period)
    print(f"[run_train] Got {len(df)} rows for {symbol}")
    return df


_dl.YahooFinanceLoader.fetch_ohlcv = _patched_fetch


# ── Patch get_default_feature_columns to drop always-NaN FX volume cols ───
from src.models import train as _model_train  # noqa: E402
_orig_get_cols = _model_train.get_default_feature_columns


def _patched_get_cols(df: pd.DataFrame):
    cols = _orig_get_cols(df)
    filtered = [c for c in cols if c not in ALWAYS_NAN_FX_COLS]
    dropped = [c for c in cols if c in ALWAYS_NAN_FX_COLS]
    if dropped:
        print(f"[run_train] Dropping always-NaN FX volume features: {dropped}")
    return filtered


_model_train.get_default_feature_columns = _patched_get_cols


# ── Run the original training main() ──────────────────────────────────────
from scripts.train import main  # noqa: E402

if __name__ == "__main__":
    main()
