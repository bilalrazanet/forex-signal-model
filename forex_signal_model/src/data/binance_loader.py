"""
src/data/binance_loader.py — Binance market data loader.

Fetches OHLCV candlestick data from Binance REST API.
Supports all Binance spot symbols: BTCUSDT, ETHUSDT, XAUUSDT, etc.

Note on XAUUSD (Gold):
    Binance does not have a XAUUSDT spot pair on standard exchange.
    Gold data is fetched from Yahoo Finance (GC=F) automatically when
    'XAUUSD' is requested via the smart router in data_loader.py.

Binance interval mapping:
    1m  = 1 minute
    5m  = 5 minutes
    15m = 15 minutes
    1h  = 1 hour
    4h  = 4 hours
    1d  = 1 day
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlencode

import pandas as pd
import requests


# ── Symbol normalisation ───────────────────────────────────────────────────
# Maps user-facing names → Binance ticker symbols
SYMBOL_MAP: dict[str, str] = {
    "BTCUSD":   "BTCUSDT",
    "BTC/USD":  "BTCUSDT",
    "BTCUSDT":  "BTCUSDT",
    "ETHUSD":   "ETHUSDT",
    "ETH/USD":  "ETHUSDT",
    "BNBUSD":   "BNBUSDT",
    "SOLUSD":   "SOLUSDT",
    "EURUSD":   "EURUSDT",   # Binance has this pair
}

# Intervals supported by Binance klines API
INTERVAL_MAP: dict[str, str] = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "60m": "1h",
    "4h":  "4h",
    "1d":  "1d",
}

# How many minutes per bar for limit calculation
INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "60m": 60, "4h": 240, "1d": 1440,
}

BINANCE_API_BASE = "https://api.binance.com"


class BinanceLoader:
    """
    Fetches public OHLCV klines from Binance REST API.

    API key is optional for public market data endpoints.
    It is used for authenticated endpoints (account info, etc.) but
    candlestick data is fully public.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        tz: Optional[str] = None,
    ):
        self.api_key    = api_key    or os.getenv("BINANCE_API_KEY", "")
        self.secret_key = secret_key or os.getenv("BINANCE_SECRET_KEY", "")
        self.tz         = tz
        self._session   = requests.Session()
        if self.api_key:
            self._session.headers.update({"X-MBX-APIKEY": self.api_key})

    # ── Symbol helpers ─────────────────────────────────────────────────────
    def _resolve_symbol(self, symbol: str) -> str:
        """Normalise user symbol to Binance ticker (e.g. BTCUSD → BTCUSDT)."""
        upper = symbol.upper().strip()
        if upper in SYMBOL_MAP:
            return SYMBOL_MAP[upper]
        # Strip common suffixes / reformat
        upper = upper.replace("/", "").replace("=X", "").replace("-", "")
        if upper in SYMBOL_MAP:
            return SYMBOL_MAP[upper]
        # If already ends with USDT / BUSD keep it
        if upper.endswith("USDT") or upper.endswith("BUSD"):
            return upper
        # Default: append USDT
        return upper + "USDT"

    def _resolve_interval(self, interval: str) -> str:
        return INTERVAL_MAP.get(interval.lower(), interval)

    # ── Period → limit (number of bars) ───────────────────────────────────
    def _period_to_limit(self, period: str, interval: str) -> int:
        mins_per_bar = INTERVAL_MINUTES.get(self._resolve_interval(interval), 5)
        if period.endswith("d"):
            total_mins = int(period[:-1]) * 24 * 60
        elif period.endswith("h"):
            total_mins = int(period[:-1]) * 60
        else:
            total_mins = 7 * 24 * 60  # default 7 days
        limit = total_mins // mins_per_bar
        return min(max(limit, 50), 1000)  # Binance max = 1000

    # ── Main fetch ─────────────────────────────────────────────────────────
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "5m",
        period: str = "10d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data from Binance.

        Args:
            symbol:   e.g. 'BTCUSD', 'BTCUSDT', 'ETHUSDT'
            interval: '1m', '5m', '15m', '1h', '4h', '1d'
            period:   '5d', '10d', '30d', '60d'

        Returns:
            pd.DataFrame with columns [Open, High, Low, Close, Volume]
            and a UTC DatetimeIndex.
        """
        binance_symbol   = self._resolve_symbol(symbol)
        binance_interval = self._resolve_interval(interval)
        limit            = self._period_to_limit(period, interval)

        url = f"{BINANCE_API_BASE}/api/v3/klines"
        params = {
            "symbol":   binance_symbol,
            "interval": binance_interval,
            "limit":    limit,
        }

        resp = self._session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()

        if not raw:
            raise RuntimeError(
                f"Binance returned empty data for {binance_symbol} {binance_interval}"
            )

        # Binance kline format:
        # [open_time, open, high, low, close, volume, close_time, ...]
        df = pd.DataFrame(raw, columns=[
            "open_time", "Open", "High", "Low", "Close", "Volume",
            "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore",
        ])

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("open_time")
        df.index.name = "Datetime"

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype(float)

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        if self.tz:
            df.index = df.index.tz_convert(self.tz)

        return df

    # ── Account info (requires API key) ───────────────────────────────────
    def get_account_info(self) -> dict:
        """Fetch account balances (requires API key + secret)."""
        if not self.api_key or not self.secret_key:
            raise ValueError("API key and secret required for account info")

        ts = int(time.time() * 1000)
        params_str = urlencode({"timestamp": ts})
        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            params_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        resp = self._session.get(
            f"{BINANCE_API_BASE}/api/v3/account",
            params={"timestamp": ts, "signature": signature},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Ticker price ───────────────────────────────────────────────────────
    def get_ticker_price(self, symbol: str) -> float:
        """Get latest ticker price for a symbol."""
        binance_symbol = self._resolve_symbol(symbol)
        resp = self._session.get(
            f"{BINANCE_API_BASE}/api/v3/ticker/price",
            params={"symbol": binance_symbol},
            timeout=10,
        )
        resp.raise_for_status()
        return float(resp.json()["price"])

    # ── 24hr stats ────────────────────────────────────────────────────────
    def get_24hr_stats(self, symbol: str) -> dict:
        """Get 24-hour price statistics."""
        binance_symbol = self._resolve_symbol(symbol)
        resp = self._session.get(
            f"{BINANCE_API_BASE}/api/v3/ticker/24hr",
            params={"symbol": binance_symbol},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
