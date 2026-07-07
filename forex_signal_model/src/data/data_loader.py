from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf


@dataclass(frozen=True)
class YahooPairConfig:
    symbol: str  # e.g. "EURUSD=X"


class YahooFinanceLoader:
    """Fetches OHLCV from Yahoo Finance.

    Supports minute intervals that Yahoo provides.

    Note: Yahoo data is not guaranteed to be perfect for tick-level trading.
    """

    def __init__(self, tz: Optional[str] = None):
        self.tz = tz

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "5m",
        period: str = "5d",
    ) -> pd.DataFrame:
        df = yf.download(
            tickers=[symbol],
            period=period,
            interval=interval,
            progress=False,
            group_by="column",
        )

        # yfinance returns multi-index columns when tickers is a list.
        # We normalize to single level: columns = ['Open','High','Low','Close','Adj Close','Volume']
        if isinstance(df.columns, pd.MultiIndex):
            # pick first ticker
            df = df.xs(symbol, axis=1, drop_level=True)

        # Ensure datetime index is present
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=False)

        if self.tz:
            df = df.tz_convert(self.tz)

        df = df.rename(columns={"Adj Close": "Close"}) if "Adj Close" in df.columns else df
        df = df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]].copy()
        df = df.dropna()
        return df


def create_data_loader(source: str = "yahoo", api_key: Optional[str] = None, tz: Optional[str] = None) -> Any:
    source = source.strip().lower()
    if source == "yahoo":
        return YahooFinanceLoader(tz=tz)
    if source == "alphavantage":
        return AlphaVantageForexLoader(api_key=api_key, tz=tz)
    raise ValueError(f"Unsupported data source: {source}. Use 'yahoo' or 'alphavantage'.")


class AlphaVantageForexLoader:
    """Fetches OHLC data from AlphaVantage FX_INTRADAY.

    AlphaVantage offers a free tier with request limits. Set `ALPHAVANTAGE_API_KEY`
    or pass `api_key` manually.
    """

    API_URL = "https://www.alphavantage.co/query"
    INTERVAL_MAP = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
        "1h": "60min",
    }

    def __init__(self, api_key: Optional[str] = None, tz: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        self.tz = tz
        if not self.api_key:
            raise ValueError(
                "AlphaVantage API key required. Set ALPHAVANTAGE_API_KEY or pass --api_key."
            )

    def _parse_symbol(self, symbol: str) -> tuple[str, str]:
        symbol = symbol.upper().strip()
        if symbol.endswith("=X"):
            symbol = symbol[:-2]
        symbol = symbol.replace("/", "")
        if len(symbol) != 6:
            raise ValueError("AlphaVantage symbol must be 6 letters like EURUSD or EURUSD=X")
        return symbol[:3], symbol[3:]

    def _parse_interval(self, interval: str) -> str:
        interval = interval.strip().lower()
        if interval not in self.INTERVAL_MAP:
            raise ValueError(
                f"Unsupported interval for AlphaVantage: {interval}. "
                f"Supported: {', '.join(self.INTERVAL_MAP)}"
            )
        return self.INTERVAL_MAP[interval]

    def _filter_by_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        period = period.strip().lower()
        if period.endswith("d"):
            days = int(period[:-1])
            cutoff = df.index.max() - timedelta(days=days)
            return df[df.index >= cutoff]
        if period.endswith("h"):
            hours = int(period[:-1])
            cutoff = df.index.max() - timedelta(hours=hours)
            return df[df.index >= cutoff]
        return df

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "5m",
        period: str = "5d",
    ) -> pd.DataFrame:
        from_symbol, to_symbol = self._parse_symbol(symbol)
        interval_key = self._parse_interval(interval)

        resp = requests.get(
            self.API_URL,
            params={
                "function": "FX_INTRADAY",
                "from_symbol": from_symbol,
                "to_symbol": to_symbol,
                "interval": interval_key,
                "outputsize": "compact",
                "apikey": self.api_key,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()

        timeseries_key = next(
            (key for key in data if key.startswith("Time Series FX")), None
        )
        if timeseries_key is None:
            raise RuntimeError(f"AlphaVantage response missing time series data: {data}")

        df = pd.DataFrame.from_dict(data[timeseries_key], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.rename(
            columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
            }
        )
        df = df[["Open", "High", "Low", "Close"]].astype(float)
        df["Volume"] = 0

        if self.tz:
            df = df.tz_localize("UTC").tz_convert(self.tz)

        df = self._filter_by_period(df, period)
        return df


def fetch_multiple(symbols: List[str], interval: str, period: str) -> pd.DataFrame:
    """Convenience function to fetch multiple pairs at once.

    Returns a DataFrame with columns suffixed by symbol, plus timestamp index.
    """
    frames = []
    loader = YahooFinanceLoader()
    for sym in symbols:
        df = loader.fetch_ohlcv(sym, interval=interval, period=period)
        df = df.add_suffix(f"_{sym}")
        frames.append(df)
    out = pd.concat(frames, axis=1).sort_index()
    return out

