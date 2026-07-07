from __future__ import annotations

import os
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.data_loader import AlphaVantageForexLoader, create_data_loader, YahooFinanceLoader
from src.data.utils import parse_interval_minutes


def test_parse_interval_minutes():
    assert parse_interval_minutes("1m") == 1
    assert parse_interval_minutes("5m") == 5
    assert parse_interval_minutes("1h") == 60
    assert parse_interval_minutes("3h") == 180


def test_create_data_loader_yahoo():
    loader = create_data_loader("yahoo")
    assert isinstance(loader, YahooFinanceLoader)


@patch.dict(os.environ, {"ALPHAVANTAGE_API_KEY": "demo"})
def test_create_data_loader_alphavantage():
    loader = create_data_loader("alphavantage")
    assert isinstance(loader, AlphaVantageForexLoader)


@patch("src.data.data_loader.requests.get")
def test_alphavantage_fetch_ohlcv(mock_get):
    # Minimal response shape for parser
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "Time Series FX (5min)": {
            "2025-01-01 10:00:00": {
                "1. open": "1.1000",
                "2. high": "1.1010",
                "3. low": "1.0990",
                "4. close": "1.1005",
            },
            "2025-01-01 10:05:00": {
                "1. open": "1.1005",
                "2. high": "1.1015",
                "3. low": "1.1000",
                "4. close": "1.1010",
            },
        }
    }

    loader = AlphaVantageForexLoader(api_key="demo")
    df = loader.fetch_ohlcv("EURUSD", interval="5m", period="1d")

    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.iloc[0]["Open"] == 1.1
    assert df.iloc[-1]["Close"] == 1.1010
