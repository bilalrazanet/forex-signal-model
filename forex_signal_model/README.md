# Forex Signal Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

ML-powered **forex & cryptocurrency trading signal predictor** with XGBoost, real-time dashboard, and backtesting support. Built for scalp (5m+) and spot trading strategies.

<div align="center">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen" alt="Status">
  <img src="https://img.shields.io/badge/Data%20Sources-Binance%20%7C%20Yahoo%20Finance-orange" alt="Data">
  <img src="https://img.shields.io/badge/Model-XGBoost-blue" alt="Model">
</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Configuration](#-configuration)
- [Backtesting](#-backtesting)
- [FAQ](#-faq)
- [Contributing](#-contributing)
- [License](#-license)
- [Disclaimer](#-disclaimer)

---

## ✨ Features

### Core ML & Signals
- **XGBoost Classifier** — Production ML model for binary trade signals (BUY/SELL/HOLD)
- **Multi-Timeframe Support** — 1m, 5m, 15m, 30m, 1h, 4h, 1d candles
- **Real-Time Predictions** — Live signal generation with entry/exit levels
- **Feature Engineering** — 50+ technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
- **Model Interpretability** — SHAP values for feature importance analysis

### Data Sources
- **Binance API** — Real-time crypto price data (BTCUSDT, ETHUSDT, SOL, ADA, LINK, etc.)
- **Yahoo Finance** — Forex & commodity pairs (EURUSD=X, GBPUSD=X, Gold, Oil, S&P 500)
- **Extensible Architecture** — Easily add AlphaVantage, IQFeed, or custom data sources

### Web Dashboard
- **Flask-based UI** — Interactive web interface at `localhost:5000`
- **Live Charts** — OHLCV candles with technical indicators
- **Signal Display** — Real-time entry/exit recommendations
- **Training Interface** — Train models directly from the dashboard
- **24h Statistics** — Price changes, volume, support/resistance levels

### Advanced Tools
- **Backtesting Pipeline** — Replay historical data with signal evaluation
- **Bulk Training** — Train multiple symbols in parallel
- **MetaTrader Bridge** — Connect to MT4/MT5 Expert Advisors
- **Error Handling** — Production-grade logging and exception management

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip or conda
- Internet connection (for live data)

### Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/bilalrazanet/forex-signal-model.git
cd forex-signal-model

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train a Model (1 minute)

```bash
# Single symbol, 5-minute candles
python scripts/train.py --symbol BTCUSDT --interval 5m

# Multiple symbols
python scripts/train.py --symbols BTCUSDT,ETHUSDT --interval 5m

# Yahoo Finance (Forex)
python scripts/train.py --data_source yahoo --symbol EURUSD=X --interval 15m
```

### Launch Dashboard (30 seconds)

```bash
python app.py
```

Open browser: **http://localhost:5000**

---

## 📦 Installation

See [INSTALLATION.md](INSTALLATION.md) for detailed instructions including:
- Virtual environment setup
- Dependency installation
- Environment variable configuration
- Troubleshooting common issues

---

## 💻 Usage

### Training Models

#### Basic Training
```bash
python scripts/train.py \
  --symbol BTCUSDT \
  --interval 5m \
  --lookback 500 \
  --train_split 0.8
```

**Parameters:**
- `--symbol` — Trading pair (BTCUSDT, EURUSD=X, etc.)
- `--interval` — Candle size (1m, 5m, 15m, 30m, 1h, 4h, 1d)
- `--lookback` — Number of historical candles (default: 500)
- `--train_split` — Train/test ratio (default: 0.8)
- `--data_source` — 'binance' or 'yahoo' (default: binance)

#### Multi-Symbol Training
```bash
python scripts/train.py \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT \
  --interval 15m
```

#### Bulk Training (All Symbols)
```bash
python scripts/bulk_train.py --interval 5m
```

### Live Predictions

```bash
python scripts/live_predict.py \
  --symbol BTCUSDT \
  --interval 5m \
  --check_interval 300
```

### Web Dashboard

```bash
python app.py --port 5000
```

---

## 🏗️ Architecture

### Project Structure

```
forex-signal-model/
├── src/
│   ├── data/
│   │   ├── binance_loader.py      # Binance API integration
│   │   ├── data_loader.py         # Yahoo Finance & data fetching
│   │   ├── features.py            # Technical indicator engineering
│   │   └── utils.py               # Data utilities
│   ├── models/
│   │   └── train.py               # XGBoost model training
│   ├── signals/
│   │   └── signals.py             # Signal generation logic
│   └── broker/
│       └── metatrader_bridge.py   # MT4/MT5 integration
├── scripts/
│   ├── train.py                   # Single model training
│   ├── bulk_train.py              # Multi-symbol training
│   ├── live_predict.py            # Real-time prediction loop
│   └── backtest.py                # Backtesting engine
├── dashboard/
│   └── index.html                 # Web UI
├── tests/
│   └── test_data_loader.py        # Unit tests
├── app.py                         # Flask web server
├── model.joblib                   # Trained XGBoost model
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🔧 Configuration

Create a `.env` file in the project root:

```env
DATA_SOURCE=binance
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_here
FLASK_ENV=development
FLASK_DEBUG=1
FLASK_PORT=5000
MODEL_PATH=./model.joblib
```

---

## 📈 Backtesting

```bash
python scripts/backtest.py \
  --symbol BTCUSDT \
  --interval 5m \
  --start_date 2023-01-01 \
  --end_date 2024-01-01
```

---

## ❓ FAQ

### Q: What symbols are supported?

**A:** Any symbol on Binance (crypto) or Yahoo Finance (forex, commodities, indices).

**Popular Crypto:** BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, LINKUSDT, LTCUSDT, AVAXUSDT

**Popular Forex:** EURUSD=X, GBPUSD=X, USDJPY=X, USDCHF=X, AUDUSD=X, USDCAD=X, NZDUSD=X

**Commodities:** SI=F (Silver), CL=F (Oil), GC=F (Gold)

### Q: How do I add custom indicators?

Edit [src/data/features.py](src/data/features.py) and add your feature function.

### Q: Is this profitable?

**A:** Past performance ≠ future results. Backtesting shows ~65% win rate in specific conditions. **Always test thoroughly and start with small positions.**

### Q: Can I deploy to production?

**A:** Yes, but use proper risk management, paper trading first, and periodic model retraining.

---

## 📚 Documentation

- **[INSTALLATION.md](INSTALLATION.md)** — Setup guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Development guidelines
- **[SETUP.md](SETUP.md)** — Configuration reference

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 License

Licensed under MIT — see [LICENSE](LICENSE).

---

## ⚠️ Disclaimer

**IMPORTANT: Educational & Research Purpose Only**

This software is provided for educational purposes. Trading carries substantial risk of loss. The authors make no representations about profitability.

**By using this software, you acknowledge:**
1. No guarantee of profit
2. Past performance ≠ future results
3. Personal responsibility for trading decisions
4. This is NOT financial advice
5. Use at your own risk

**Always conduct your own research (DYOR), test with paper trading first, and never risk money you can't afford to lose.**

---

**Made with ❤️ by Bilal Raza**

⭐ If this project helped you, consider starring the repository!
