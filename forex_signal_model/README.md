# Forex Signal Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine-learning trading signal project for crypto, forex, and commodities with an XGBoost model, live prediction workflow, and a Flask-based dashboard. The current repository is focused on Binance market data for real-time analysis and signal generation.

---

## ✨ What’s included

- XGBoost-based signal model training and inference
- Binance OHLCV data loading and market statistics
- Yahoo Finance support for forex and commodity symbols
- Flask dashboard for live charts, signal display, and training workflow
- Feature engineering with technical indicators such as RSI, MACD, ATR, Bollinger Bands, and volume-based metrics
- Training, backtesting, and live prediction scripts
- MetaTrader bridge support for exporting signals

---

## 🚀 Quick start

### 1. Clone and install

```bash
git clone https://github.com/bilalrazanet/forex-signal-model.git
cd forex-signal-model
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model

```bash
python scripts/train.py --symbol BTCUSDT --interval 5m
```

### 3. Start the dashboard

```bash
python app.py
```

Then open http://localhost:5000.

---

## 🧠 Current capabilities

- Train models for single symbols or multiple symbols
- Run live prediction loops for ongoing signal generation
- View market data and trade signals from the built-in dashboard
- Use Binance for crypto and PAXGUSDT for XAU/USD gold exposure
- Use Yahoo Finance symbols such as EURUSD=X and other forex pairs

---

## 📁 Project layout

```text
forex-signal-model/
├── app.py
├── model.joblib
├── requirements.txt
├── scripts/
│   ├── train.py
│   ├── live_predict.py
│   └── backtest.py
├── src/
│   ├── broker/
│   ├── data/
│   ├── models/
│   └── signals/
├── dashboard/
├── tests/
└── README.md
```

---

## 📚 Documentation

- [INSTALLATION.md](INSTALLATION.md)
- [SETUP.md](SETUP.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)

---

## ⚠️ Disclaimer

This project is for educational and research purposes only. It does not provide guaranteed trading results and should not be used as financial advice.

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
