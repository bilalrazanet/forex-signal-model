# Detailed Setup Guide

## Prerequisites

- Python 3.8+
- Git
- pip (Python package manager)
- Virtual environment tool (venv or conda)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/bilalrazanet/forex-signal-model.git
cd forex-signal-model
```

### 2. Create Virtual Environment

**Using venv (recommended):**
```bash
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

**Using conda:**
```bash
conda create -n forex-signals python=3.10
conda activate forex-signals
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Optional: Development dependencies**
```bash
pip install pytest pytest-cov black flake8 ipython jupyter
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Data source configuration
DATA_SOURCE=binance  # or 'yahoo'

# AlphaVantage (optional, for alternative data)
ALPHAVANTAGE_API_KEY=your_api_key_here

# Flask configuration
FLASK_ENV=development
FLASK_DEBUG=1
FLASK_PORT=5000

# Model configuration
MODEL_PATH=./model.joblib
```

## Verification

### Verify Installation

```bash
python -c "import pandas, sklearn, xgboost, flask; print('All dependencies installed!')"
```

### Run Tests

```bash
pytest tests/ -v
```

### Check Data Connection

```bash
python -c "from src.data.binance_loader import BinanceLoader; bl = BinanceLoader(); print(bl.get_ticker_price('BTCUSDT'))"
```

## Quick Start

### Training a Model

```bash
# Single symbol
python scripts/train.py --symbol BTCUSDT --interval 5m

# Multiple symbols
python scripts/train.py --symbols BTCUSDT,ETHUSDT --interval 5m --lookback 500

# With Yahoo Finance
python scripts/train.py --data_source yahoo --symbol EURUSD=X --interval 5m
```

### Live Prediction

```bash
python scripts/live_predict.py --symbol BTCUSDT --interval 5m --check_interval 300
```

### Dashboard

```bash
python app.py
# Open http://localhost:5000 in your browser
```

## Troubleshooting

### "Module not found" errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### API connection issues
- Check internet connection
- Verify API endpoints are accessible
- For Binance: https://api.binance.com/api/v3/ping
- For Yahoo Finance: ensure yfinance can download data

### Model not loading
- Check `model.joblib` exists in project root
- Retrain model: `python scripts/train.py --symbol BTCUSDT --interval 5m`

### Permission denied on Windows
- Run PowerShell as Administrator
- Or use: `python -m scripts.train --symbol BTCUSDT`

## Performance Optimization

### Speed Up Training
```bash
# Reduce lookback period
python scripts/train.py --symbol BTCUSDT --lookback 200

# Reduce interval
python scripts/train.py --symbol BTCUSDT --interval 15m
```

### Reduce Memory Usage
```bash
# Process one symbol at a time
python scripts/train.py --symbol BTCUSDT --interval 1h
```

## Next Steps

1. Read [README.md](README.md) for project overview
2. Check [scripts/](scripts/) for available tools
3. Review [src/models/train.py](src/models/train.py) for model configuration
4. See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
