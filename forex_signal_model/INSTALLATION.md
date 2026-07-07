# Installation Guide

## Prerequisites

- **Python**: 3.9 or higher
- **pip**: Latest version
- **Git**: For version control (optional)

## Step-by-Step Installation

### 1. Clone or Download the Repository

```bash
git clone https://github.com/bilalrazanet/forex-signal-model.git
cd forex-signal-model
```

### 2. Create a Virtual Environment (Recommended)

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### On Windows (Command Prompt):
```cmd
python -m venv .venv
.\.venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **Data Processing**: numpy, pandas
- **Machine Learning**: scikit-learn, xgboost, shap
- **Data Sources**: yfinance, requests
- **Web Server**: Flask, Flask-Cors
- **Testing**: pytest
- **Utilities**: python-dotenv, joblib

### 4. Verify Installation

```bash
python -c "import pandas, numpy, xgboost, flask; print('✓ All dependencies installed')"
```

## Configuration

### Optional: API Keys for Enhanced Data

#### AlphaVantage (Forex/Crypto)
```bash
# Create .env file in project root
echo API_KEY=your_api_key_here > .env
```

Get a free key at: https://www.alphavantage.co/

#### Binance (Crypto Only)
No API key required for public market data. Optional for trading.

### Environment Variables

Create a `.env` file in the project root:

```env
# Binance API (optional)
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here

# AlphaVantage API (optional)
ALPHA_VANTAGE_API_KEY=your_key_here

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
```

## Running the Application

### Option 1: Train a Model

```bash
python scripts/train.py --symbol EURUSD=X --interval 5m
```

**Parameters:**
- `--symbol`: Trading pair (e.g., EURUSD=X, BTCUSDT)
- `--interval`: Candle interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)
- `--data_source`: 'yahoo' (default) or 'binance'
- `--api_key`: For alphavantage data source

### Option 2: Live Prediction Loop

```bash
python scripts/live_predict.py --symbol EURUSD=X --interval 5m
```

### Option 3: Start Web Dashboard

**Using Python:**
```bash
python app.py
```

**Using Batch File (Windows):**
```bash
run_dashboard.bat
```

Then open: **http://localhost:5000** in your browser.

## Troubleshooting

### Issue: `ModuleNotFoundError`

```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt
```

### Issue: `yfinance` Download Fails

The data source might be experiencing rate limits. Retry or switch to AlphaVantage:

```bash
python scripts/train.py --symbol EURUSD=X --interval 5m --data_source alphavantage --api_key YOUR_API_KEY
```

### Issue: Binance Data Not Available

Ensure you're using correct Binance trading symbols:
- BTCUSDT (Bitcoin)
- ETHUSDT (Ethereum)
- BNBUSDT (Binance Coin)

### Issue: Port 5000 Already in Use (Flask)

```bash
# Use a different port
python app.py --port 5001
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 2 GB | 8 GB |
| Disk Space | 500 MB | 2 GB |
| Python | 3.9 | 3.10+ |
| Internet | Required | Required (for live data) |

## Uninstalling

```bash
# Remove virtual environment
rm -r .venv  # macOS/Linux
rmdir /s .venv  # Windows
```

## Next Steps

- Read the [README.md](README.md) for project overview
- Check [scripts/](scripts/) for available commands
- Review [tests/](tests/) for usage examples
- Explore [dashboard/](dashboard/) for web interface

## Need Help?

- Open an issue: https://github.com/bilalrazanet/forex-signal-model/issues
- Check existing documentation
- Review code comments and docstrings

Happy trading! 🚀
