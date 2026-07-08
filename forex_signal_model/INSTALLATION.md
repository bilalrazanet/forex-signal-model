# Installation Guide

## Prerequisites

- Python 3.9 or higher
- pip
- Git
- Internet access for live market data

## 1. Clone the repository

```bash
git clone https://github.com/bilalrazanet/forex-signal-model.git
cd forex-signal-model
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Verify the setup

```bash
python -c "import pandas, numpy, xgboost, flask; print('Dependencies installed successfully')"
```

## 5. Run the project

### Train a model

```bash
python scripts/train.py --symbol BTCUSDT --interval 5m
```

### Launch the dashboard

```bash
python app.py
```

Open http://localhost:5000 in your browser.

## Optional environment variables

Create a .env file in the project root if needed:

```env
FLASK_ENV=development
FLASK_DEBUG=1
FLASK_PORT=5000
MODEL_PATH=./model.joblib
```

## Troubleshooting

- If a module is missing, reinstall dependencies with `pip install -r requirements.txt`.
- If the dashboard port is occupied, use `python app.py --port 5001`.
- If the model file is missing, retrain it with the command above.
