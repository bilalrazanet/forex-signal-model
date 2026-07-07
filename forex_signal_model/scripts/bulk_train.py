import sys
import os
import requests
import time

# List of symbols to train on
SYMBOLS = [
    # Crypto
    "BTCUSD", "ETHUSD", "BNBUSD", "SOLUSD",
    # Commodities
    "XAUUSD", "SILVER", "OIL",
    # Forex
    "EURUSD", "GBPUSD", "USDJPY",
    # Indices
    "SPX", "NDX"
]

INTERVALS = ["5m", "15m", "1h"]
PERIODS = {
    "5m": "30d",
    "15m": "60d",
    "1h": "90d"
}

def train_all():
    print("Starting bulk training for AI Bot...")
    for sym in SYMBOLS:
        for iv in INTERVALS:
            period = PERIODS[iv]
            print(f"Training {sym} on {iv} interval ({period} history)...")
            
            payload = {
                "symbol": sym,
                "interval": iv,
                "period": period,
                "horizon_minutes": 15
            }
            
            try:
                # We assume the Flask app is running locally for the API
                response = requests.post("http://localhost:5000/api/train", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    print(f"  -> SUCCESS! Trained on {data.get('rows_trained')} rows. AUC: {data.get('cv_auc')}")
                else:
                    print(f"  -> FAILED: {response.text}")
            except Exception as e:
                print(f"  -> FAILED: Cannot connect to API. Is app.py running? {e}")
                
            time.sleep(2) # Prevent rate limits (especially for Yahoo Finance)

if __name__ == "__main__":
    train_all()
