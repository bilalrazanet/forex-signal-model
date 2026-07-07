"""Train BTC, ETH, XAU (PAXG) via Binance — all horizons."""
import requests, time

symbols = [
    ("BTCUSD",  "5m", "30d", 15, "BTC/USD  (Binance)"),
    ("ETHUSD",  "5m", "30d", 15, "ETH/USD  (Binance)"),
    ("XAUUSD",  "5m", "30d", 15, "XAU/USD  (Binance PAXG, 5m bars)"),
]

print("\n" + "="*60)
print("  ForexAI — Training (Binance-Only Edition)")
print("="*60)

for sym, iv, per, horizon, label in symbols:
    print(f"\n  [{label}]")
    try:
        r = requests.post(
            "http://localhost:5000/api/train",
            json={"symbol": sym, "interval": iv, "period": per,
                  "horizon_minutes": horizon},
            timeout=180,
        )
        d = r.json()
        err = d.get("error")
        if err:
            print(f"  ERROR: {err[:250]}")
        else:
            auc   = round(d.get("cv_auc") or 0, 4)
            rows  = d.get("rows_trained")
            feats = d.get("feature_count")
            hor   = d.get("horizon_minutes")
            src   = d.get("source", "?")
            print(f"  OK | rows={rows} | AUC={auc} | features={feats} | horizon={hor}min | {src}")
    except Exception as ex:
        print(f"  EXCEPTION: {ex}")
    time.sleep(1)

print("\n" + "="*60)
print("  Done! Open: http://localhost:5000")
print("="*60 + "\n")
