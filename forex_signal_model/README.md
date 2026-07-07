# Forex Signal Model (Scalp + Spot) — XNS/XM (MetaTrader-style)

This project is scaffolded to:
- Fetch live/near-live Forex data (default: Yahoo Finance via `yfinance`; extensible)
- Train a predictive model that outputs **trade signals** suitable for **scalp (>= 5 minutes)** and **spot** horizons
- Provide a broker bridge layer for **MetaTrader 4/5-style** execution (XM/XNS commonly use MetaTrader)

> Note: “Perfect” prediction is not realistically achievable; this scaffold focuses on robust engineering, proper labeling, backtesting hooks, and extensibility.

## Project layout
- `src/`
  - `data/` data loaders + feature engineering
  - `models/` model definitions + training
  - `signals/` signal generation + thresholding
  - `broker/` MetaTrader bridge stubs
- `scripts/`
  - `train.py` training entrypoint
  - `live_predict.py` live inference loop

## Quick start
1. Create venv
   - `python -m venv .venv`
   - `.
     .venv\Scripts\activate`
2. Install deps
   - `pip install -r requirements.txt`
3. Train
   - `python scripts/train.py --symbol EURUSD=X --interval 5m`
   - `python scripts/train.py --symbols EURUSD=X,GBPUSD=X --interval 5m`
   - `python scripts/train.py --data_source alphavantage --symbol EURUSD=X --interval 5m --api_key YOUR_API_KEY`
4. Live predict
   - `python scripts/live_predict.py --symbol EURUSD=X --interval 5m`
   - `python scripts/live_predict.py --data_source alphavantage --symbol EURUSD=X --interval 5m --api_key YOUR_API_KEY`

## Broker integration (XM/XNS)
XM/XNS execution typically uses MetaTrader (MT4/MT5). This repo includes a **bridge interface** so you can plug in:
- An MT4/MT5 Expert Advisor (EA) that calls a local REST endpoint, OR
- A Python service that writes signals to a file/pipe which your EA reads.

Implementations are in `src/broker/`.

### MT4/MT5 execution notes
- Use an EA to read `signals_out.json` or POST to a local HTTP endpoint.
- `scripts/live_predict.py` can publish JSON signals via `FileSignalBroker` or `HttpSignalBroker`.
- Keep the EA and signal writer on the same LAN or local machine for low latency.

