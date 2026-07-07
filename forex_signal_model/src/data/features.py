"""
src/data/features.py  —  Enhanced with SMC + Vantage Market strategies

Knowledge sources:
  1. ACY Securities — "Complete Step-by-Step Guide to Day Trading Gold (XAU/USD)
     with Smart Money Concepts (SMC)"
     Key concepts: Fair Value Gap (FVG), Liquidity Sweep, Displacement,
     Kill Zones (London/NY sessions), Previous Day High/Low (PDH/PDL),
     Asian session high/low, multi-timeframe bias (H1 direction, M5 execution)

  2. Vantage Markets — "Trading XAUUSD Tips and Strategies"
     Key concepts: EMA 21/50 trend bias, RSI divergence, MACD zero-line
     crossover, Supply/Demand zones (Rally-Base-Rally / Drop-Base-Drop),
     risk-on / risk-off sentiment, multi-timeframe confluence

All these signals have been translated into computable features below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Basic helpers ──────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up   = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up   = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ── ORIGINAL features (kept exactly as before) ────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for ML — original + SMC + Vantage Market features.

    Assumes df has columns: Open, High, Low, Close, Volume.
    """
    out = df.copy()

    # ── Original features ──────────────────────────────────────────────────
    out["ret_1"]     = out["Close"].pct_change(1)
    out["log_ret_1"] = np.log(out["Close"]).diff(1)

    out["ema_fast"] = ema(out["Close"], 12)
    out["ema_slow"] = ema(out["Close"], 26)
    out["ema_diff"] = (out["ema_fast"] - out["ema_slow"]) / out["Close"]

    out["rsi_14"]  = rsi(out["Close"], 14)
    out["atr_14"]  = atr(out["High"], out["Low"], out["Close"], 14)
    out["atr_pct"] = out["atr_14"] / out["Close"]

    out["range"]       = (out["High"] - out["Low"]) / out["Close"]
    out["body"]        = (out["Close"] - out["Open"]).abs() / out["Close"]
    out["upper_wick"]  = (out["High"] - out[["Open", "Close"]].max(axis=1)) / out["Close"]
    out["lower_wick"]  = (out[["Open", "Close"]].min(axis=1) - out["Low"]) / out["Close"]

    if "Volume" in out.columns:
        out["vol_chg"] = out["Volume"].pct_change(1)
        out["vol_z"]   = (
            (out["Volume"] - out["Volume"].rolling(30).mean())
            / out["Volume"].rolling(30).std()
        )

    # ══════════════════════════════════════════════════════════════════════
    # NEW FEATURES — learned from ACY + Vantage Market articles
    # ══════════════════════════════════════════════════════════════════════

    # ── [Vantage] EMA 21 / 50 trend bias ──────────────────────────────────
    # "Trade in the direction of the trend when price pulls back to these MAs"
    out["ema_21"]        = ema(out["Close"], 21)
    out["ema_50"]        = ema(out["Close"], 50)
    out["ema_200"]       = ema(out["Close"], 200)
    out["price_vs_ema21"]= (out["Close"] - out["ema_21"]) / out["Close"]   # >0 = price above ema21 (bullish bias)
    out["price_vs_ema50"]= (out["Close"] - out["ema_50"]) / out["Close"]
    out["ema21_vs_ema50"]= (out["ema_21"] - out["ema_50"]) / out["Close"]  # golden/death cross signal
    out["ema50_vs_ema200"]=(out["ema_50"] - out["ema_200"]) / out["Close"] # long-term trend bias

    # ── [Vantage] MACD zero-line crossover ────────────────────────────────
    # "MACD crossovers (crossing above/below zero line) signal momentum shifts"
    macd_line          = out["ema_fast"] - out["ema_slow"]
    macd_signal_line   = macd_line.ewm(span=9, adjust=False).mean()
    out["macd"]        = macd_line / out["Close"]
    out["macd_signal"] = macd_signal_line / out["Close"]
    out["macd_hist"]   = (macd_line - macd_signal_line) / out["Close"]
    out["macd_cross"]  = np.sign(macd_line) - np.sign(macd_line.shift(1))  # +2=bullish cross, -2=bearish

    # ── [Vantage] RSI momentum bands & divergence proxy ───────────────────
    # "RSI > 50 = bullish momentum; RSI divergence = early reversal warning"
    out["rsi_gt50"]   = (out["rsi_14"] > 50).astype(int)         # 1=bullish momentum
    out["rsi_ob"]     = (out["rsi_14"] > 70).astype(int)         # overbought
    out["rsi_os"]     = (out["rsi_14"] < 30).astype(int)         # oversold
    out["rsi_slope"]  = out["rsi_14"].diff(3)                     # RSI direction last 3 bars
    # RSI divergence proxy: price going up but RSI going down
    price_slope_3     = out["Close"].pct_change(3)
    rsi_slope_3       = out["rsi_14"].diff(3)
    out["rsi_div"]    = np.sign(price_slope_3) - np.sign(rsi_slope_3)  # +2 / -2 / 0

    # ── [ACY/SMC] Fair Value Gap (FVG) ────────────────────────────────────
    # "A strong displacement leaves a Fair Value Gap — pocket of unfilled orders"
    # FVG = gap between candle[i-2].High and candle[i].Low  (bullish FVG)
    #      or candle[i-2].Low  and candle[i].High (bearish FVG)
    bullish_fvg = out["Low"] - out["High"].shift(2)      # >0 = gap above = bullish FVG
    bearish_fvg = out["Low"].shift(2) - out["High"]      # >0 = gap below = bearish FVG
    out["fvg_bull"]   = bullish_fvg.clip(lower=0) / out["Close"]  # normalized gap size
    out["fvg_bear"]   = bearish_fvg.clip(lower=0) / out["Close"]
    out["fvg_net"]    = (out["fvg_bull"] - out["fvg_bear"])        # net FVG bias

    # ── [ACY/SMC] Displacement (strong impulse move) ──────────────────────
    # "Look for a strong impulse move away from the sweep"
    # Displacement = current candle body >> recent average body (aggressive move)
    avg_body_10        = out["body"].rolling(10).mean()
    out["displacement"]= (out["body"] / (avg_body_10 + 1e-9)).clip(upper=5)  # >2 = strong move

    # ── [ACY/SMC] Liquidity Sweep ─────────────────────────────────────────
    # "Look for stop runs at marked highs/lows"
    # Sweep = price pierces a recent swing high/low then closes back inside
    swing_high_20  = out["High"].rolling(20).max().shift(1)
    swing_low_20   = out["Low"].rolling(20).min().shift(1)
    out["sweep_high"] = ((out["High"] > swing_high_20) & (out["Close"] < swing_high_20)).astype(int)
    out["sweep_low"]  = ((out["Low"]  < swing_low_20)  & (out["Close"] > swing_low_20)).astype(int)
    out["liquidity_sweep"] = out["sweep_high"] - out["sweep_low"]  # +1=high swept, -1=low swept

    # ── [ACY/SMC] Previous Day High / Low (PDH/PDL) ───────────────────────
    # "Mark PDH/PDL as untapped key levels — price reacts at these zones"
    if isinstance(out.index, pd.DatetimeIndex):
        # Daily OHLC for PDH/PDL computation
        daily = out["Close"].resample("1D").ohlc()
        pdh   = daily["high"].shift(1).reindex(out.index, method="ffill")
        pdl   = daily["low"].shift(1).reindex(out.index, method="ffill")
        out["dist_pdh"]    = (out["Close"] - pdh) / out["Close"]   # dist to prev day high
        out["dist_pdl"]    = (out["Close"] - pdl) / out["Close"]   # dist to prev day low
        out["near_pdh"]    = (out["dist_pdh"].abs() < 0.001).astype(int)  # within 0.1%
        out["near_pdl"]    = (out["dist_pdl"].abs() < 0.001).astype(int)
    else:
        # Fallback: use rolling 288 bars (≈ 1 trading day at 5m bars)
        out["dist_pdh"] = (out["Close"] - out["High"].rolling(288).max().shift(1)) / out["Close"]
        out["dist_pdl"] = (out["Close"] - out["Low"].rolling(288).min().shift(1))  / out["Close"]
        out["near_pdh"] = (out["dist_pdh"].abs() < 0.001).astype(int)
        out["near_pdl"] = (out["dist_pdl"].abs() < 0.001).astype(int)

    # ── [ACY/SMC] Asian Session High / Low ───────────────────────────────
    # "Asian session (low volume) builds liquidity. London breaks it."
    if isinstance(out.index, pd.DatetimeIndex):
        utc_hour = out.index.tz_convert("UTC").hour if out.index.tz else out.index.hour
        # Asian session = 00:00–08:00 UTC
        asian_mask = (utc_hour >= 0) & (utc_hour < 8)
        # Rolling 96 bars (8 hrs at 5m) as Asian range proxy
        out["asian_range"] = (
            out["High"].where(pd.Series(asian_mask, index=out.index))
            .rolling(96, min_periods=1).max()
            - out["Low"].where(pd.Series(asian_mask, index=out.index))
            .rolling(96, min_periods=1).min()
        ) / out["Close"]
    else:
        out["asian_range"] = (
            out["High"].rolling(96, min_periods=1).max()
            - out["Low"].rolling(96, min_periods=1).min()
        ) / out["Close"]

    # ── [ACY/SMC] Kill Zone — London & New York session flags ─────────────
    # "Trade during London (12AM–6AM EST) and New York (9:30AM–12PM, 1PM–4PM EST)"
    # EST = UTC-5 (standard) / UTC-4 (daylight saving) — we use UTC offsets
    if isinstance(out.index, pd.DatetimeIndex):
        try:
            utc_idx  = out.index.tz_convert("UTC")
        except Exception:
            utc_idx  = out.index
        utc_h    = utc_idx.hour + utc_idx.minute / 60.0
        # London Kill Zone: 05:00–10:00 UTC (= midnight–6AM EST in winter)
        out["london_kz"] = ((utc_h >= 5) & (utc_h < 10)).astype(int)
        # New York AM Kill Zone: 14:30–17:00 UTC (= 9:30AM–12PM EST)
        out["ny_am_kz"]  = ((utc_h >= 14.5) & (utc_h < 17)).astype(int)
        # New York PM Kill Zone: 18:00–21:00 UTC (= 1PM–4PM EST)
        out["ny_pm_kz"]  = ((utc_h >= 18) & (utc_h < 21)).astype(int)
        # Combined: is this bar in any kill zone?
        out["in_kill_zone"] = (out["london_kz"] | out["ny_am_kz"] | out["ny_pm_kz"]).astype(int)
    else:
        # Fallback: no timezone info, skip kill zone features
        out["london_kz"]    = 0
        out["ny_am_kz"]     = 0
        out["ny_pm_kz"]     = 0
        out["in_kill_zone"] = 0

    # ── [Vantage] Supply / Demand zone — Rally-Base-Rally ─────────────────
    # "Look for areas where price moved away aggressively (order blocks)"
    # Proxy: a consolidation (low ATR) followed by a strong breakout
    atr_5           = atr(out["High"], out["Low"], out["Close"], 5)
    atr_20          = atr(out["High"], out["Low"], out["Close"], 20)
    out["rba_proxy"] = (atr_5 / (atr_20 + 1e-9))    # <0.7 = consolidation; >1.5 = breakout

    # ── [Vantage] Multi-timeframe bias proxy ──────────────────────────────
    # "Use H4/Daily for trend, M5 for entry"
    # Proxy: 48-bar EMA (≈4h at 5m bars) vs current price
    out["ema_h4_proxy"]   = ema(out["Close"], 48)
    out["htf_bias"]       = np.sign(out["Close"] - out["ema_h4_proxy"])   # +1 above, -1 below

    # ── [Vantage] Stochastic Oscillator ──────────────────────────────────
    # Additional momentum indicator often cited for XAUUSD
    low_14   = out["Low"].rolling(14).min()
    high_14  = out["High"].rolling(14).max()
    out["stoch_k"] = 100 * (out["Close"] - low_14) / (high_14 - low_14 + 1e-9)
    out["stoch_d"] = out["stoch_k"].rolling(3).mean()
    out["stoch_cross"] = np.sign(out["stoch_k"] - out["stoch_d"])

    # ── Time-of-day cyclical encoding (original) ──────────────────────────
    if isinstance(out.index, pd.DatetimeIndex):
        try:
            h = out.index.tz_convert("UTC").hour
        except Exception:
            h = out.index.hour
        out["hour_sin"] = np.sin(2 * np.pi * h / 24)
        out["hour_cos"] = np.cos(2 * np.pi * h / 24)
        dow = out.index.dayofweek
        out["dow_sin"]  = np.sin(2 * np.pi * dow / 5)
        out["dow_cos"]  = np.cos(2 * np.pi * dow / 5)
    else:
        out["hour_sin"] = 0.0
        out["hour_cos"] = 1.0
        out["dow_sin"]  = 0.0
        out["dow_cos"]  = 1.0

    # ══════════════════════════════════════════════════════════════════════
    # NEWEST FEATURES — XAUUSD strategies from ResearchGate & Scribd Guides
    # ══════════════════════════════════════════════════════════════════════

    # 1. Bollinger Bands (20, 2)
    bb_mid = out["Close"].rolling(20).mean()
    bb_std = out["Close"].rolling(20).std()
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std
    out["bb_pct"] = (out["Close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"] + 1e-9)
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / (bb_mid + 1e-9)
    out["bb_upper_cross"] = (out["Close"] > out["bb_upper"]).astype(int)
    out["bb_lower_cross"] = (out["Close"] < out["bb_lower"]).astype(int)

    # 2. ADX (14) (Average Directional Index) for trend strength
    up_move = out["High"] - out["High"].shift(1)
    down_move = out["Low"].shift(1) - out["Low"]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # True Range (TR)
    prev_close = out["Close"].shift(1)
    tr_series = pd.concat([
        (out["High"] - out["Low"]).abs(),
        (out["High"] - prev_close).abs(),
        (out["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    
    smooth_tr = tr_series.ewm(alpha=1/14, adjust=False).mean()
    smooth_plus_dm = pd.Series(plus_dm, index=out.index).ewm(alpha=1/14, adjust=False).mean()
    smooth_minus_dm = pd.Series(minus_dm, index=out.index).ewm(alpha=1/14, adjust=False).mean()
    
    plus_di = 100 * smooth_plus_dm / (smooth_tr + 1e-9)
    minus_di = 100 * smooth_minus_dm / (smooth_tr + 1e-9)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    out["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di

    # 3. Optimal Trade Entry (OTE) Fib Retracement proxies (from recent 50 bars)
    low_50 = out["Low"].rolling(50).min()
    high_50 = out["High"].rolling(50).max()
    out["fib_ratio"] = (out["Close"] - low_50) / (high_50 - low_50 + 1e-9)
    out["ote_zone"] = ((out["fib_ratio"] >= 0.618) & (out["fib_ratio"] <= 0.786)).astype(int)

    return out


# ── make_horizon_labels stays exactly as before ───────────────────────────

def make_horizon_labels(
    df: pd.DataFrame,
    horizon_bars: int,
    tp_atr_mult: float = 1.0,
    sl_atr_mult: float = 1.0,
    fee_bps: float = 1.0,
) -> pd.DataFrame:
    """Label trades as up/down/flat based on whether TP/SL is hit first.

    For scalp: horizon_bars corresponds to >= 5 minutes, depending on interval.

    Returns df with columns:
    - y (1=buy, 0=sell)
    - y_side (int)
    - y_conf (float proxy confidence)

    This is a pragmatic labeling approach; you should validate on your broker fills.
    """
    out = df.copy()

    if "atr_14" not in out.columns:
        raise ValueError("Call add_features() before make_horizon_labels().")

    close = out["Close"].values
    high  = out["High"].values
    low   = out["Low"].values
    atrv  = out["atr_14"].values

    y      = np.full(len(out), np.nan)
    y_conf = np.full(len(out), np.nan)

    fee = fee_bps / 10000.0

    for i in range(len(out) - horizon_bars):
        entry    = close[i]
        atr_here = atrv[i]
        if np.isnan(atr_here) or atr_here <= 0:
            continue

        tp   = entry + tp_atr_mult * atr_here
        sl   = entry - sl_atr_mult * atr_here
        tp_s = entry - tp_atr_mult * atr_here
        sl_s = entry + sl_atr_mult * atr_here

        h_slice = high[i + 1 : i + 1 + horizon_bars]
        l_slice = low[i + 1  : i + 1 + horizon_bars]

        buy_tp_idx   = np.where(h_slice >= tp)[0]
        buy_sl_idx   = np.where(l_slice <= sl)[0]
        sell_tp_idx  = np.where(l_slice <= tp_s)[0]
        sell_sl_idx  = np.where(h_slice >= sl_s)[0]

        buy_wins  = False
        sell_wins = False
        buy_first = np.inf
        sell_first= np.inf

        if buy_tp_idx.size > 0 and buy_sl_idx.size > 0:
            buy_wins  = buy_tp_idx[0] < buy_sl_idx[0]
            buy_first = buy_tp_idx[0] if buy_wins else buy_sl_idx[0]
        elif buy_tp_idx.size > 0:
            buy_wins  = True
            buy_first = buy_tp_idx[0]
        elif buy_sl_idx.size > 0:
            buy_wins  = False
            buy_first = buy_sl_idx[0]

        if sell_tp_idx.size > 0 and sell_sl_idx.size > 0:
            sell_wins  = sell_tp_idx[0] < sell_sl_idx[0]
            sell_first = sell_tp_idx[0] if sell_wins else sell_sl_idx[0]
        elif sell_tp_idx.size > 0:
            sell_wins  = True
            sell_first = sell_tp_idx[0]
        elif sell_sl_idx.size > 0:
            sell_wins  = False
            sell_first = sell_sl_idx[0]

        if buy_wins and not sell_wins:
            y[i]      = 1
            y_conf[i] = 1 / (1 + buy_first)
        elif sell_wins and not buy_wins:
            y[i]      = 0
            y_conf[i] = 1 / (1 + sell_first)
        elif buy_wins and sell_wins:
            if buy_first < sell_first:
                y[i]      = 1
                y_conf[i] = 1 / (1 + buy_first)
            else:
                y[i]      = 0
                y_conf[i] = 1 / (1 + sell_first)
        else:
            # Fallback: if neither TP nor SL was hit in horizon_bars,
            # label based on sign of close difference at the end of horizon.
            future_close = close[i + horizon_bars]
            if future_close > entry:
                y[i]      = 1
                y_conf[i] = 0.5  # lower confidence since TP wasn't hit
            elif future_close < entry:
                y[i]      = 0
                y_conf[i] = 0.5
            else:
                y[i]      = np.nan
                y_conf[i] = np.nan

    out["y"]      = y
    out["y_conf"] = y_conf
    out = out.dropna(subset=["y"])
    out["y"] = out["y"].astype(int)
    return out
