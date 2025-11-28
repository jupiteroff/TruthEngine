#!/usr/bin/env python3
"""
Truth Engine v11 — Enhanced MALP predictor (Binance REST, auto-frequency, Tech Indicators)

Usage:
    python3 truthengine.py

Terminal prompts:
    Enter currency (e.g. BTC or BTCUSDT or ETH): BTC
    Enter prediction target (examples):
      - relative: 5m  (minutes), 2h (hours), 3d (days), 1M (months ~30d)
      - absolute UTC datetime: 2025-12-01 10:00
    (script will fetch history, train MALP, output forecast + CCC)

Dependencies:
    pip install requests numpy pandas scipy statsmodels
"""

import math
import time
import json
from datetime import datetime, timedelta, timezone
import requests
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------
# Config / defaults
# ---------------------------
BINANCE_REST = "https://api.binance.com/api/v3/klines"
SAMPLES = 800            # Increased samples for better training with more features
SEQ_LEN = 30             # window length in candles for features
MAX_PER_REQUEST = 1000   # Binance max limit per request
OUTPUT_TRAIN_CSV = "malp_v11_train.csv"

# Interval seconds mapping
INTERVALS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400, "1w": 604800, "1M": 2592000
}

# ---------------------------
# Helpers: parse user input
# ---------------------------
def normalize_symbol(inp):
    s = inp.strip().upper()
    if s.endswith("USDT") or s.endswith("BUSD") or s.endswith("USD"):
        return s
    # default to USDT pair
    return s + "USDT"

def parse_target(text):
    """
    Accept strings like '5m', '2h', '3d', '1M' or absolute 'YYYY-MM-DD HH:MM'
    Returns horizon_seconds (int) and target_datetime (UTC)
    """
    t = text.strip()
    # try absolute datetime
    try:
        # if user provided only date, allow 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'
        if len(t) >= 10 and (t[4] == '-' and t[7] == '-'):
            # try parse with time
            dt = None
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(t, fmt)
                    break
                except:
                    pass
            if dt is None:
                raise ValueError
            # assume UTC if no tz
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int((dt - datetime.now(timezone.utc)).total_seconds()), dt.astimezone(timezone.utc)
    except Exception:
        pass
    # relative parsing
    num = ""
    unit = ""
    for ch in t:
        if ch.isdigit():
            num += ch
        else:
            unit += ch
    if num == "":
        raise ValueError("Invalid target format")
    n = float(num)
    unit = unit.strip().lower()
    if unit in ("m","min","mins","minute","minutes"):
        secs = int(n * 60)
    elif unit in ("h","hr","hour","hours"):
        secs = int(n * 3600)
    elif unit in ("d","day","days"):
        secs = int(n * 86400)
    elif unit in ("M","mon","month","months"):
        secs = int(n * 2592000)  # approximate 30 days
    else:
        raise ValueError("Unknown time unit; use m/h/d/M or absolute datetime")
    target_dt = datetime.now(timezone.utc) + timedelta(seconds=secs)
    return secs, target_dt

# ---------------------------
# Choose interval by horizon
# ---------------------------
def choose_interval(horizon_seconds):
    if horizon_seconds <= 6 * 3600:
        return "1m"
    if horizon_seconds <= 7 * 24 * 3600:
        return "1h"
    return "1d"

# ---------------------------
# Technical Indicators
# ---------------------------
def add_technical_indicators(df):
    """
    Adds RSI, MACD, Bollinger Bands to the dataframe.
    """
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50) # Neutral fill

    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
    
    # Fill NaNs from rolling windows
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    return df

# ---------------------------
# Binance REST klines fetch
# ---------------------------
def fetch_klines(symbol, interval, required_candles):
    """
    Fetch 'required_candles' most recent candles for symbol@interval.
    Returns dataframe with columns: open_time, open, high, low, close, volume, close_time
    """
    limit = min(MAX_PER_REQUEST, required_candles)
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_REST, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    candles = data
    # if need more than limit, fetch earlier batches using startTime
    while len(candles) < required_candles:
        # earliest openTime in current candles
        earliest = int(candles[0][0])
        # request previous batch ending before earliest
        params = {"symbol": symbol, "interval": interval, "endTime": earliest - 1, "limit": limit}
        resp = requests.get(BINANCE_REST, params=params, timeout=15)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        # prepend
        candles = batch + candles
        # safety sleep to avoid rate limit
        time.sleep(0.15)
    # trim to required_candles (keep most recent)
    candles = candles[-required_candles:]
    # build DataFrame
    df = pd.DataFrame(candles, columns=[
        "open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base_vol","taker_quote_vol","ignore"
    ])
    # convert types
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    numeric_cols = ["open","high","low","close","volume"]
    for c in numeric_cols:
        df[c] = df[c].astype(float)
    
    return df[["open_time","open","high","low","close","volume","close_time"]]

# ---------------------------
# Build supervised dataset
# ---------------------------
def build_dataset_from_klines(df, seq_len, horizon_steps):
    """
    df must be chronological ascending.
    horizon_steps = number of candles ahead to use as label
    returns X (N,K), Y (N,), sample_times
    """
    # Pre-calculate indicators on the whole series
    df = add_technical_indicators(df)
    
    closes = df["close"].values
    vols = df["volume"].values
    
    # Indicator arrays
    rsi = df['rsi'].values
    macd = df['macd'].values
    macd_sig = df['macd_signal'].values
    bb_up = df['bb_upper'].values
    bb_low = df['bb_lower'].values

    N = len(closes)
    max_i = N - horizon_steps
    X_list = []
    Y_list = []
    sample_times = []
    
    # We need enough history for indicators (max 26 for MACD), so start a bit later if needed
    # But df is already trimmed to required_candles which included buffer.
    # We start at seq_len to have a window.
    
    for i in range(seq_len, max_i):
        window = closes[i-seq_len:i]
        vwin = vols[i-seq_len:i]
        
        last = float(window[-1])
        mean = float(window.mean())
        std = float(window.std(ddof=0))
        
        # slope normalized
        x = np.arange(len(window))
        slope = float(np.polyfit(x, window, 1)[0]) / (mean if mean != 0 else 1.0)
        momentum = float((window[-1] - window[0]) / (window[0] if window[0] != 0 else 1.0))
        vmean = float(vwin.mean())
        ret = float((window[-1] - window[0]) / (window[0] if window[0] != 0 else 1.0))
        
        # Technical indicators at step i-1 (the last known candle in the window)
        idx = i - 1
        curr_rsi = float(rsi[idx])
        curr_macd = float(macd[idx])
        curr_macd_sig = float(macd_sig[idx])
        # Distance to BB bands normalized by price
        dist_bb_up = float((bb_up[idx] - last) / last)
        dist_bb_low = float((last - bb_low[idx]) / last)

        X_list.append([
            last, mean, std, slope, momentum, vmean, ret, 
            curr_rsi, curr_macd, curr_macd_sig, dist_bb_up, dist_bb_low
        ])
        Y_list.append(float(closes[i + horizon_steps - 1]))
        sample_times.append(df["close_time"].iloc[i])
        
    X = np.array(X_list, dtype=float)
    Y = np.array(Y_list, dtype=float)
    return X, Y, sample_times, df

# ---------------------------
# Scaling
# ---------------------------
class SimpleScaler:
    def __init__(self):
        self.mean = None
        self.scale = None
        
    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        # Avoid division by zero
        self.scale[self.scale == 0] = 1.0
        return (X - self.mean) / self.scale
        
    def transform(self, X):
        if self.mean is None: return X
        return (X - self.mean) / self.scale

# ---------------------------
# MALP optimizer (maximize CCC)
# ---------------------------
def ccc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2 or y_pred.size < 2:
        return 0.0
    mt, mp = y_true.mean(), y_pred.mean()
    vt, vp = y_true.var(), y_pred.var()
    cov = np.cov(y_true, y_pred)[0, 1]
    denom = vt + vp + (mt - mp) ** 2
    return (2 * cov) / denom if denom != 0 else 0.0

def fit_malp(X, Y, maxiter=2000):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    N, K = X.shape
    def neg_ccc(p):
        alpha = p[0]; beta = p[1:]
        pred = alpha + X.dot(beta)
        return -ccc(Y, pred)
    init = np.zeros(K + 1, dtype=float)
    init[0] = float(np.mean(Y)) if Y.size else 0.0
    
    # Initial guess for beta could be small random to break symmetry if needed, but zeros usually fine
    res = minimize(neg_ccc, init, method="Nelder-Mead", options={"maxiter": maxiter})
    a = float(res.x[0]); b = res.x[1:].astype(float)
    pred = a + X.dot(b)
    score = ccc(Y, pred)
    return a, b, score, pred

# ---------------------------
# Core Prediction Logic
# ---------------------------
def get_prediction(symbol_input, target_input):
    """
    Core function to run the prediction pipeline.
    Returns a dictionary with results or raises Exception.
    """
    symbol = normalize_symbol(symbol_input)
    try:
        horizon_seconds, target_dt = parse_target(target_input)
    except Exception as e:
        raise ValueError(f"Error parsing target: {e}")
        
    if horizon_seconds <= 0:
        raise ValueError("Target must be in the future.")

    # choose interval automatically
    interval = choose_interval(horizon_seconds)
    interval_seconds = INTERVALS[interval]
    # compute horizon in candle steps (round up)
    horizon_steps = math.ceil(horizon_seconds / interval_seconds)
    # compute required candles to fetch
    required_candles = SAMPLES + horizon_steps + SEQ_LEN + 50
    
    # Fetch data
    try:
        df = fetch_klines(symbol, interval, required_candles)
    except Exception as e:
        raise RuntimeError(f"Error fetching klines from Binance: {e}")

    # ensure chronological ascending
    df = df.sort_values("close_time").reset_index(drop=True)

    # build dataset
    X, Y, sample_times, df_with_ind = build_dataset_from_klines(df, SEQ_LEN, horizon_steps)
    if X.shape[0] < 50:
        raise ValueError("Not enough labelled samples to train. Try increasing history or reducing horizon.")

    # Split Train/Test (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    
    # Scale features
    scaler = SimpleScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    a, b, train_score, _ = fit_malp(X_train_scaled, Y_train, maxiter=3000)
    
    # Evaluate
    test_preds = a + X_test_scaled.dot(b)
    test_score = ccc(Y_test, test_preds)
    
    # Prepare inference
    last_df_slice = df_with_ind.iloc[-SEQ_LEN:]
    last_window = last_df_slice["close"].values
    last_vol_window = last_df_slice["volume"].values
    
    last_last = float(last_window[-1])
    last_mean = float(last_window.mean())
    last_std = float(last_window.std(ddof=0))
    x = np.arange(len(last_window))
    last_slope = float(np.polyfit(x, last_window, 1)[0]) / (last_mean if last_mean != 0 else 1.0)
    last_momentum = float((last_window[-1] - last_window[0]) / (last_window[0] if last_window[0] != 0 else 1.0))
    last_vmean = float(last_vol_window.mean())
    last_ret = float((last_window[-1] - last_window[0]) / (last_window[0] if last_window[0] != 0 else 1.0))
    
    # Indicators for the last candle
    last_idx = df_with_ind.index[-1]
    last_rsi = float(df_with_ind.at[last_idx, 'rsi'])
    last_macd = float(df_with_ind.at[last_idx, 'macd'])
    last_macd_sig = float(df_with_ind.at[last_idx, 'macd_signal'])
    last_bb_up = float(df_with_ind.at[last_idx, 'bb_upper'])
    last_bb_low = float(df_with_ind.at[last_idx, 'bb_lower'])
    
    dist_bb_up = float((last_bb_up - last_last) / last_last)
    dist_bb_low = float((last_last - last_bb_low) / last_last)
    
    feat = np.array([
        last_last, last_mean, last_std, last_slope, last_momentum, last_vmean, last_ret,
        last_rsi, last_macd, last_macd_sig, dist_bb_up, dist_bb_low
    ], dtype=float)
    
    # Scale inference features
    feat_scaled = scaler.transform(feat.reshape(1, -1)).flatten()

    forecast_val = float(a + np.dot(b, feat_scaled))
    last_price = float(last_last)
    pct = (forecast_val / last_price - 1.0) * 100.0
    
    feature_names = ["last","mean","std","slope","momentum","vol_mean","ret", "rsi", "macd", "macd_sig", "dist_bb_up", "dist_bb_low"]
    beta_dict = {name: float(coef) for name, coef in zip(feature_names, b)}

    return {
        "symbol": symbol,
        "target_dt": target_dt,
        "current_time": df['close_time'].iloc[-1],
        "last_price": last_price,
        "forecast_price": forecast_val,
        "pct_change": pct,
        "train_ccc": train_score,
        "test_ccc": test_score,
        "betas": beta_dict,
        "alpha": a,
        "interval": interval
    }

# ---------------------------
# Main interactive flow
# ---------------------------
def main():
    print("\n=== MALP v11 — Binance-only Enhanced Predictor ===\n")
    sym_in = input("Enter cryptocurrency (e.g. BTC or BTCUSDT or ETH): ").strip()
    target_text = input("Enter prediction target (e.g. 5m, 2h, 3d, 1M or absolute UTC 'YYYY-MM-DD HH:MM'): ").strip()
    
    try:
        res = get_prediction(sym_in, target_text)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"\nSymbol: {res['symbol']}")
    print(f"Target datetime (UTC): {res['target_dt'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Fetching candles at interval {res['interval']}...\n")
    
    # Note: We don't save CSV in the library function to keep it clean, 
    # but for CLI we might want to. For now, we skip saving CSV in refactor to keep it simple.
    
    print(f"Training complete.")
    print(f"Train CCC: {res['train_ccc']:.4f}")
    print(f"Test CCC:  {res['test_ccc']:.4f}")
    
    if res['test_ccc'] < 0.8:
        print("Warning: Test score is low. Prediction might be unreliable.")

    print("\n=== FORECAST ===")
    print(f"As of (latest candle end UTC): {res['current_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Predicting for {res['target_dt'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Last {res['interval']} close: {res['last_price']:,.6f}")
    print(f"MALP forecast: {res['forecast_price']:,.6f}   ({res['pct_change']:+.3f}%)")
    print("\nBeta coefficients (scaled importance):")
    for name, coef in res['betas'].items():
        print(f"  {name}: {coef:.6f}")
    print(f"Alpha (intercept): {res['alpha']:.6f}")
    print("\nDone.\n")

if __name__ == "__main__":
    main()
