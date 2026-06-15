import os
import sys
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# 強制設定 site-packages 路徑
sys.path.insert(0, "/opt/render/project/src/.venv/lib/python3.11/site-packages")

import finmind
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")

def get_all_stocks():
    api = finmind.FinMind(token=FINMIND_TOKEN)
    info = api.get("TaiwanStockInfo")
    info = info[(info["type"] == "Common Stock") & (info["stock_id"].str.len() == 4)]
    return info["stock_id"].tolist()

def fetch_daily(sid, start_date):
    api = finmind.FinMind(token=FINMIND_TOKEN)
    try:
        data = api.get("TaiwanStockPrice", data_id=sid, start_date=start_date, end_date=datetime.today().strftime("%Y-%m-%d"))
        if data.empty: return None
        data["date"] = pd.to_datetime(data["date"])
        data.sort_values("date", inplace=True)
        data.set_index("date", inplace=True)
        return data
    except: return None

def minervini_check(data):
    if data is None or len(data) < 200: return False
    close = data["close"]
    try:
        ma50 = close.rolling(50).mean()
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()
        last = close.iloc[-1]
        if not (last > ma50.iloc[-1] > ma150.iloc[-1] > ma200.iloc[-1]): return False
        if len(ma200) >= 25 and ma200.iloc[-1] <= ma200.iloc[-25]: return False
        if "max" in data.columns and len(data) >= 250:
            if last < data["max"].rolling(250).max().iloc[-1] * 0.75: return False
        return True
    except: return False

def vcp_math_check(data):
    if data is None or len(data) < 60: return False
    try:
        close = data["close"]
        volume = data["volume"]
        high = data["max"]
        low = data["min"]
        vol_ma_20 = volume.rolling(20).mean()
        recent_vol = volume.iloc[-3:].mean()
        if pd.isna(vol_ma_20.iloc[-1]): return False
        contractions = 0
        trough_prices = []
        in_pullback = False
        for i in range(20, len(close) - 5):
            try:
                pc = (close.iloc[i] - close.iloc[i-5]) / close.iloc[i-5] * 100
                vc = (volume.iloc[i] - volume.iloc[i-5]) / volume.iloc[i-5] * 100 if volume.iloc[i-5] != 0 else 0
            except: continue
            if not in_pullback and pc < -2 and vc < -15: in_pullback = True
            if in_pullback and pc > 0:
                if len(trough_prices) == 0 or close.iloc[i] > trough_prices[-1]:
                    trough_prices.append(close.iloc[i]); contractions += 1
                in_pullback = False
        if contractions < 2: return False
        rs_lookback = min(60, len(close))
        rs = min(99, max(1, int(50 + (close.iloc[-1] - close.iloc[-rs_lookback]) / close.iloc[-rs_lookback] * 200)))
        return {
            "symbol": "",
            "price": round(float(close.iloc[-1]), 2),
            "change_pct": round(float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100), 2),
            "rs_score": rs, "contractions": contractions,
            "volume_ratio": round(float(recent_vol / vol_ma_20.iloc[-1]), 2),
            "quality": "A" if contractions >= 3 else "B" if contractions >= 2 else "C",
            "ma50": round(float(close.rolling(50).mean().iloc[-1]), 2) if len(close) >= 50 else None,
            "ma150": round(float(close.rolling(150).mean().iloc[-1]), 2) if len(close) >= 150 else None,
            "ma200": round(float(close.rolling(200).mean().iloc[-1]), 2) if len(close) >= 200 else None,
            "high_52w": round(float(high.rolling(250).max().iloc[-1]), 2) if len(high) >= 250 else round(float(high.max()), 2),
            "low_52w": round(float(low.rolling(250).min().iloc[-1]), 2) if len(low) >= 250 else round(float(low.min()), 2),
        }
    except: return False

def full_scan():
    start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    stocks = get_all_stocks()
    total = len(stocks)
    layer1_results = []
    for i in range(0, total, 100):
        batch = stocks[i:i+100]
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_daily, sid, start_date): sid for sid in batch}
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    df = future.result(timeout=15)
                    if minervini_check(df): layer1_results.append((sid, df))
                except: pass
        time.sleep(0.3)
    layer2_results = []
    for sid, df in layer1_results:
        result = vcp_math_check(df)
        if result:
            result["symbol"] = sid
            layer2_results.append(result)
    layer2_results.sort(key=lambda x: (-x["rs_score"]))
    return {"total": total, "layer1": len(layer1_results), "layer2": len(layer2_results), "candidates": layer2_results[:10]}

@app.get("/scan")
def scan():
    try:
        return full_scan()
    except Exception as e:
        return {"error": str(e), "total": 0, "layer1": 0, "layer2": 0, "candidates": []}

@app.get("/health")
def health():
    return {"status": "ok"}
