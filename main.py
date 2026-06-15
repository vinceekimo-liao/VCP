# ============================================================
# 強制安裝依賴（解決 Render 虛擬環境路徑問題）
# ============================================================
import subprocess
import sys
import os

def ensure_package(package_name, pip_name=None):
    """確保套件已安裝，若無則自動安裝"""
    if pip_name is None:
        pip_name = package_name
    try:
        __import__(package_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

# 依序檢查並安裝所有依賴
ensure_package("finmind", "FinMind==1.6.6")
ensure_package("pandas", "pandas>=2.0.0")
ensure_package("numpy", "numpy>=1.24.0")
ensure_package("fastapi", "fastapi==0.104.1")
ensure_package("uvicorn", "uvicorn==0.24.0")
# ============================================================

import finmind
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = {"data": None, "timestamp": None}
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
        if recent_vol > vol_ma_20.iloc[-1] * 0.8: return False
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
        if rs < 60: return False
        return {
            "price": round(float(close.iloc[-1]), 2),
            "change_pct": round(float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100), 2),
            "rs_score": rs, "contractions": contractions,
            "volume_ratio": round(float(recent_vol / vol_ma_20.iloc[-1]), 2),
            "ma50": round(float(close.rolling(50).mean().iloc[-1]), 2) if len(close) >= 50 else None,
            "ma150": round(float(close.rolling(150).mean().iloc[-1]), 2) if len(close) >= 150 else None,
            "ma200": round(float(close.rolling(200).mean().iloc[-1]), 2) if len(close) >= 200 else None,
            "high_52w": round(float(high.rolling(250).max().iloc[-1]), 2) if len(high) >= 250 else round(float(high.max()), 2),
            "low_52w": round(float(low.rolling(250).min().iloc[-1]), 2) if len(low) >= 250 else round(float(low.min()), 2),
        }
    except: return False

def full_scan():
    print("開始全市場掃描...")
    start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    stocks = get_all_stocks()
    total = len(stocks)
    print(f"總股票數: {total}")
    layer1_results = []
    batch_size = 100
    for i in range(0, total, batch_size):
        batch = stocks[i:i+batch_size]
        print(f"批次 {i//batch_size + 1}/{(total-1)//batch_size + 1}")
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_daily, sid, start_date): sid for sid in batch}
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    df = future.result(timeout=15)
                    if minervini_check(df): layer1_results.append((sid, df))
                except: pass
        time.sleep(0.5)
    layer1_count = len(layer1_results)
    print(f"第一層通過: {layer1_count}")
    layer2_results = []
    for sid, df in layer1_results:
        result = vcp_math_check(df)
        if result:
            result["symbol"] = sid
            if result["contractions"] >= 3 and result["volume_ratio"] >= 1.5: result["quality"] = "A"
            elif result["contractions"] >= 2: result["quality"] = "B"
            else: result["quality"] = "C"
            layer2_results.append(result)
    layer2_count = len(layer2_results)
    print(f"第二層通過: {layer2_count}")
    layer2_results.sort(key=lambda x: (x["quality"] != "A", x["quality"] != "B", -x["rs_score"]))
    return {"total": total, "layer1": layer1_count, "layer2": layer2_count, "candidates": layer2_results[:10], "timestamp": datetime.now().isoformat()}

@app.get("/scan")
def scan(force: bool = Query(False)):
    global cache
    if not force and cache["data"] and cache["timestamp"]:
        if (datetime.now() - cache["timestamp"]).seconds < 1800: return cache["data"]
    try:
        result = full_scan()
        cache["data"] = result; cache["timestamp"] = datetime.now()
        return result
    except Exception as e:
        return {"error": str(e), "total": 0, "layer1": 0, "layer2": 0, "candidates": []}

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
