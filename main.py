import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import finmind
except ImportError:
    install_package("FinMind==1.6.6")
    import finmind
    
import finmind
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

app = FastAPI()

# 允許前端跨域請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════
# 全域快取（避免重複計算）
# ══════════════════════════════════════════════════
cache = {"data": None, "timestamp": None}

# ══════════════════════════════════════════════════
# FinMind 初始化（從環境變數讀取 Token）
# ══════════════════════════════════════════════════
import os
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "你的預設Token")

# ══════════════════════════════════════════════════
# 資料獲取
# ══════════════════════════════════════════════════
def get_all_stocks():
    """取得上市櫃所有普通股代號"""
    api = finmind.FinMind(token=FINMIND_TOKEN)
    info = api.get("TaiwanStockInfo")
    info = info[(info["type"] == "Common Stock") & 
                (info["stock_id"].str.len() == 4)]
    return info["stock_id"].tolist()

def fetch_daily(sid, start_date):
    """拉取單檔股票日線"""
    api = finmind.FinMind(token=FINMIND_TOKEN)
    try:
        data = api.get("TaiwanStockPrice", 
                       data_id=sid,
                       start_date=start_date,
                       end_date=datetime.today().strftime("%Y-%m-%d"))
        if data.empty:
            return None
        data["date"] = pd.to_datetime(data["date"])
        data.sort_values("date", inplace=True)
        data.set_index("date", inplace=True)
        return data
    except Exception as e:
        print(f"  fetch_daily error for {sid}: {e}")
        return None

# ══════════════════════════════════════════════════
# 第一層：Minervini 趨勢模板
# ══════════════════════════════════════════════════
def minervini_check(data):
    """檢查是否符合 Minervini 趨勢模板"""
    if data is None or len(data) < 200:
        return False
    
    close = data["close"]
    
    try:
        ma50 = close.rolling(50).mean()
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()
        
        last = close.iloc[-1]
        last_ma50 = ma50.iloc[-1]
        last_ma150 = ma150.iloc[-1]
        last_ma200 = ma200.iloc[-1]
        
        # 條件 1：收盤 > MA50 > MA150 > MA200
        if not (last > last_ma50 > last_ma150 > last_ma200):
            return False
        
        # 條件 2：MA200 近 25 日向上
        if len(ma200) >= 25:
            ma200_slice = ma200.iloc[-25:]
            if ma200_slice.iloc[-1] <= ma200_slice.iloc[0]:
                return False
        
        # 條件 3：股價距 52 週高點 ≤ 25%
        if "max" in data.columns and len(data) >= 250:
            high_52w = data["max"].rolling(250).max().iloc[-1]
            if last < high_52w * 0.75:
                return False
        
        return True
    except Exception as e:
        print(f"  minervini_check error: {e}")
        return False

# ══════════════════════════════════════════════════
# 第二層：VCP 數學波動收縮
# ══════════════════════════════════════════════════
def vcp_math_check(data):
    """檢查是否符合 VCP 數學特徵"""
    if data is None or len(data) < 60:
        return False
    
    try:
        close = data["close"]
        volume = data["volume"]
        high = data["max"]
        low = data["min"]
        
        # 窒息量：近 3 日均量 < 20 日均量 × 0.5
        vol_ma_20 = volume.rolling(20).mean()
        recent_vol = volume.iloc[-3:].mean()
        
        if pd.isna(vol_ma_20.iloc[-1]):
            return False
        
        # 放寬窒息量條件到 0.8（避免過嚴）
        if recent_vol > vol_ma_20.iloc[-1] * 0.8:
            return False
        
        # 計算收縮次數
        contractions = 0
        trough_prices = []
        in_pullback = False
        
        for i in range(20, len(close) - 5):
            try:
                pct_change = (close.iloc[i] - close.iloc[i-5]) / close.iloc[i-5] * 100
                vol_change = (volume.iloc[i] - volume.iloc[i-5]) / volume.iloc[i-5] * 100 if volume.iloc[i-5] != 0 else 0
            except:
                continue
            
            if not in_pullback and pct_change < -2 and vol_change < -15:
                in_pullback = True
            if in_pullback and pct_change > 0:
                if len(trough_prices) == 0 or close.iloc[i] > trough_prices[-1]:
                    trough_prices.append(close.iloc[i])
                    contractions += 1
                in_pullback = False
        
        if contractions < 2:
            return False
        
        # RS 強度檢查（放寬到 60）
        rs_lookback = min(60, len(close))
        rs = min(99, max(1, int(
            50 + (close.iloc[-1] - close.iloc[-rs_lookback]) / close.iloc[-rs_lookback] * 200
        )))
        if rs < 60:
            return False
        
        # 回傳詳細數據
        return {
            "price": round(float(close.iloc[-1]), 2),
            "change_pct": round(float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100), 2),
            "rs_score": rs,
            "contractions": contractions,
            "volume_ratio": round(float(recent_vol / vol_ma_20.iloc[-1]), 2),
            "ma50": round(float(close.rolling(50).mean().iloc[-1]), 2) if len(close) >= 50 else None,
            "ma150": round(float(close.rolling(150).mean().iloc[-1]), 2) if len(close) >= 150 else None,
            "ma200": round(float(close.rolling(200).mean().iloc[-1]), 2) if len(close) >= 200 else None,
            "high_52w": round(float(high.rolling(250).max().iloc[-1]), 2) if len(high) >= 250 else round(float(high.max()), 2),
            "low_52w": round(float(low.rolling(250).min().iloc[-1]), 2) if len(low) >= 250 else round(float(low.min()), 2),
        }
    except Exception as e:
        print(f"  vcp_math_check error: {e}")
        return False

# ══════════════════════════════════════════════════
# 主掃描函數（修正語法錯誤）
# ══════════════════════════════════════════════════
def full_scan():
    """執行完整漏斗篩選"""
    print("開始全市場掃描...")
    start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    
    # 取得所有股票代號
    stocks = get_all_stocks()
    total = len(stocks)
    print(f"總股票數: {total}")
    
    # 第一層：趨勢模板（平行處理）
    layer1_results = []
    batch_size = 100
    
    for i in range(0, total, batch_size):
        batch = stocks[i:i+batch_size]
        print(f"處理批次 {i//batch_size + 1}/{(total//batch_size)+1} ({len(batch)} 檔)")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_daily, sid, start_date): sid for sid in batch}
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    df = future.result(timeout=15)
                    if minervini_check(df):
                        layer1_results.append((sid, df))
                except Exception as e:
                    print(f"  {sid} 失敗: {e}")
        
        time.sleep(0.5)  # 避免 FinMind API 限流
    
    layer1_count = len(layer1_results)
    print(f"第一層通過: {layer1_count}")
    
    # 第二層：VCP 數學過濾
    layer2_results = []
    for sid, df in layer1_results:
        result = vcp_math_check(df)
        if result:
            result["symbol"] = sid
            # 品質評級
            if result["contractions"] >= 3 and result["volume_ratio"] >= 1.5:
                result["quality"] = "A"
            elif result["contractions"] >= 2:
                result["quality"] = "B"
            else:
                result["quality"] = "C"
            layer2_results.append(result)
    
    layer2_count = len(layer2_results)
    print(f"第二層通過: {layer2_count}")
    
    # 依品質排序，A級優先
    layer2_results.sort(key=lambda x: (x["quality"] != "A", x["quality"] != "B", -x["rs_score"]))
    
    return {
        "total": total,
        "layer1": layer1_count,
        "layer2": layer2_count,
        "candidates": layer2_results[:10],  # 最多回傳 10 檔
        "timestamp": datetime.now().isoformat()
    }

# ══════════════════════════════════════════════════
# API 端點
# ══════════════════════════════════════════════════
@app.get("/scan")
def scan(force: bool = Query(False)):
    """執行漏斗篩選"""
    global cache
    
    # 如果快取在 30 分鐘內且不強制更新，直接回傳
    if not force and cache["data"] and cache["timestamp"]:
        elapsed = (datetime.now() - cache["timestamp"]).seconds
        if elapsed < 1800:  # 30 分鐘
            return cache["data"]
    
    try:
        result = full_scan()
        cache["data"] = result
        cache["timestamp"] = datetime.now()
        return result
    except Exception as e:
        return {
            "error": str(e),
            "total": 0,
            "layer1": 0,
            "layer2": 0,
            "candidates": []
        }

@app.get("/health")
def health():
    """健康檢查"""
    return {"status": "ok", "time": datetime.now().isoformat()}

# ══════════════════════════════════════════════════
# 啟動設定
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
