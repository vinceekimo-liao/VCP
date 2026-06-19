import os
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from FinMind.data import DataLoader

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")

# ========== 資料取得 ==========
def get_all_stocks():
    api = DataLoader()
    if FINMIND_TOKEN:
        api.login_by_token(FINMIND_TOKEN)
    info = api.taiwan_stock_info()
    if info is None or info.empty:
        return []
    # 實際欄位名稱：type, stock_id
    # 可能大小寫不同，進行簡單相容
    type_col = "type" if "type" in info.columns else [c for c in info.columns if c.lower() == "type"][0]
    id_col = "stock_id" if "stock_id" in info.columns else [c for c in info.columns if c.lower() == "stock_id"][0]
    info = info[(info[type_col] == "Common Stock") & (info[id_col].astype(str).str.len() == 4)]
    return info[id_col].tolist()

def fetch_daily(sid, start_date):
    api = DataLoader()
    if FINMIND_TOKEN:
        api.login_by_token(FINMIND_TOKEN)
    try:
        data = api.taiwan_stock_price(
            stock_id=sid,
            start_date=start_date,
            end_date=datetime.today().strftime("%Y-%m-%d")
        )
        if data.empty:
            return None
        data["date"] = pd.to_datetime(data["date"])
        data.sort_values("date", inplace=True)
        data.set_index("date", inplace=True)
        return data
    except:
        return None

# ========== 第一層：Minervini 趨勢模板（已修正欄位） ==========
def minervini_check(data):
    if data is None or len(data) < 200:
        return False

    # FinMind 實際欄位：close, max, min, Trading_Volume
    try:
        close = data["close"]
        high = data["max"]          # FinMind 用 "max"
    except KeyError:
        return False

    try:
        ma50 = close.rolling(50).mean()
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()

        last = close.iloc[-1]
        # 放寬：收盤 > MA150 且 > MA200
        if not (last > ma150.iloc[-1] and last > ma200.iloc[-1]):
            return False

        # MA200 近 25 日向上
        if len(ma200) >= 25 and ma200.iloc[-1] <= ma200.iloc[-25]:
            return False

        # 距 52 週高點 ≤ 25% (使用 min_periods=200 避免 NaN)
        if len(high) >= 250:
            high_52w = high.rolling(250, min_periods=200).max().iloc[-1]
            if pd.notna(high_52w) and last < high_52w * 0.75:
                return False

        return True
    except:
        return False

# ========== 第二層：VCP 數學波動收縮（已修正欄位） ==========
def vcp_math_check(data):
    if data is None or len(data) < 60:
        return False

    # FinMind 實際欄位：close, max, min, Trading_Volume
    try:
        close = data["close"]
        volume = data["Trading_Volume"]   # 關鍵：成交量
        high = data["max"]
        low = data["min"]
    except KeyError:
        return False

    try:
        vol_ma_20 = volume.rolling(20).mean()
        recent_vol = volume.iloc[-3:].mean()
        if pd.isna(vol_ma_20.iloc[-1]):
            return False

        # 波動率輔助：20 日標準差
        rolling_std = close.rolling(20).std()
        latest_std = rolling_std.iloc[-1]
        if pd.isna(latest_std):
            return False
        std_min_60 = rolling_std.rolling(60, min_periods=20).min().iloc[-1]

        # 收縮次數
        contractions = 0
        trough_prices = []
        in_pullback = False
        for i in range(20, len(close) - 5):
            try:
                pc = (close.iloc[i] - close.iloc[i-5]) / close.iloc[i-5] * 100
                vc = (volume.iloc[i] - volume.iloc[i-5]) / volume.iloc[i-5] * 100 if volume.iloc[i-5] != 0 else 0
            except:
                continue
            if not in_pullback and pc < -2 and vc < -15:
                in_pullback = True
            if in_pullback and pc > 0:
                if len(trough_prices) == 0 or close.iloc[i] > trough_prices[-1]:
                    trough_prices.append(close.iloc[i])
                    contractions += 1
                in_pullback = False

        # 放寬：若波動率處在 60 天低點，即使收縮次數為 0 也給過
        is_low_vol = (latest_std <= std_min_60 * 1.05) if pd.notna(std_min_60) else False
        if contractions == 0 and not is_low_vol:
            return False

        # RS 強度
        rs_lookback = min(60, len(close))
        rs = min(99, max(1, int(50 + (close.iloc[-1] - close.iloc[-rs_lookback]) / close.iloc[-rs_lookback] * 200)))

        # 品質
        quality_score = 0
        if contractions >= 3: quality_score += 2
        elif contractions >= 2: quality_score += 1
        if is_low_vol: quality_score += 2
        if rs >= 70: quality_score += 1
        if rs >= 85: quality_score += 1
        quality = "A" if quality_score >= 4 else "B" if quality_score >= 2 else "C"

        return {
            "symbol": "",
            "price": round(float(close.iloc[-1]), 2),
            "change_pct": round(float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100), 2),
            "rs_score": rs,
            "contractions": contractions,
            "volume_ratio": round(float(recent_vol / vol_ma_20.iloc[-1]), 2),
            "quality": quality,
            "ma50": round(float(close.rolling(50).mean().iloc[-1]), 2) if len(close) >= 50 else None,
            "ma150": round(float(close.rolling(150).mean().iloc[-1]), 2) if len(close) >= 150 else None,
            "ma200": round(float(close.rolling(200).mean().iloc[-1]), 2) if len(close) >= 200 else None,
            "high_52w": round(float(high.rolling(250, min_periods=200).max().iloc[-1]), 2) if len(high) >= 250 else round(float(high.max()), 2),
            "low_52w": round(float(low.rolling(250, min_periods=200).min().iloc[-1]), 2) if len(low) >= 250 else round(float(low.min()), 2),
        }
    except Exception as e:
        print(f"  vcp_math_check error: {e}")
        return False

# ========== 主掃描 ==========
def full_scan():
    start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    stocks = get_all_stocks()
    total = len(stocks)
    print(f"股票清單數量: {total}")
    if total == 0:
        return {"total": 0, "layer1": 0, "layer2": 0, "candidates": [], "error": "No stocks (check token or network)"}

    layer1_results = []
    for i in range(0, total, 100):
        batch = stocks[i:i+100]
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_daily, sid, start_date): sid for sid in batch}
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    df = future.result(timeout=15)
                    if minervini_check(df):
                        layer1_results.append((sid, df))
                except:
                    pass
        time.sleep(0.3)

    layer1_count = len(layer1_results)
    print(f"第一層通過: {layer1_count}")

    layer2_results = []
    for sid, df in layer1_results:
        result = vcp_math_check(df)
        if result:
            result["symbol"] = sid
            layer2_results.append(result)

    layer2_count = len(layer2_results)
    print(f"第二層通過: {layer2_count}")
    layer2_results.sort(key=lambda x: (-x["rs_score"]))
    return {"total": total, "layer1": layer1_count, "layer2": layer2_count, "candidates": layer2_results[:10]}

@app.get("/scan")
def scan():
    try:
        return full_scan()
    except Exception as e:
        return {"error": str(e), "total": 0, "layer1": 0, "layer2": 0, "candidates": []}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/test-token")
def test_token():
    return {
        "token_exists": bool(FINMIND_TOKEN),
        "token_preview": FINMIND_TOKEN[:8] + "..." if len(FINMIND_TOKEN) > 8 else "(空)"
    }
