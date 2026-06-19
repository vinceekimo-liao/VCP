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

# ========== 輔助：欄位名稱對照 ==========
def normalize_columns(df):
    """將可能的欄位名稱（大小寫、簡稱）統一為小寫標準名稱"""
    mapping = {
        'close': 'close', 'Close': 'close', 'CLOSE': 'close',
        'volume': 'volume', 'Volume': 'volume', 'VOLUME': 'volume',
        'max': 'high', 'high': 'high', 'High': 'high', 'HIGH': 'high',
        'min': 'low', 'low': 'low', 'Low': 'low', 'LOW': 'low',
        'open': 'open', 'Open': 'open', 'OPEN': 'open',
        'date': 'date', 'Date': 'date', 'DATE': 'date',
    }
    rename_dict = {}
    for col in df.columns:
        if col in mapping:
            rename_dict[col] = mapping[col]
    if rename_dict:
        df = df.rename(columns=rename_dict)
    # 若仍然缺少必要欄位，嘗試從原始 dataframe 建立
    if 'high' not in df.columns and 'max' in df.columns:
        df['high'] = df['max']
    if 'low' not in df.columns and 'min' in df.columns:
        df['low'] = df['min']
    return df

# ========== 資料取得 ==========
def get_all_stocks():
    api = DataLoader()
    if FINMIND_TOKEN:
        api.login_by_token(FINMIND_TOKEN)
    info = api.taiwan_stock_info()
    info = info[(info["type"] == "Common Stock") & (info["stock_id"].str.len() == 4)]
    return info["stock_id"].tolist()

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
        # 標準化欄位名稱
        data = normalize_columns(data)
        return data
    except Exception as e:
        print(f"  fetch_daily error for {sid}: {e}")
        return None

# ========== 第一層：Minervini 趨勢模板 ==========
def minervini_check(data):
    if data is None or len(data) < 200:
        return False
    try:
        close = data["close"]
        high = data["high"] if "high" in data.columns else None
    except KeyError:
        return False

    try:
        ma50 = close.rolling(50).mean()
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()

        last = close.iloc[-1]
        if not (last > ma50.iloc[-1] > ma150.iloc[-1] > ma200.iloc[-1]):
            return False

        # MA200 近 25 日向上
        if len(ma200) >= 25 and ma200.iloc[-1] <= ma200.iloc[-25]:
            return False

        # 距 52 週高點 ≤ 25%
        if high is not None and len(high) >= 250:
            high_52w = high.rolling(250).max().iloc[-1]
            if last < high_52w * 0.75:
                return False

        return True
    except:
        return False

# ========== 第二層：VCP 數學波動收縮（放寬版） ==========
def vcp_math_check(data):
    if data is None or len(data) < 60:
        return False
    try:
        close = data["close"]
        volume = data["volume"]
        high = data["high"]
        low = data["low"]
    except KeyError:
        return False

    try:
        vol_ma_20 = volume.rolling(20).mean()
        recent_vol = volume.iloc[-3:].mean()
        if pd.isna(vol_ma_20.iloc[-1]):
            return False

        # ===== 暫時移除窒息量條件 =====
        # if recent_vol > vol_ma_20.iloc[-1] * 0.8:
        #     return False
        # 記錄但不過濾
        dry_up = recent_vol < vol_ma_20.iloc[-1] * 0.8
        # =============================

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

        # ===== 放寬收縮次數門檻為 ≥1 =====
        if contractions < 1:
            return False
        # ================================

        # RS 強度
        rs_lookback = min(60, len(close))
        rs = min(99, max(1, int(50 + (close.iloc[-1] - close.iloc[-rs_lookback]) / close.iloc[-rs_lookback] * 200)))

        # 品質評級（加入窒息量作為加分項，但不強制）
        quality_score = 0
        if contractions >= 3:
            quality_score += 2
        elif contractions >= 2:
            quality_score += 1
        if dry_up:
            quality_score += 2
        if rs >= 70:
            quality_score += 1
        if rs >= 85:
            quality_score += 1

        if quality_score >= 4:
            quality = "A"
        elif quality_score >= 2:
            quality = "B"
        else:
            quality = "C"

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
            "high_52w": round(float(high.rolling(250).max().iloc[-1]), 2) if len(high) >= 250 else round(float(high.max()), 2),
            "low_52w": round(float(low.rolling(250).min().iloc[-1]), 2) if len(low) >= 250 else round(float(low.min()), 2),
        }
    except Exception as e:
        print(f"  vcp_math_check error: {e}")
        return False

# ========== 主掃描函數 ==========
def full_scan():
    start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    stocks = get_all_stocks()
    total = len(stocks)
    print(f"開始掃描，總股票數：{total}")

    # 第一層
    layer1_results = []
    batch_size = 100
    for i in range(0, total, batch_size):
        batch = stocks[i:i+batch_size]
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
    print(f"第一層通過：{layer1_count} 檔")

    # 第二層
    layer2_results = []
    for sid, df in layer1_results:
        result = vcp_math_check(df)
        if result:
            result["symbol"] = sid
            layer2_results.append(result)

    layer2_count = len(layer2_results)
    print(f"第二層通過：{layer2_count} 檔")

    # 依 RS 排序
    layer2_results.sort(key=lambda x: (-x["rs_score"]))
    return {"total": total, "layer1": layer1_count, "layer2": layer2_count, "candidates": layer2_results[:10]}

# ========== API 端點 ==========
@app.get("/scan")
def scan():
    try:
        return full_scan()
    except Exception as e:
        return {"error": str(e), "total": 0, "layer1": 0, "layer2": 0, "candidates": []}

@app.get("/health")
def health():
    return {"status": "ok"}
