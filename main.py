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
        return data
    except Exception as e:
        return None

# ========== 第一層：Minervini 趨勢模板（放寬） ==========
def minervini_check(data):
    if data is None or len(data) < 200:
        return False

    # 使用大寫欄位（DataLoader 預設格式）
    try:
        close = data["Close"]
        high = data["High"]
    except KeyError:
        # 若為小寫，改用小寫
        try:
            close = data["close"]
            high = data["high"]
        except KeyError:
            return False

    try:
        ma50 = close.rolling(50).mean()
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()

        last = close.iloc[-1]

        # 放寬：只需收盤 > MA150 和 MA200（不強制 MA50 > MA150）
        if not (last > ma150.iloc[-1] and last > ma200.iloc[-1]):
            return False

        # MA200 近 25 日向上
        if len(ma200) >= 25 and ma200.iloc[-1] <= ma200.iloc[-25]:
            return False

        # 距 52 週高點 ≤ 25%
        if len(high) >= 250:
            high_52w = high.rolling(250).max().iloc[-1]
            if last < high_52w * 0.75:
                return False

        return True
    except:
        return False

# ========== 第二層：VCP 數學波動收縮（終極放寬） ==========
def vcp_math_check(data):
    if data is None or len(data) < 60:
        return False

    # 使用大寫欄位
    try:
        close = data["Close"]
        volume = data["Volume"]
        high = data["High"]
        low = data["Low"]
    except KeyError:
        try:
            close = data["close"]
            volume = data["volume"]
            high = data["high"]
            low = data["low"]
        except KeyError:
            return False

    try:
        # 計算波動率（直接用 20 日標準差）
        rolling_std = close.rolling(20).std()
        latest_std = rolling_std.iloc[-1]
        std_min_60 = rolling_std.rolling(60).min().iloc[-1]

        # 計算收縮次數
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

        # 放寬：若波動率處於 60 天低點，即使收縮次數為 0 也給過
        is_low_vol = latest_std <= std_min_60 * 1.05
        if contractions == 0 and not is_low_vol:
            return False

        # RS 強度
        rs_lookback = min(60, len(close))
        rs = min(99, max(1, int(50 + (close.iloc[-1] - close.iloc[-rs_lookback]) / close.iloc[-rs_lookback] * 200)))

        # 品質評級
        quality_score = 0
        if contractions >= 3: quality_score += 2
        elif contractions >= 2: quality_score += 1
        if is_low_vol: quality_score += 2
        if rs >= 70: quality_score += 1
        if rs >= 85: quality_score += 1

        quality = "A" if quality_score >= 4 else "B" if quality_score >= 2 else "C"

        vol_ma_20 = volume.rolling(20).mean()
        recent_vol = volume.iloc[-3:].mean()

        return {
            "symbol": "",
            "price": round(float(close.iloc[-1]), 2),
            "change_pct": round(float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100), 2),
            "rs_score": rs,
            "contractions": contractions,
            "volume_ratio": round(float(recent_vol / vol_ma_20.iloc[-1]), 2) if not pd.isna(vol_ma_20.iloc[-1]) else 0,
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
    print(f"第一層通過：{layer1_count} 檔")

    layer2_results = []
    for sid, df in layer1_results:
        result = vcp_math_check(df)
        if result:
            result["symbol"] = sid
            layer2_results.append(result)

    layer2_count = len(layer2_results)
    print(f"第二層通過：{layer2_count} 檔")

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
