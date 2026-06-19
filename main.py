import os
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from fastapi import FastAPI
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

# ========== 一次性下載全市場歷史資料 ==========
def get_all_market_data(days_lookback=400):
    """
    一次 API 請求下載所有股票的每日數據。
    返回一個未過濾的 DataFrame（包含所有股票、所有日期）。
    """
    api = DataLoader()
    if FINMIND_TOKEN:
        api.login_by_token(FINMIND_TOKEN)

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=days_lookback)).strftime("%Y-%m-%d")

    print(f"⏳ 一次性下載全市場資料：{start_date} ～ {end_date}")
    df = api.taiwan_stock_daily(start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        print("❌ 下載的資料為空")
        return pd.DataFrame()
    print(f"✅ 下載完成，總資料筆數：{len(df)}")
    return df

# ========== 取得股票清單 ==========
def get_filtered_stock_ids():
    api = DataLoader()
    if FINMIND_TOKEN:
        api.login_by_token(FINMIND_TOKEN)
    info = api.taiwan_stock_info()
    if info is None or info.empty:
        return []

    info = info[~info["stock_name"].str.contains("權|ETF|存託憑證", na=False)]
    info = info[info["stock_id"].str.len() == 4]
    stock_ids = info["stock_id"].unique().tolist()
    print(f"📋 普通股代號數量：{len(stock_ids)}")
    return stock_ids

# ========== 批次篩選：向量化處理 ==========
def batch_process_filter(df, valid_stock_ids):
    if df.empty:
        return pd.DataFrame()

    df = df[df["stock_id"].isin(valid_stock_ids)].copy()

    required_cols = {"stock_id", "date", "close", "high", "low", "volume"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"❌ 缺少必要欄位：{missing}")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    for col in ["close", "high", "low", "volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values(["stock_id", "date"])

    print("🧮 向量化計算均線與 52 週高點...")
    grouped = df.groupby("stock_id")

    df["ma50"] = grouped["close"].transform(lambda x: x.rolling(50, min_periods=1).mean())
    df["ma150"] = grouped["close"].transform(lambda x: x.rolling(150, min_periods=1).mean())
    df["ma200"] = grouped["close"].transform(lambda x: x.rolling(200, min_periods=1).mean())
    df["high_52w"] = grouped["high"].transform(lambda x: x.rolling(250, min_periods=1).max())

    # 🔧 微調：reset_index 確保索引乾淨
    latest = df.groupby("stock_id").tail(1).copy().reset_index(drop=True)

    cond = (
        (latest["close"] > latest["ma150"]) &
        (latest["close"] > latest["ma200"]) &
        (latest["ma200"] >= latest.groupby("stock_id")["ma200"].shift(25) * 0.98) &
        (latest["close"] >= latest["high_52w"] * 0.75)
    )

    candidates = latest[cond].copy()
    print(f"🎯 第一層篩選後剩餘：{len(candidates)} 檔")
    return candidates

# ========== VCP 第二層篩選 ==========
def vcp_math_check(stock_df):
    if stock_df is None or len(stock_df) < 60:
        return None

    close  = pd.to_numeric(stock_df["close"], errors='coerce').dropna()
    high   = pd.to_numeric(stock_df["high"],  errors='coerce').dropna()
    low    = pd.to_numeric(stock_df["low"],   errors='coerce').dropna()
    volume = pd.to_numeric(stock_df["volume"], errors='coerce').dropna()

    if len(close) < 60 or len(volume) < 60:
        return None

    try:
        vol_ma_20 = volume.rolling(20).mean()
        recent_vol = volume.iloc[-3:].mean()
        if pd.isna(vol_ma_20.iloc[-1]):
            return None

        rolling_std = close.rolling(20).std()
        latest_std = rolling_std.iloc[-1]
        if pd.isna(latest_std):
            return None
        std_min_60 = rolling_std.rolling(60, min_periods=20).min().iloc[-1]

        contractions = 0
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
                contractions += 1
                in_pullback = False

        is_low_vol = (latest_std <= std_min_60 * 1.05) if pd.notna(std_min_60) else False
        if contractions == 0 and not is_low_vol:
            return None

        rs_lookback = min(60, len(close))
        rs = min(99, max(1, int(50 + (close.iloc[-1] - close.iloc[-rs_lookback]) / close.iloc[-rs_lookback] * 200)))

        qs = (1 if contractions >= 2 else 0) + (2 if is_low_vol else 0) + (1 if rs >= 70 else 0) + (1 if rs >= 85 else 0)
        quality = "A" if qs >= 4 else "B" if qs >= 2 else "C"

        return {
            "symbol": stock_df["stock_id"].iloc[0],
            "price": round(float(close.iloc[-1]), 2),
            "change_pct": round(float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100), 2),
            "rs_score": rs,
            "contractions": contractions,
            "volume_ratio": round(float(recent_vol / vol_ma_20.iloc[-1]), 2),
            "quality": quality,
            "ma50": round(float(close.rolling(50).mean().iloc[-1]), 2) if len(close) >= 50 else None,
            "ma150": round(float(close.rolling(150).mean().iloc[-1]), 2) if len(close) >= 150 else None,
            "ma200": round(float(close.rolling(200).mean().iloc[-1]), 2) if len(close) >= 200 else None,
            "high_52w": round(float(high.rolling(250, min_periods=1).max().iloc[-1]), 2),
            "low_52w": round(float(low.rolling(250, min_periods=1).min().iloc[-1]), 2),
        }
    except Exception as e:
        print(f"  VCP error: {e}")
        return None

# ========== 主掃描流程 ==========
def full_scan():
    valid_ids = get_filtered_stock_ids()
    if not valid_ids:
        return {"total": 0, "layer1": 0, "layer2": 0, "candidates": []}

    df_all = get_all_market_data(days_lookback=400)
    if df_all.empty:
        return {"total": 0, "layer1": 0, "layer2": 0, "candidates": []}

    layer1_df = batch_process_filter(df_all, valid_ids)
    layer1_count = len(layer1_df)
    if layer1_df.empty:
        return {"total": len(valid_ids), "layer1": 0, "layer2": 0, "candidates": []}

    selected_ids = layer1_df["stock_id"].tolist()
    hist_data = df_all[df_all["stock_id"].isin(selected_ids)]

    layer2_results = []
    for sid, grp in hist_data.groupby("stock_id"):
        grp = grp.set_index("date").sort_index()
        res = vcp_math_check(grp)
        if res:
            layer2_results.append(res)

    layer2_results.sort(key=lambda x: -x["rs_score"])
    layer2_count = len(layer2_results)

    print(f"✅ 第一層: {layer1_count} 檔，第二層: {layer2_count} 檔")
    return {
        "total": len(valid_ids),
        "layer1": layer1_count,
        "layer2": layer2_count,
        "candidates": layer2_results[:10]
    }

@app.get("/scan")
def scan():
    try:
        return full_scan()
    except Exception as e:
        return {"error": str(e), "total": 0, "layer1": 0, "layer2": 0, "candidates": []}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/test/{sid}")
def test_stock(sid: str):
    api = DataLoader()
    if FINMIND_TOKEN:
        api.login_by_token(FINMIND_TOKEN)
    df = api.taiwan_stock_daily(
        stock_id=sid,
        start_date=(datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d"),
        end_date=datetime.today().strftime("%Y-%m-%d")
    )
    if df is None or df.empty:
        return {"error": "no data"}
    return {"columns": df.columns.tolist(), "rows": len(df)}
