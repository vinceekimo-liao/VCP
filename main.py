import os
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# ========== 取得普通股列表（修复版） ==========
def get_all_stocks():
    api = DataLoader()
    if FINMIND_TOKEN:
        api.login_by_token(FINMIND_TOKEN)
    info = api.taiwan_stock_info()
    if info is None or info.empty:
        print("❌ 无法获取股票列表")
        return []

    # 直接过滤：长度4位，排除名称含“权”、“ETF”、“存托凭证”
    info = info[
        (info["stock_id"].str.len() == 4) &
        (~info["stock_name"].str.contains("权|ETF|存托凭证"))
    ]
    stock_ids = info["stock_id"].unique().tolist()
    print(f"✅ 过滤后股票数量: {len(stock_ids)}")
    return stock_ids

# ========== 下载单只股票历史数据（增强兼容性） ==========
def fetch_daily(sid, start_date):
    api = DataLoader()
    if FINMIND_TOKEN:
        api.login_by_token(FINMIND_TOKEN)
    try:
        data = api.taiwan_stock_daily(
            stock_id=sid,
            start_date=start_date,
            end_date=datetime.today().strftime("%Y-%m-%d")
        )
        if data is None or data.empty:
            return None
        # 标准化列名
        rename_map = {}
        for col in data.columns:
            if col.lower() in ["close", "收盤價"]:
                rename_map[col] = "close"
            elif col.lower() in ["max", "high", "最高價"]:
                rename_map[col] = "high"
            elif col.lower() in ["min", "low", "最低價"]:
                rename_map[col] = "low"
            elif col.lower() in ["trading_volume", "volume", "成交量"]:
                rename_map[col] = "volume"
            elif col.lower() == "date":
                rename_map[col] = "date"
        if rename_map:
            data = data.rename(columns=rename_map)
        # 确保 date 列
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
            data.sort_values("date", inplace=True)
            data.set_index("date", inplace=True)
        return data
    except Exception as e:
        print(f"  {sid} 下载失败: {e}")
        return None

# ========== 第一层：Minervini 趋势模板（动态字段） ==========
def minervini_check(data):
    if data is None or len(data) < 200:
        return False

    # 确保有 close 和 high 列
    close = None
    high = None
    for col in data.columns:
        if col.lower() == "close":
            close = data[col]
        elif col.lower() in ["high", "max"]:
            high = data[col]
    if close is None or high is None:
        return False

    close = pd.to_numeric(close, errors='coerce').dropna()
    high = pd.to_numeric(high, errors='coerce').dropna()
    if len(close) < 200 or len(high) < 200:
        return False

    try:
        ma50 = close.rolling(50).mean()
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()
        last = close.iloc[-1]
        last_ma150 = ma150.iloc[-1]
        last_ma200 = ma200.iloc[-1]

        if pd.isna(last_ma150) or pd.isna(last_ma200) or pd.isna(last):
            return False
        if not (last > last_ma150 and last > last_ma200):
            return False
        if len(ma200) >= 25:
            ma200_25d_ago = ma200.iloc[-25]
            if pd.notna(ma200_25d_ago) and ma200.iloc[-1] <= ma200_25d_ago:
                return False
        if len(high) >= 200:
            high_52w = high.rolling(250, min_periods=1).max().iloc[-1]
            if pd.notna(high_52w) and last < high_52w * 0.75:
                return False
        return True
    except Exception as e:
        print(f"  minervini_check error: {e}")
        return False

# ========== 第二层：VCP 波动收缩 ==========
def vcp_math_check(data):
    if data is None or len(data) < 60:
        return False

    close = None; volume = None; high = None; low = None
    for col in data.columns:
        c = col.lower()
        if c == "close": close = data[col]
        elif c in ["high", "max"]: high = data[col]
        elif c in ["low", "min"]: low = data[col]
        elif c in ["trading_volume", "volume"]: volume = data[col]
    if close is None or volume is None or high is None or low is None:
        return False

    close = pd.to_numeric(close, errors='coerce').dropna()
    volume = pd.to_numeric(volume, errors='coerce').dropna()
    high = pd.to_numeric(high, errors='coerce').dropna()
    low = pd.to_numeric(low, errors='coerce').dropna()
    if len(close) < 60 or len(volume) < 60:
        return False

    try:
        vol_ma_20 = volume.rolling(20).mean()
        recent_vol = volume.iloc[-3:].mean()
        if pd.isna(vol_ma_20.iloc[-1]): return False

        rolling_std = close.rolling(20).std()
        latest_std = rolling_std.iloc[-1]
        if pd.isna(latest_std): return False
        std_min_60 = rolling_std.rolling(60, min_periods=20).min().iloc[-1]

        contractions = 0
        in_pullback = False
        for i in range(20, len(close)-5):
            try:
                pc = (close.iloc[i]-close.iloc[i-5])/close.iloc[i-5]*100
                vc = (volume.iloc[i]-volume.iloc[i-5])/volume.iloc[i-5]*100 if volume.iloc[i-5]!=0 else 0
            except: continue
            if not in_pullback and pc<-2 and vc<-15: in_pullback=True
            if in_pullback and pc>0:
                contractions += 1
                in_pullback = False

        is_low_vol = (latest_std <= std_min_60*1.05) if pd.notna(std_min_60) else False
        if contractions==0 and not is_low_vol: return False

        rs = min(99, max(1, int(50+(close.iloc[-1]-close.iloc[-60])/close.iloc[-60]*200)))
        qs = (1 if contractions>=2 else 0)+(2 if is_low_vol else 0)+(1 if rs>=70 else 0)+(1 if rs>=85 else 0)
        quality = "A" if qs>=4 else "B" if qs>=2 else "C"

        return {
            "symbol":"","price":round(float(close.iloc[-1]),2),
            "change_pct":round(float((close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100),2),
            "rs_score":rs,"contractions":contractions,
            "volume_ratio":round(float(recent_vol/vol_ma_20.iloc[-1]),2),
            "quality":quality,
            "ma50":round(float(close.rolling(50).mean().iloc[-1]),2) if len(close)>=50 else None,
            "ma150":round(float(close.rolling(150).mean().iloc[-1]),2) if len(close)>=150 else None,
            "ma200":round(float(close.rolling(200).mean().iloc[-1]),2) if len(close)>=200 else None,
            "high_52w":round(float(high.rolling(250,min_periods=1).max().iloc[-1]),2) if len(high)>=200 else round(float(high.max()),2),
            "low_52w":round(float(low.rolling(250,min_periods=1).min().iloc[-1]),2) if len(low)>=200 else round(float(low.min()),2),
        }
    except Exception as e:
        print(f"vcp error: {e}")
        return False

# ========== 全扫描（控制请求速率） ==========
def full_scan():
    start = datetime.today() - timedelta(days=400)
    start_str = start.strftime("%Y-%m-%d")
    stocks = get_all_stocks()
    total = len(stocks)
    print(f"📊 待扫描股票: {total}")
    if total == 0:
        return {"total":0,"layer1":0,"layer2":0,"candidates":[]}

    # 第一层：分批下载，最大并行5，每批间隔2秒
    layer1_results = []
    batch_size = 20
    for i in range(0, total, batch_size):
        batch = stocks[i:i+batch_size]
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_daily, sid, start_str): sid for sid in batch}
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    df = future.result(timeout=25)
                    if minervini_check(df):
                        layer1_results.append((sid, df))
                except: pass
        time.sleep(2.0)

    layer1_count = len(layer1_results)
    print(f"✅ 第一层通过: {layer1_count} 档")

    # 第二层
    layer2_results = []
    for sid, df in layer1_results:
        res = vcp_math_check(df)
        if res:
            res["symbol"] = sid
            layer2_results.append(res)

    layer2_count = len(layer2_results)
    print(f"✅ 第二层通过: {layer2_count} 档")
    layer2_results.sort(key=lambda x: -x["rs_score"])
    return {"total":total,"layer1":layer1_count,"layer2":layer2_count,"candidates":layer2_results[:10]}

@app.get("/scan")
def scan():
    try:
        return full_scan()
    except Exception as e:
        return {"error":str(e),"total":0,"layer1":0,"layer2":0,"candidates":[]}

@app.get("/health")
def health():
    return {"status":"ok"}

# 单股测试端点（可选）
@app.get("/test/{sid}")
def test_stock(sid: str):
    df = fetch_daily(sid, (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d"))
    if df is None:
        return {"error": "no data"}
    print("Columns:", df.columns.tolist())
    print(df.head())
    return {"columns": df.columns.tolist(), "rows": len(df)}
