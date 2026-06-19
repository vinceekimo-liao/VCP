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

# ══════════════════════════════════════════════════
# 取得普通股（排除權證、ETF、特別股等）
# ══════════════════════════════════════════════════
def get_all_stocks():
    api = DataLoader()
    if FINMIND_TOKEN:
        api.login_by_token(FINMIND_TOKEN)
    info = api.taiwan_stock_info()
    if info is None or info.empty:
        print("❌ 無法取得股票清單")
        return []

    # 動態尋找欄位（相容大小寫）
    type_col = next((c for c in info.columns if c.lower().strip() == "type"), None)
    id_col   = next((c for c in info.columns if c.lower().strip() == "stock_id"), None)
    name_col = next((c for c in info.columns if c.lower().strip() in ["stock_name", "name"]), None)

    if not type_col or not id_col:
        print(f"❌ 找不到必要欄位，現有：{info.columns.tolist()}")
        return []

    # 只取普通股
    mask = info[type_col].str.lower().str.contains('common', na=False)
    known = ['Common Stock', 'common stock', '股票', 'Equity']
    mask |= info[type_col].isin(known)
    common = info[mask]

    # 過濾權證、ETF、存託憑證（透過名稱）
    if name_col:
        bad_names = common[name_col].str.contains('權|ETF|存託憑證|反1|正2|加權', na=False)
        common = common[~bad_names]

    # 只留股票代碼為 4 碼
    common = common[common[id_col].astype(str).str.len() == 4]

    stock_ids = common[id_col].tolist()
    print(f"✅ 普通股總數（已排除雜項）：{len(stock_ids)}")
    return stock_ids

# ══════════════════════════════════════════════════
# 下載單一股票歷史資料（修正 API 參數）
# ══════════════════════════════════════════════════
def fetch_daily(sid, start_date):
    api = DataLoader()
    if FINMIND_TOKEN:
        api.login_by_token(FINMIND_TOKEN)
    try:
        # ✅ 正確的參數：stock_id, start_date, end_date
        data = api.taiwan_stock_daily(
            stock_id=sid,
            start_date=start_date,
            end_date=datetime.today().strftime("%Y-%m-%d")
        )
        if data is None or data.empty:
            return None
        data["date"] = pd.to_datetime(data["date"])
        data.sort_values("date", inplace=True)
        data.set_index("date", inplace=True)
        return data
    except Exception as e:
        # 不打印太頻繁，只記錄前幾次
        print(f"  {sid} 下載失敗：{e}")
        return None

# ══════════════════════════════════════════════════
# 第一層：Minervini 趨勢模板（安全強化）
# ══════════════════════════════════════════════════
def minervini_check(data):
    if data is None or len(data) < 200:
        return False

    # 動態抓取欄位
    close = None
    high = None
    for col in data.columns:
        if col.lower() in ["close", "closeprice"]:
            close = data[col]
        if col.lower() in ["max", "high"]:
            high = data[col]
    if close is None or high is None:
        return False

    # 強制轉數值，並丟棄 NaN
    close = pd.to_numeric(close, errors='coerce').dropna()
    high  = pd.to_numeric(high,  errors='coerce').dropna()

    if len(close) < 200 or len(high) < 200:
        return False

    try:
        ma50  = close.rolling(50).mean()
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()

        last_ma150 = ma150.iloc[-1]
        last_ma200 = ma200.iloc[-1]
        last = close.iloc[-1]

        # 檢查 NaN
        if pd.isna(last_ma150) or pd.isna(last_ma200) or pd.isna(last):
            return False

        # 放寬條件：收盤 > MA150 且 > MA200
        if not (last > last_ma150 and last > last_ma200):
            return False

        # MA200 近 25 日向上
        if len(ma200) >= 25:
            ma200_25d_ago = ma200.iloc[-25]
            if pd.notna(ma200_25d_ago) and ma200.iloc[-1] <= ma200_25d_ago:
                return False

        # 距 52 週高點 ≤ 25%
        if len(high) >= 200:
            high_52w = high.rolling(250, min_periods=1).max().iloc[-1]
            if pd.notna(high_52w) and last < high_52w * 0.75:
                return False

        return True
    except Exception as e:
        print(f"  minervini_check error: {e}")
        return False

# ══════════════════════════════════════════════════
# 第二層：VCP 波動收縮（安全強化）
# ══════════════════════════════════════════════════
def vcp_math_check(data):
    if data is None or len(data) < 60:
        return False

    # 動態抓取
    close = None; volume = None; high = None; low = None
    for col in data.columns:
        c = col.lower()
        if c in ["close", "closeprice"]:          close = data[col]
        if c in ["trading_volume", "volume"]:     volume = data[col]
        if c in ["max", "high"]:                  high = data[col]
        if c in ["min", "low"]:                   low = data[col]
    if close is None or volume is None or high is None or low is None:
        return False

    close  = pd.to_numeric(close, errors='coerce').dropna()
    volume = pd.to_numeric(volume, errors='coerce').dropna()
    high   = pd.to_numeric(high, errors='coerce').dropna()
    low    = pd.to_numeric(low, errors='coerce').dropna()

    if len(close) < 60 or len(volume) < 60:
        return False

    try:
        vol_ma_20  = volume.rolling(20).mean()
        recent_vol = volume.iloc[-3:].mean()
        if pd.isna(vol_ma_20.iloc[-1]):
            return False

        rolling_std = close.rolling(20).std()
        latest_std  = rolling_std.iloc[-1]
        if pd.isna(latest_std):
            return False
        std_min_60  = rolling_std.rolling(60, min_periods=20).min().iloc[-1]

        contractions = 0
        in_pullback  = False
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
            return False

        rs = min(99, max(1, int(50 + (close.iloc[-1] - close.iloc[-60]) / close.iloc[-60] * 200)))
        qs = (1 if contractions >= 2 else 0) + (2 if is_low_vol else 0) + (1 if rs >= 70 else 0) + (1 if rs >= 85 else 0)
        quality = "A" if qs >= 4 else "B" if qs >= 2 else "C"

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
            "high_52w": round(float(high.rolling(250, min_periods=1).max().iloc[-1]), 2) if len(high) >= 200 else round(float(high.max()), 2),
            "low_52w": round(float(low.rolling(250, min_periods=1).min().iloc[-1]), 2) if len(low) >= 200 else round(float(low.min()), 2),
        }
    except Exception as e:
        print(f"  vcp error: {e}")
        return False

# ══════════════════════════════════════════════════
# 主掃描（控制並行與延遲）
# ══════════════════════════════════════════════════
def full_scan():
    start = datetime.today() - timedelta(days=400)
    start_str = start.strftime("%Y-%m-%d")
    stocks = get_all_stocks()
    total = len(stocks)
    print(f"📊 總候選股票: {total}")
    if total == 0:
        return {"total": 0, "layer1": 0, "layer2": 0, "candidates": []}

    # 第一層：下載 + 趨勢篩選（並行控制：最多 5 個 worker，每批間隔 2 秒）
    layer1_results = []
    batch_size = 20
    for i in range(0, total, batch_size):
        batch = stocks[i:i+batch_size]
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_daily, sid, start_str): sid for sid in batch}
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    df = future.result(timeout=20)
                    if minervini_check(df):
                        layer1_results.append((sid, df))
                except:
                    pass
        # 重要：每批之間暫停，確保每小時不超過 600 次請求
        time.sleep(2.0)

    layer1_count = len(layer1_results)
    print(f"✅ 第一層通過: {layer1_count} 檔")

    # 第二層
    layer2_results = []
    for sid, df in layer1_results:
        res = vcp_math_check(df)
        if res:
            res["symbol"] = sid
            layer2_results.append(res)

    layer2_count = len(layer2_results)
    print(f"✅ 第二層通過: {layer2_count} 檔")
    layer2_results.sort(key=lambda x: -x["rs_score"])
    return {"total": total, "layer1": layer1_count, "layer2": layer2_count, "candidates": layer2_results[:10]}

# ══════════════════════════════════════════════════
# API 端點
# ══════════════════════════════════════════════════
@app.get("/scan")
def scan():
    try:
        return full_scan()
    except Exception as e:
        return {"error": str(e), "total": 0, "layer1": 0, "layer2": 0, "candidates": []}

@app.get("/health")
def health():
    return {"status": "ok"}
