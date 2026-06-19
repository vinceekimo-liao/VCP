import os
import time
from datetime import datetime, timedelta
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

# ========== 工具：取得所有普通股代號 ==========
def get_all_stocks():
    api = DataLoader()
    if FINMIND_TOKEN:
        api.login_by_token(FINMIND_TOKEN)
    info = api.taiwan_stock_info()
    if info is None or info.empty:
        print("❌ taiwan_stock_info() 回傳空資料")
        return []

    # 動態尋找 type 與 stock_id 欄位
    type_col = next((c for c in info.columns if c.lower().strip() == "type"), None)
    id_col = next((c for c in info.columns if c.lower().strip() == "stock_id"), None)
    if not type_col or not id_col:
        print(f"❌ 找不到必要欄位，現有：{info.columns.tolist()}")
        return []

    # 寬鬆篩選普通股
    mask = (
        info[type_col].str.lower().str.contains('common', na=False) |
        info[type_col].str.contains('股', na=False)
    )
    known_types = ['Common Stock', 'common stock', '股票', 'Equity']
    mask |= info[type_col].isin(known_types)

    common = info[mask]
    if common.empty:
        common = info[info[id_col].astype(str).str.len() == 4]   # fallback
    else:
        common = common[common[id_col].astype(str).str.len() == 4]

    stock_ids = common[id_col].tolist()
    print(f"✅ 普通股數量：{len(stock_ids)}")
    return stock_ids

# ========== 核心改造：批量下載歷史資料 ==========
def build_stock_data(stock_ids, start_date, end_date):
    """
    一次性逐日下載全市場股價，再按股票重組為 DataFrame。
    返回 dict：{symbol: DataFrame}，DataFrame 格式與原 fetch_daily 相同。
    """
    api = DataLoader()
    if FINMIND_TOKEN:
        api.login_by_token(FINMIND_TOKEN)

    # 取得日期範圍（只取交易日）
    date_range = pd.bdate_range(start=start_date, end=end_date)
    all_data = []

    for d in date_range:
        date_str = d.strftime("%Y-%m-%d")
        try:
            # 一次下載當日所有股票價格
            daily = api.taiwan_stock_price_list(date=date_str)
            if daily is not None and not daily.empty:
                all_data.append(daily)
        except Exception as e:
            print(f"  ⚠️ 無法下載 {date_str}：{e}")
        # 稍微暫停，避免請求過快
        time.sleep(0.05)

    if not all_data:
        print("❌ 完全沒有下載到任何資料")
        return {}

    df = pd.concat(all_data, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["stock_id"].isin(stock_ids)]         # 只保留我們要的股票
    df = df.sort_values(["stock_id", "date"])

    # 按股票分組，並設定索引
    stock_data = {}
    for symbol, group in df.groupby("stock_id"):
        group = group.set_index("date").sort_index()
        stock_data[symbol] = group
    return stock_data

# ========== 第一層：Minervini 趨勢模板 ==========
def minervini_check(data):
    if data is None or len(data) < 200:
        return False

    # 自動適配欄位名稱
    close = data["close"] if "close" in data.columns else data.get("Close")
    high = data["max"] if "max" in data.columns else data.get("High")
    if close is None or high is None:
        return False

    try:
        ma50 = close.rolling(50).mean()
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()
        last = close.iloc[-1]

        if not (last > ma150.iloc[-1] and last > ma200.iloc[-1]):
            return False
        if len(ma200) >= 25 and ma200.iloc[-1] <= ma200.iloc[-25]:
            return False
        if len(high) >= 200:
            high_52w = high.rolling(250, min_periods=1).max().iloc[-1]
            if pd.notna(high_52w) and last < high_52w * 0.75:
                return False
        return True
    except:
        return False

# ========== 第二層：VCP 數學波動收縮 ==========
def vcp_math_check(data):
    if data is None or len(data) < 60:
        return False

    close = data["close"] if "close" in data.columns else data.get("Close")
    volume = data["Trading_Volume"] if "Trading_Volume" in data.columns else data.get("Volume")
    high = data["max"] if "max" in data.columns else data.get("High")
    low = data["min"] if "min" in data.columns else data.get("Low")
    if close is None or volume is None or high is None or low is None:
        return False

    try:
        vol_ma_20 = volume.rolling(20).mean()
        recent_vol = volume.iloc[-3:].mean()
        if pd.isna(vol_ma_20.iloc[-1]):
            return False

        rolling_std = close.rolling(20).std()
        latest_std = rolling_std.iloc[-1]
        if pd.isna(latest_std):
            return False
        std_min_60 = rolling_std.rolling(60, min_periods=20).min().iloc[-1]

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

        is_low_vol = (latest_std <= std_min_60 * 1.05) if pd.notna(std_min_60) else False
        if contractions == 0 and not is_low_vol:
            return False

        rs_lookback = min(60, len(close))
        rs = min(99, max(1, int(50 + (close.iloc[-1] - close.iloc[-rs_lookback]) / close.iloc[-rs_lookback] * 200)))

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
            "high_52w": round(float(high.rolling(250, min_periods=1).max().iloc[-1]), 2) if len(high) >= 200 else round(float(high.max()), 2),
            "low_52w": round(float(low.rolling(250, min_periods=1).min().iloc[-1]), 2) if len(low) >= 200 else round(float(low.min()), 2),
        }
    except Exception as e:
        print(f"  vcp_math_check error: {e}")
        return False

# ========== 主掃描（已改為批量下載） ==========
def full_scan():
    start = datetime.today() - timedelta(days=400)
    end = datetime.today()
    stocks = get_all_stocks()
    total = len(stocks)
    print(f"📊 總股票數: {total}")
    if total == 0:
        return {"total": 0, "layer1": 0, "layer2": 0, "candidates": []}

    print("⏳ 開始批量下載歷史資料（每日全市場）...")
    stock_data = build_stock_data(stocks, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    print(f"✅ 已下載 {len(stock_data)} 檔股票的歷史資料")

    # 第一層篩選
    layer1_results = []
    for sid, df in stock_data.items():
        if minervini_check(df):
            layer1_results.append((sid, df))
    layer1_count = len(layer1_results)
    print(f"✅ 第一層通過: {layer1_count} 檔")

    # 第二層篩選
    layer2_results = []
    for sid, df in layer1_results:
        result = vcp_math_check(df)
        if result:
            result["symbol"] = sid
            layer2_results.append(result)
    layer2_count = len(layer2_results)
    print(f"✅ 第二層通過: {layer2_count} 檔")

    layer2_results.sort(key=lambda x: (-x["rs_score"]))
    return {"total": total, "layer1": layer1_count, "layer2": layer2_count, "candidates": layer2_results[:10]}

# ========== API ==========
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
