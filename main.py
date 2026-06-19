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

# ══════════════════════════════════════════════════
# 工具：取得普通股代號
# ══════════════════════════════════════════════════
def get_all_stocks():
    api = DataLoader()
    if FINMIND_TOKEN:
        api.login_by_token(FINMIND_TOKEN)
    info = api.taiwan_stock_info()
    if info is None or info.empty:
        print("❌ 無法取得股票清單")
        return []

    type_col = next((c for c in info.columns if c.lower().strip() == "type"), None)
    id_col   = next((c for c in info.columns if c.lower().strip() == "stock_id"), None)
    if not type_col or not id_col:
        print(f"❌ 找不到 type/stock_id 欄位，現有：{info.columns.tolist()}")
        return []

    mask = (
        info[type_col].str.lower().str.contains('common', na=False) |
        info[type_col].str.contains('股', na=False)
    )
    known = ['Common Stock', 'common stock', '股票', 'Equity']
    mask |= info[type_col].isin(known)

    common = info[mask]
    if common.empty:
        common = info[info[id_col].astype(str).str.len() == 4]
    else:
        common = common[common[id_col].astype(str).str.len() == 4]

    return common[id_col].tolist()

# ══════════════════════════════════════════════════
# 批量下載全市場歷史資料（一次一天）
# ══════════════════════════════════════════════════
def build_stock_data(stock_ids, start_date, end_date):
    api = DataLoader()
    if FINMIND_TOKEN:
        api.login_by_token(FINMIND_TOKEN)

    date_range = pd.bdate_range(start=start_date, end=end_date)
    frames = []
    for d in date_range:
        ds = d.strftime("%Y-%m-%d")
        try:
            daily = api.taiwan_stock_price_list(date=ds)
            if daily is not None and not daily.empty:
                frames.append(daily)
        except Exception as e:
            print(f"  ⚠️ 下載 {ds} 失敗：{e}")
        time.sleep(0.05)

    if not frames:
        print("❌ 完全沒有下載到任何資料")
        return {}

    df_all = pd.concat(frames, ignore_index=True)

    # ★ 除錯：印出欄位名稱與第一筆資料
    print("📋 下載欄位名稱：", df_all.columns.tolist())
    print("📋 前兩筆資料：\n", df_all.head(2).to_string())

    # 只保留目標股票
    df_all = df_all[df_all["stock_id"].isin(stock_ids)]
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all = df_all.sort_values(["stock_id", "date"])

    stock_data = {}
    for sid, grp in df_all.groupby("stock_id"):
        grp = grp.set_index("date").sort_index()
        stock_data[sid] = grp
    return stock_data

# ══════════════════════════════════════════════════
# 第一層：Minervini 趨勢模板（已增強欄位相容）
# ══════════════════════════════════════════════════
def minervini_check(data):
    if data is None or len(data) < 200:
        return False

    # 動態抓取 close 與 high 欄位（優先小寫）
    close = None
    high = None
    for col in data.columns:
        if col.lower() in ["close", "closeprice"]:
            close = data[col]
        if col.lower() in ["max", "high"]:
            high = data[col]
    if close is None or high is None:
        return False

    try:
        ma50 = close.rolling(50).mean()
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()
        last = close.iloc[-1]

        # 條件放寬：只需收盤 > MA150 且 > MA200
        if not (last > ma150.iloc[-1] and last > ma200.iloc[-1]):
            return False

        # MA200 近 25 日必須向上
        if len(ma200) >= 25 and ma200.iloc[-1] <= ma200.iloc[-25]:
            return False

        # 距 52 週高點 ≤ 25%
        if len(high) >= 200:
            high_52w = high.rolling(250, min_periods=1).max().iloc[-1]
            if pd.notna(high_52w) and last < high_52w * 0.75:
                return False

        return True
    except:
        return False

# ══════════════════════════════════════════════════
# 第二層：VCP 波動收縮（已增強欄位相容）
# ══════════════════════════════════════════════════
def vcp_math_check(data):
    if data is None or len(data) < 60:
        return False

    # 動態抓取 close, volume, high, low
    close = None; volume = None; high = None; low = None
    for col in data.columns:
        c = col.lower()
        if c in ["close", "closeprice"]:          close = data[col]
        if c in ["trading_volume", "volume"]:     volume = data[col]
        if c in ["max", "high"]:                  high = data[col]
        if c in ["min", "low"]:                   low = data[col]
    if close is None or volume is None or high is None or low is None:
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

        is_low_vol = latest_std <= std_min_60*1.05 if pd.notna(std_min_60) else False
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

# ══════════════════════════════════════════════════
# 主掃描
# ══════════════════════════════════════════════════
def full_scan():
    start = datetime.today() - timedelta(days=400)
    end   = datetime.today()
    stocks = get_all_stocks()
    total = len(stocks)
    print(f"📊 總股票數: {total}")
    if total == 0:
        return {"total":0,"layer1":0,"layer2":0,"candidates":[]}

    stock_data = build_stock_data(stocks, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    print(f"✅ 成功下載 {len(stock_data)} 檔股票資料")

    # 第一層
    layer1 = []
    for sid, df in stock_data.items():
        if minervini_check(df):
            layer1.append((sid, df))
    print(f"第一層通過: {len(layer1)} 檔")

    # 第二層
    layer2 = []
    for sid, df in layer1:
        res = vcp_math_check(df)
        if res:
            res["symbol"] = sid
            layer2.append(res)
    print(f"第二層通過: {len(layer2)} 檔")
    layer2.sort(key=lambda x: -x["rs_score"])
    return {"total":total,"layer1":len(layer1),"layer2":len(layer2),"candidates":layer2[:10]}

@app.get("/scan")
def scan():
    try: return full_scan()
    except Exception as e: return {"error":str(e),"total":0,"layer1":0,"layer2":0,"candidates":[]}

@app.get("/health")
def health():
    return {"status":"ok"}
