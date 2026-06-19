import os
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from FinMind.data import DataLoader

# ========== 設定 ==========
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# 速率控制：每小時最多 500 次 → 每筆間隔 7.2 秒，取 7.5 秒保險
REQUEST_INTERVAL = 7.5

# ========== 工具 ==========
def get_filtered_stock_ids():
    """取得普通股代號（排除權證、ETF）"""
    api = DataLoader()
    api.login_by_token(FINMIND_TOKEN)
    info = api.taiwan_stock_info()
    info = info[~info["stock_name"].str.contains("權|ETF|存託憑證", na=False)]
    info = info[info["stock_id"].str.len() == 4]
    return info["stock_id"].unique().tolist()

def fetch_daily(sid, start_date, end_date):
    """下載單一股票歷史日線"""
    api = DataLoader()
    api.login_by_token(FINMIND_TOKEN)
    try:
        data = api.taiwan_stock_daily(
            stock_id=sid,
            start_date=start_date,
            end_date=end_date
        )
        if data is None or data.empty:
            return None
        data["date"] = pd.to_datetime(data["date"])
        data.sort_values("date", inplace=True)
        data.set_index("date", inplace=True)
        return data
    except Exception as e:
        print(f"  {sid} 失敗：{e}")
        return None

# ========== 篩選邏輯 ==========
def minervini_check(data):
    if data is None or len(data) < 200:
        return False
    try:
        close = data["close"]
        high  = data["high"]
    except KeyError:
        return False

    close = pd.to_numeric(close, errors='coerce').dropna()
    high  = pd.to_numeric(high,  errors='coerce').dropna()
    if len(close) < 200 or len(high) < 200:
        return False

    ma50  = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()
    last  = close.iloc[-1]
    if pd.isna(ma150.iloc[-1]) or pd.isna(ma200.iloc[-1]):
        return False
    if not (last > ma150.iloc[-1] and last > ma200.iloc[-1]):
        return False
    if len(ma200) >= 25:
        if (ma200.iloc[-1] / ma200.iloc[-25] - 1) < -0.02:
            return False
    if len(high) >= 200:
        high_52w = high.rolling(250, min_periods=1).max().iloc[-1]
        if pd.notna(high_52w) and last < high_52w * 0.75:
            return False
    return True

def vcp_math_check(data):
    if data is None or len(data) < 60:
        return None
    close  = pd.to_numeric(data["close"], errors='coerce').dropna()
    high   = pd.to_numeric(data["high"], errors='coerce').dropna()
    low    = pd.to_numeric(data["low"], errors='coerce').dropna()
    volume = pd.to_numeric(data["volume"], errors='coerce').dropna()
    if len(close) < 60 or len(volume) < 60:
        return None

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
    for i in range(20, len(close)-5):
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

    is_low_vol = latest_std <= std_min_60 * 1.05 if pd.notna(std_min_60) else False
    if contractions == 0 and not is_low_vol:
        return None

    rs = min(99, max(1, int(50 + (close.iloc[-1] - close.iloc[-60]) / close.iloc[-60] * 200)))
    qs = (1 if contractions >= 2 else 0) + (2 if is_low_vol else 0) + (1 if rs >= 70 else 0) + (1 if rs >= 85 else 0)
    quality = "A" if qs >= 4 else "B" if qs >= 2 else "C"

    return {
        "symbol": data["stock_id"].iloc[0],
        "price": round(float(close.iloc[-1]), 2),
        "change_pct": round(float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100), 2),
        "rs_score": rs,
        "contractions": contractions,
        "volume_ratio": round(float(recent_vol / vol_ma_20.iloc[-1]), 2),
        "quality": quality,
        "ma50": round(float(close.rolling(50).mean().iloc[-1]), 2),
        "ma150": round(float(close.rolling(150).mean().iloc[-1]), 2),
        "ma200": round(float(close.rolling(200).mean().iloc[-1]), 2),
        "high_52w": round(float(high.rolling(250, min_periods=1).max().iloc[-1]), 2),
        "low_52w": round(float(low.rolling(250, min_periods=1).min().iloc[-1]), 2),
    }

# ========== Telegram 通知 ==========
def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram 發送失敗：{e}")

# ========== 主流程 ==========
def main():
    print("===== VCP 夜間掃描啟動 =====")
    stocks = get_filtered_stock_ids()
    total = len(stocks)
    print(f"總候選股票：{total}")

    start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    end_date   = datetime.today().strftime("%Y-%m-%d")
    candidates = []

    for idx, sid in enumerate(stocks, 1):
        loop_start = time.time()
        # 下載
        df = fetch_daily(sid, start_date, end_date)
        if df is not None and minervini_check(df):
            res = vcp_math_check(df)
            if res:
                candidates.append(res)
                print(f"  ✅ {sid} 通過 (已累計 {len(candidates)})")

        # 進度顯示
        if idx % 100 == 0:
            print(f"📊 進度：{idx}/{total}，已發現 {len(candidates)} 檔")

        # 速率控制
        elapsed = time.time() - loop_start
        sleep_time = max(0, REQUEST_INTERVAL - elapsed)
        time.sleep(sleep_time)

    # 報告
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    if candidates:
        # 按 RS 排序
        candidates.sort(key=lambda x: -x["rs_score"])
        msg = f"<b>📈 VCP 夜間掃描結果 ({now_str})</b>\n"
        msg += f"掃描 {total} 檔，符合條件 {len(candidates)} 檔\n\n"
        for i, c in enumerate(candidates[:10], 1):
            msg += f"<b>#{i} {c['symbol']}</b>  {c['price']} ({c['change_pct']:+.2f}%)\n"
            msg += f"RS {c['rs_score']} | 收縮 {c['contractions']}次 | 量比 {c['volume_ratio']}x | 品質 {c['quality']}\n\n"
    else:
        msg = f"<b>📈 VCP 夜間掃描結果 ({now_str})</b>\n"
        msg += f"掃描 {total} 檔，無符合條件股票"

    print(msg)
    send_telegram_message(msg)
    print("===== 掃描完成 =====")

if __name__ == "__main__":
    main()
