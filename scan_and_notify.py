import os
import time
from datetime import datetime, timedelta
from collections import deque

import pandas as pd
import numpy as np
import requests
from FinMind.data import DataLoader

# ========== 設定 ==========
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# 請求頻率控制 ── 滑動窗口 (每小時 500 次)
_request_times = deque()
REQUEST_LIMIT = 500
REQUEST_WINDOW = 3600
_request_lock = None  # 單執行緒無需鎖，但保留以備多線程（GitHub Actions 是單線程）

def _wait_for_slot():
    """等待直到可用請求槽位"""
    global _request_times
    now = time.time()
    # 清除超過 1 小時的舊記錄
    while _request_times and now - _request_times[0] > REQUEST_WINDOW:
        _request_times.popleft()
    if len(_request_times) >= REQUEST_LIMIT:
        oldest = _request_times[0]
        wait_sec = oldest + REQUEST_WINDOW - now + 1
        print(f"⏳ 請求已達 {REQUEST_LIMIT} 次，暫停 {int(wait_sec)} 秒")
        time.sleep(wait_sec)
        return _wait_for_slot()
    _request_times.append(time.time())

# ========== 工具 ==========
def get_filtered_stock_ids():
    """取得普通股代號（排除權證、ETF）"""
    _wait_for_slot()
    api = DataLoader()
    api.login_by_token(FINMIND_TOKEN)
    info = api.taiwan_stock_info()
    info = info[~info["stock_name"].str.contains("權|ETF|存託憑證", na=False)]
    info = info[info["stock_id"].str.len() == 4]
    return info["stock_id"].unique().tolist()

def fetch_daily(sid, start_date, end_date):
    """下載單一股票歷史日線"""
    _wait_for_slot()
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

def _get_col(data, *names):
    for n in names:
        if n in data.columns:
            return data[n]
    return None

# ========== 第一層：Minervini（放寬版） ==========
def minervini_check(data):
    if data is None or len(data) < 200:
        return False
    close = _get_col(data, "close", "Close")
    high  = _get_col(data, "max", "high", "High")
    if close is None or high is None:
        return False
    close = pd.to_numeric(close, errors='coerce').dropna()
    high  = pd.to_numeric(high,  errors='coerce').dropna()
    if len(close) < 200 or len(high) < 200:
        return False
    try:
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()
        last  = close.iloc[-1]
        cond_ma = (last > ma150.iloc[-1]) or (last > ma200.iloc[-1])
        if not cond_ma:
            return False
        if len(high) >= 200:
            high_52w = high.rolling(250, min_periods=1).max().iloc[-1]
            if pd.notna(high_52w) and last < high_52w * 0.65:
                return False
        return True
    except:
        return False

# ========== 第二層：VCP（收緊版） ==========
def vcp_math_check(data):
    if data is None or len(data) < 60:
        return None

    close  = _get_col(data, "close", "Close")
    volume = _get_col(data, "Trading_Volume", "volume", "Volume")
    if close is None or volume is None:
        return None

    close  = pd.to_numeric(close, errors='coerce')
    volume = pd.to_numeric(volume, errors='coerce')

    df_clean = pd.DataFrame({"close": close, "volume": volume}).dropna()
    df_clean = df_clean[(df_clean["close"] > 0) & (df_clean["volume"] > 0)]

    if len(df_clean) < 60:
        return None

    close  = df_clean["close"]
    volume = df_clean["volume"]

    try:
        vol_ma_20 = volume.rolling(20).mean()
        recent_vol = volume.iloc[-3:].mean()
        if pd.isna(vol_ma_20.iloc[-1]) or vol_ma_20.iloc[-1] == 0:
            return None
        vol_ratio = recent_vol / vol_ma_20.iloc[-1]

        # 計算收縮次數
        contractions = 0
        in_pullback = False
        for i in range(5, len(close)):
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

        today_change = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) >= 2 else 0

        # RS 計算
        rs_lookback = min(60, len(close))
        past_close = close.iloc[-rs_lookback]
        if past_close <= 0:
            return None
        rs_raw = 50 + (close.iloc[-1] - past_close) / past_close * 200
        rs = int(max(1, min(99, round(float(rs_raw)))))

        # ── 收緊後的過濾條件 ──
        if rs < 60:
            return None

        cond1 = (contractions >= 2) and (vol_ratio >= 1.0)
        cond2 = (contractions >= 1) and (vol_ratio >= 1.3)
        cond3 = (today_change > 2.0) and (vol_ratio > 1.3)
        cond4 = (contractions >= 5) and (vol_ratio >= 0.8) and (rs >= 92)
        cond5 = (contractions >= 3) and (vol_ratio >= 0.9) and (rs >= 95)

        if not (cond1 or cond2 or cond3 or cond4 or cond5):
            return None

        qs = 0
        if contractions >= 2: qs += 1
        if vol_ratio >= 1.2: qs += 1
        if rs >= 80: qs += 1
        quality = "A" if qs >= 2 else "B" if qs >= 1 else "C"

        return {
            "symbol": str(data["stock_id"].iloc[0]) if "stock_id" in data.columns else "",
            "price": round(float(close.iloc[-1]), 2),
            "change_pct": round(float(today_change), 2),
            "rs_score": rs,
            "contractions": contractions,
            "volume_ratio": round(float(vol_ratio), 2),
            "quality": quality,
        }
    except Exception as e:
        print(f"  VCP error: {e}")
        return None

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
    print("===== VCP 夜間掃描啟動 (GitHub Actions) =====")
    stocks = get_filtered_stock_ids()
    total = len(stocks)
    print(f"總候選股票：{total}")

    start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    end_date   = datetime.today().strftime("%Y-%m-%d")
    candidates = []

    for idx, sid in enumerate(stocks, 1):
        loop_start = time.time()
        df = fetch_daily(sid, start_date, end_date)
        if df is not None and minervini_check(df):
            res = vcp_math_check(df)
            if res:
                candidates.append(res)
                print(f"  ✅ {sid} 通過 (已累計 {len(candidates)})")

        if idx % 100 == 0:
            print(f"📊 進度：{idx}/{total}，已發現 {len(candidates)} 檔，本小時請求：{len(_request_times)}")

        # 基礎間隔，但 _wait_for_slot 已確保不超量，這裡只是減少 CPU 空轉
        elapsed = time.time() - loop_start
        time.sleep(max(0, 7.5 - elapsed))

    # 報告（使用台灣時間）
    tw_time = (datetime.utcnow() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M")
    if candidates:
        candidates.sort(key=lambda x: -x["rs_score"])
        msg = f"<b>📈 VCP 夜間掃描結果 ({tw_time})</b>\n"
        msg += f"掃描 {total} 檔，符合條件 {len(candidates)} 檔\n\n"
        for i, c in enumerate(candidates[:15], 1):
            msg += f"<b>#{i} {c['symbol']}</b>  {c['price']} ({c['change_pct']:+.2f}%)\n"
            msg += f"RS {c['rs_score']} | 收縮 {c['contractions']}次 | 量比 {c['volume_ratio']}x | 品質 {c['quality']}\n\n"
    else:
        msg = f"<b>📈 VCP 夜間掃描結果 ({tw_time})</b>\n"
        msg += f"掃描 {total} 檔，無符合條件股票"

    print(msg)
    send_telegram_message(msg)
    print("===== 掃描完成 =====")

if __name__ == "__main__":
    main()