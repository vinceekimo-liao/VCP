import os
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np
from FinMind.data import DataLoader

# ========== 設定 ==========
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN")

# ========== Telegram 通知 ==========
def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("缺少 Telegram 設定，跳過通知")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram 發送失敗：{e}")

# ========== 以下是原始掃描邏輯 (main.py 中的函式) ==========
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
        if data.empty: return None
        data["date"] = pd.to_datetime(data["date"])
        data.sort_values("date", inplace=True)
        data.set_index("date", inplace=True)
        return data
    except: return None

def minervini_check(data):
    if data is None or len(data) < 200: return False
    close = data["close"]
    try:
        ma50 = close.rolling(50).mean()
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()
        last = close.iloc[-1]
        if not (last > ma50.iloc[-1] > ma150.iloc[-1] > ma200.iloc[-1]): return False
        if len(ma200) >= 25 and ma200.iloc[-1] <= ma200.iloc[-25]: return False
        if "max" in data.columns and len(data) >= 250:
            if last < data["max"].rolling(250).max().iloc[-1] * 0.75: return False
        return True
    except: return False

def vcp_math_check(data):
    if data is None or len(data) < 60: return False
    try:
        close = data["close"]
        volume = data["volume"]
        high = data["max"]
        low = data["min"]
        vol_ma_20 = volume.rolling(20).mean()
        recent_vol = volume.iloc[-3:].mean()
        if pd.isna(vol_ma_20.iloc[-1]): return False
        contractions = 0
        trough_prices = []
        in_pullback = False
        for i in range(20, len(close) - 5):
            try:
                pc = (close.iloc[i] - close.iloc[i-5]) / close.iloc[i-5] * 100
                vc = (volume.iloc[i] - volume.iloc[i-5]) / volume.iloc[i-5] * 100 if volume.iloc[i-5] != 0 else 0
            except: continue
            if not in_pullback and pc < -2 and vc < -15: in_pullback = True
            if in_pullback and pc > 0:
                if len(trough_prices) == 0 or close.iloc[i] > trough_prices[-1]:
                    trough_prices.append(close.iloc[i]); contractions += 1
                in_pullback = False
        if contractions < 2: return False
        rs_lookback = min(60, len(close))
        rs = min(99, max(1, int(50 + (close.iloc[-1] - close.iloc[-rs_lookback]) / close.iloc[-rs_lookback] * 200)))
        return {
            "symbol": "",
            "price": round(float(close.iloc[-1]), 2),
            "change_pct": round(float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100), 2),
            "rs_score": rs, "contractions": contractions,
            "volume_ratio": round(float(recent_vol / vol_ma_20.iloc[-1]), 2),
            "quality": "A" if contractions >= 3 else "B" if contractions >= 2 else "C",
        }
    except: return False

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
                    if minervini_check(df): layer1_results.append((sid, df))
                except: pass
        time.sleep(0.3)
    print(f"第一層通過：{len(layer1_results)} 檔")
    layer2_results = []
    for sid, df in layer1_results:
        result = vcp_math_check(df)
        if result:
            result["symbol"] = sid
            layer2_results.append(result)
    layer2_results.sort(key=lambda x: (-x["rs_score"]))
    print(f"第二層通過：{len(layer2_results)} 檔")
    return {"total": total, "layer1": len(layer1_results), "layer2": len(layer2_results), "candidates": layer2_results[:10]}

# ========== 主程式：掃描 + 通知 ==========
if __name__ == "__main__":
    result = full_scan()
    candidates = result["candidates"]

    # 組合 Telegram 訊息
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"<b>📈 VCP 漏斗掃描結果</b> ({now})\n"
    msg += f"全台股 {result['total']} 檔 → 趨勢模板 {result['layer1']} 檔 → VCP 收縮 {result['layer2']} 檔\n\n"

    if candidates:
        for i, c in enumerate(candidates, 1):
            msg += f"<b>#{i} {c['symbol']}</b>\n"
            msg += f"價格：{c['price']} ({c['change_pct']:+.2f}%)\n"
            msg += f"RS：{c['rs_score']} | 收縮：{c['contractions']}次 | 量比：{c['volume_ratio']}x\n"
            msg += f"品質：{c['quality']} 級\n\n"
    else:
        msg += "⚠️ 本日無符合 VCP 條件之股票"

    print(msg)
    send_telegram_message(msg)
