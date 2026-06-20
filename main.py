import os
import time
import threading
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests
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

# ========== 全域變數 ==========
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

scan_results = []          # 儲存最終候選股
is_scanning = False        # 防止重複觸發
scan_lock = threading.Lock()
last_report_msg = "尚無報告"  # 供前端顯示

# ========== Telegram 通知 ==========
def send_telegram_msg(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ 缺少 Telegram 設定")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Telegram 發送失敗：{e}")

# ========== 取得股票清單（快取 24 小時） ==========
_stock_ids_cache = {"ids": [], "ts": 0}

def get_filtered_stock_ids():
    now = time.time()
    if _stock_ids_cache["ids"] and (now - _stock_ids_cache["ts"]) < 86400:
        return _stock_ids_cache["ids"]

    api = DataLoader()
    api.login_by_token(FINMIND_TOKEN)
    info = api.taiwan_stock_info()
    if info is None or info.empty:
        return []

    # 排除權證、ETF、存託憑證
    info = info[~info["stock_name"].str.contains("權|ETF|存託憑證", na=False)]
    info = info[info["stock_id"].str.len() == 4]
    ids = info["stock_id"].unique().tolist()
    _stock_ids_cache["ids"] = ids
    _stock_ids_cache["ts"] = now
    print(f"📋 普通股代號數量：{len(ids)}")
    return ids

# ========== 資料下載 ==========
def fetch_daily(sid, start_date, end_date):
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
        print(f"  {sid} 下載失敗：{e}")
        return None

# ========== 第一層：Minervini 趨勢模板 ==========
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

    try:
        ma50  = close.rolling(50).mean()
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()
        last  = close.iloc[-1]
        if pd.isna(ma150.iloc[-1]) or pd.isna(ma200.iloc[-1]):
            return False
        if not (last > ma150.iloc[-1] and last > ma200.iloc[-1]):
            return False
        # MA200 趨勢放寬：只淘汰跌幅 > 2%
        if len(ma200) >= 25:
            if (ma200.iloc[-1] / ma200.iloc[-25] - 1) < -0.02:
                return False
        # 距 52 週高點 ≤ 25%
        if len(high) >= 200:
            high_52w = high.rolling(250, min_periods=1).max().iloc[-1]
            if pd.notna(high_52w) and last < high_52w * 0.75:
                return False
        return True
    except:
        return False

# ========== 第二層：VCP 波動收縮 ==========
def vcp_math_check(data):
    if data is None or len(data) < 60:
        return None
    close  = pd.to_numeric(data["close"], errors='coerce').dropna()
    high   = pd.to_numeric(data["high"], errors='coerce').dropna()
    low    = pd.to_numeric(data["low"], errors='coerce').dropna()
    volume = pd.to_numeric(data["volume"], errors='coerce').dropna()
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

        is_low_vol = (latest_std <= std_min_60 * 1.05) if pd.notna(std_min_60) else False
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
        }
    except Exception as e:
        print(f"  VCP error: {e}")
        return None

# ========== 報告建立 ==========
def build_report(total, results):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    if not results:
        return f"📉 <b>每日 VCP 報告 ({now_str})</b>\n掃描 {total} 檔，無符合條件股票"
    sorted_results = sorted(results, key=lambda x: -x["rs_score"])
    msg = f"📈 <b>每日 VCP 報告 ({now_str})</b>\n掃描 {total} 檔，符合 {len(results)} 檔\n\n"
    for i, c in enumerate(sorted_results[:10], 1):
        msg += f"🔹 <b>{c['symbol']}</b> | 價:{c['price']} | RS:{c['rs_score']} | 品質:{c['quality']}\n"
    return msg

# ========== 背景掃描任務 ==========
def background_scanner():
    global scan_results, is_scanning, last_report_msg
    with scan_lock:
        if is_scanning:
            return
        is_scanning = True

    try:
        stocks = get_filtered_stock_ids()
        total = len(stocks)
        start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")
        local_results = []

        for idx, sid in enumerate(stocks, 1):
            loop_start = time.time()
            df = fetch_daily(sid, start_date, end_date)
            if df is not None and minervini_check(df):
                res = vcp_math_check(df)
                if res:
                    local_results.append(res)
                    print(f"✅ {sid} 通過 (累計 {len(local_results)})")

            if idx % 100 == 0:
                print(f"📊 進度：{idx}/{total}，已發現 {len(local_results)} 檔")

            # 速率控制：每小時上限 500 次 => 間隔 7.5 秒
            elapsed = time.time() - loop_start
            time.sleep(max(0, 7.5 - elapsed))

        with scan_lock:
            scan_results = local_results

        # 掃描完成，自動發送通知並儲存報告
        report_msg = build_report(total, scan_results)
        last_report_msg = report_msg
        send_telegram_msg(report_msg)

    except Exception as e:
        print(f"掃描失敗：{e}")
    finally:
        with scan_lock:
            is_scanning = False

# ========== API 端點 ==========
@app.get("/start_scan")
def start_scan():
    """由 GitHub Actions 於 23:00 呼叫，啟動背景掃描"""
    global is_scanning
    if is_scanning:
        return {"status": "already scanning"}
    thread = threading.Thread(target=background_scanner)
    thread.start()
    return {"status": "scan started"}

@app.get("/send_report")
def send_report():
    """由 GitHub Actions 於 07:30 呼叫，發送報告並清空結果"""
    global scan_results, last_report_msg
    total = len(get_filtered_stock_ids())
    msg = build_report(total, scan_results)
    last_report_msg = msg
    send_telegram_msg(msg)
    scan_results.clear()
    return {"status": "report sent"}

@app.get("/latest_report")
def latest_report():
    """提供前端取得最近一次報告內容"""
    global last_report_msg
    return {"report": last_report_msg}

@app.get("/scan")
def scan():
    """立即觸發同步掃描（供前端手動掃描使用）"""
    try:
        stocks = get_filtered_stock_ids()
        total = len(stocks)
        start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")
        layer1 = []
        layer2 = []
        for sid in stocks:
            df = fetch_daily(sid, start_date, end_date)
            if df is not None and minervini_check(df):
                layer1.append(sid)
                res = vcp_math_check(df)
                if res:
                    res["symbol"] = sid
                    layer2.append(res)
        layer2.sort(key=lambda x: -x["rs_score"])
        return {
            "total": total,
            "layer1": len(layer1),
            "layer2": len(layer2),
            "candidates": layer2[:10]
        }
    except Exception as e:
        return {"error": str(e), "total": 0, "layer1": 0, "layer2": 0, "candidates": []}

@app.get("/health")
def health():
    return {"status": "ok", "scanning": is_scanning}
