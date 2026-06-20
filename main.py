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

scan_results = []
is_scanning = False
scan_lock = threading.Lock()
last_report_msg = "尚無報告"

# 全域 DataLoader 實例（節省資源）
_api_instance = None
def get_api():
    global _api_instance
    if _api_instance is None:
        _api_instance = DataLoader()
        _api_instance.login_by_token(FINMIND_TOKEN)
    return _api_instance

# ========== Telegram ==========
def send_telegram_msg(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"Telegram 發送失敗：{e}")

# ========== 股票清單 ==========
_stock_ids_cache = {"ids": [], "ts": 0}

def get_filtered_stock_ids():
    now = time.time()
    if _stock_ids_cache["ids"] and (now - _stock_ids_cache["ts"]) < 86400:
        return _stock_ids_cache["ids"]
    api = get_api()
    info = api.taiwan_stock_info()
    if info is None or info.empty:
        return []
    info = info[~info["stock_name"].str.contains("權|ETF|存託憑證", na=False)]
    info = info[info["stock_id"].str.len() == 4]
    ids = info["stock_id"].unique().tolist()
    _stock_ids_cache["ids"] = ids
    _stock_ids_cache["ts"] = now
    return ids

# ========== 資料下載 ==========
def fetch_daily(sid, start_date, end_date):
    api = get_api()
    try:
        data = api.taiwan_stock_daily(stock_id=sid, start_date=start_date, end_date=end_date)
        if data is None or data.empty:
            return None
        data["date"] = pd.to_datetime(data["date"])
        data.sort_values("date", inplace=True)
        data.set_index("date", inplace=True)
        return data
    except Exception as e:
        print(f"  {sid} 下載失敗：{e}")
        return None

# ========== 篩選邏輯（與之前相同，此處簡略） ==========
def minervini_check(data): ...  # 與之前完全一致，省略以節省篇幅
def vcp_math_check(data): ...  # 與之前完全一致

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

# ========== 背景掃描（夜間用） ==========
def background_scanner():
    global scan_results, is_scanning, last_report_msg
    with scan_lock:
        if is_scanning: return
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
                if res: local_results.append(res)
            if idx % 100 == 0:
                print(f"📊 背景掃描進度：{idx}/{total}")
            elapsed = time.time() - loop_start
            time.sleep(max(0, 7.5 - elapsed))
        with scan_lock: scan_results = local_results
        last_report_msg = build_report(total, scan_results)
        send_telegram_msg(last_report_msg)
    except Exception as e:
        print(f"背景掃描失敗：{e}")
    finally:
        with scan_lock: is_scanning = False

# ========== 非同步手動掃描 ==========
_manual_scan_status = {"running": False, "total": 0, "done": 0, "results": []}

def manual_scanner():
    global _manual_scan_status
    _manual_scan_status["running"] = True
    _manual_scan_status["done"] = 0
    _manual_scan_status["results"] = []
    stocks = get_filtered_stock_ids()
    total = len(stocks)
    _manual_scan_status["total"] = total
    start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    for sid in stocks:
        df = fetch_daily(sid, start_date, end_date)
        if df is not None and minervini_check(df):
            res = vcp_math_check(df)
            if res:
                res["symbol"] = sid
                _manual_scan_status["results"].append(res)
        _manual_scan_status["done"] += 1
        time.sleep(7.5)  # 控制速率
    _manual_scan_status["running"] = False

@app.get("/start_scan_async")
def start_scan_async():
    global _manual_scan_status
    if _manual_scan_status["running"]:
        return {"status": "already running"}
    thread = threading.Thread(target=manual_scanner)
    thread.start()
    return {"status": "started"}

@app.get("/scan_status")
def scan_status():
    return {
        "running": _manual_scan_status["running"],
        "total": _manual_scan_status["total"],
        "done": _manual_scan_status["done"],
        "candidates": _manual_scan_status["results"] if not _manual_scan_status["running"] else []
    }

# ========== 其他端點 ==========
@app.get("/start_scan")
def start_scan():
    return start_scan_async()  # 統一手動與夜間觸發

@app.get("/send_report")
def send_report():
    global scan_results, last_report_msg
    total = len(get_filtered_stock_ids())
    msg = build_report(total, scan_results)
    last_report_msg = msg
    send_telegram_msg(msg)
    scan_results.clear()
    return {"status": "report sent"}

@app.get("/latest_report")
def latest_report():
    return {"report": last_report_msg}

@app.get("/health")
def health():
    return {"status": "ok", "scanning": is_scanning or _manual_scan_status["running"]}
