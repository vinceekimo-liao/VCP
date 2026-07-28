import os
import time
import json
import threading
from datetime import datetime, timedelta
from collections import deque

import pandas as pd
import numpy as np
import requests
import resend
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from FinMind.data import DataLoader

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 環境變數 ==========
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", "")

# 黑名單檔案
BLACKLIST_FILE = "blacklist.json"
if os.path.exists(BLACKLIST_FILE):
    with open(BLACKLIST_FILE, "r") as f:
        blacklist = json.load(f)
else:
    blacklist = {}

def save_blacklist():
    with open(BLACKLIST_FILE, "w") as f:
        json.dump(blacklist, f)

# ========== 全域變數 ==========
scan_results = []
trade_signals = []
any_scan_running = False
scan_lock = threading.Lock()
last_report_msg = "尚無報告"

# 請求頻率控制
_request_times = deque()
REQUEST_LIMIT = 500
REQUEST_WINDOW = 3600
MIN_INTERVAL = 7.0
_request_lock = threading.Lock()

_api_instance = None
def get_api():
    global _api_instance
    if _api_instance is None:
        _api_instance = DataLoader()
        _api_instance.login_by_token(FINMIND_TOKEN)
    return _api_instance

# ========== 輔助函數 ==========
def send_telegram_msg(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ 缺少 Telegram 設定")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"Telegram 發送失敗：{e}")

def send_email_report(results, total_scanned):
    if not RESEND_API_KEY:
        print("⚠️ 未設定 RESEND_API_KEY，跳過 Email 寄送")
        return
    if not EMAIL_SENDER or not EMAIL_RECEIVER:
        print("⚠️ 缺少 EMAIL_SENDER 或 EMAIL_RECEIVER，跳過寄送")
        return
    if not results:
        print("⚠️ 無候選股票，不寄送 Email")
        return

    resend.api_key = RESEND_API_KEY

    try:
        df = pd.DataFrame(results)
        df = df.sort_values("rs_score", ascending=False)
        df = df.rename(columns={
            "symbol": "代號", "price": "股價", "change_pct": "漲跌幅%",
            "rs_score": "RS", "contractions": "收縮次數",
            "volume_ratio": "量比", "quality": "品質",
            "buy_signal": "進場訊號"
        })

        df_buy = df[df["進場訊號"] == True].copy()

        filename = f"VCP_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="全部漏斗候選", index=False)
            df_buy.to_excel(writer, sheet_name="最終進場訊號", index=False)

        print(f"📁 Excel 檔案已產生：{filename} (全部 {len(df)} 檔，進場 {len(df_buy)} 檔)")

        with open(filename, "rb") as f:
            file_content = f.read()

        attachment = {
            "filename": filename,
            "content": list(file_content)
        }

        params = {
            "from": f"VCP 掃描器 <{EMAIL_SENDER}>",
            "to": [EMAIL_RECEIVER],
            "subject": f"📈 VCP 每日報告 {datetime.now().strftime('%Y-%m-%d')}",
            "html": f"<p>今日共掃描 {total_scanned} 檔，符合漏斗條件 {len(df)} 檔，其中最終進場訊號 {len(df_buy)} 檔。<br>詳細請見附件。</p>",
            "attachments": [attachment]
        }
        response = resend.Emails.send(params)
        print(f"✅ Email 已寄送至 {EMAIL_RECEIVER}，Resend ID: {response['id']}")

    except Exception as e:
        print(f"❌ Email 寄送失敗：{type(e).__name__} - {str(e)}")

def convert_numpy(obj):
    """遞迴將 numpy 型別轉為 Python 原生型別，確保 JSON 可序列化"""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj

# ========== 無死鎖限流 ==========
def _wait_for_slot():
    while True:
        with _request_lock:
            now = time.time()
            while _request_times and now - _request_times[0] > REQUEST_WINDOW:
                _request_times.popleft()
            if len(_request_times) < REQUEST_LIMIT:
                if not _request_times or (now - _request_times[-1] >= MIN_INTERVAL):
                    _request_times.append(now)
                    return
                else:
                    wait = MIN_INTERVAL - (now - _request_times[-1])
            else:
                oldest = _request_times[0]
                wait = oldest + REQUEST_WINDOW - now + 0.1
        time.sleep(wait)

# ========== 股票清單、下載 ==========
_stock_ids_cache = {"ids": [], "ts": 0}
def get_filtered_stock_ids():
    now = time.time()
    if _stock_ids_cache["ids"] and (now - _stock_ids_cache["ts"]) < 86400:
        return _stock_ids_cache["ids"]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            _wait_for_slot()
            api = get_api()
            info = api.taiwan_stock_info()
            if info is None or info.empty:
                raise ValueError("回傳空資料")
            info = info[~info["stock_name"].str.contains("權|ETF|存託憑證", na=False)]
            info = info[info["stock_id"].str.len() == 4]
            ids = info["stock_id"].unique().tolist()
            _stock_ids_cache["ids"] = ids
            _stock_ids_cache["ts"] = now
            print(f"📋 普通股代號數量：{len(ids)}")
            return ids
        except Exception as e:
            print(f"❌ 取得股票清單失敗 (嘗試 {attempt+1}/{max_retries})：{e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print("🚨 無法取得股票清單，掃描終止")
                return []
    return []

def fetch_daily(sid, start_date, end_date):
    _wait_for_slot()
    api = get_api()
    try:
        data = api.taiwan_stock_daily(stock_id=sid, start_date=start_date, end_date=end_date)
        if data is None or data.empty:
            print(f"  {sid} 回傳空資料")
            return None
        data["date"] = pd.to_datetime(data["date"])
        data.sort_values("date", inplace=True)
        data.set_index("date", inplace=True)
        return data
    except Exception as e:
        print(f"  {sid} 下載失敗：{str(e)[:100]}")
        return None

def _get_col(data, *names):
    for n in names:
        if n in data.columns:
            return data[n]
    return None

# ========== 加權指數檢查 ==========
def get_market_status():
    try:
        df = fetch_daily("TAIEX", (datetime.today() - timedelta(days=200)).strftime("%Y-%m-%d"),
                         datetime.today().strftime("%Y-%m-%d"))
        if df is None or len(df) < 20:
            return None, None
        close = df["close"]
        ma20 = close.rolling(20).mean()
        last = close.iloc[-1]
        return bool(last > ma20.iloc[-1]), round(float(last), 2)
    except Exception as e:
        print(f"加權指數檢查失敗：{e}")
        return None, None

# ========== 第一層：Minervini 完整 Stage 2 趨勢範本 ==========
def minervini_check(data):
    if data is None or len(data) < 250:
        return False
        
    close = _get_col(data, "close", "Close")
    high  = _get_col(data, "max", "high", "High")
    low   = _get_col(data, "min", "low", "Low")
    
    if close is None or high is None or low is None:
        return False
        
    close = pd.to_numeric(close, errors='coerce').dropna()
    high  = pd.to_numeric(high,  errors='coerce').dropna()
    low   = pd.to_numeric(low,   errors='coerce').dropna()
    
    if len(close) < 250 or len(high) < 250 or len(low) < 250:
        return False

    try:
        ma50  = close.rolling(50).mean()
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()
        
        c_last = close.iloc[-1]
        m50_last  = ma50.iloc[-1]
        m150_last = ma150.iloc[-1]
        m200_last = ma200.iloc[-1]
        
        # 1. 均線多頭排列
        if not (c_last > m50_last and m50_last > m150_last and m150_last > m200_last):
            return False
            
        # 2. 200MA 向上
        if m200_last <= ma200.iloc[-20]:
            return False
            
        # 3. 52 週高低點
        high_52w = high.tail(250).max()
        low_52w  = low.tail(250).min()
        
        if c_last < low_52w * 1.30:
            return False
            
        if c_last < high_52w * 0.75:
            return False
            
        return True
    except Exception as e:
        return False

# ========== 第二層：VCP 籌碼乾涸與動能爆量檢查 ==========
def vcp_math_check(data):
    if data is None or len(data) < 120:
        return None

    close  = pd.to_numeric(_get_col(data, "close", "Close"), errors='coerce')
    high   = pd.to_numeric(_get_col(data, "max", "high", "High"), errors='coerce')
    low    = pd.to_numeric(_get_col(data, "min", "low", "Low"), errors='coerce')
    volume = pd.to_numeric(_get_col(data, "Trading_Volume", "volume", "Volume"), errors='coerce')

    df_clean = pd.DataFrame({"close": close, "high": high, "low": low, "volume": volume}).dropna()
    if len(df_clean) < 120:
        return None

    c = df_clean["close"]
    h = df_clean["high"]
    l = df_clean["low"]
    v = df_clean["volume"]

    try:
        # 1. 量比
        vol_ma20 = v.rolling(20).mean()
        if pd.isna(vol_ma20.iloc[-1]) or vol_ma20.iloc[-1] == 0:
            return None
        vol_ratio = v.iloc[-1] / vol_ma20.iloc[-1]

        # 2. VDU
        vdu_flag = v.iloc[-4:-1].mean() < (vol_ma20.iloc[-1] * 0.85)

        # 3. VCP 收縮
        range_t1 = (h.iloc[-40:-20].max() - l.iloc[-40:-20].min()) / l.iloc[-40:-20].min()
        range_t2 = (h.iloc[-20:-1].max() - l.iloc[-20:-1].min()) / l.iloc[-20:-1].min()
        vcp_contracting = range_t1 > range_t2

        contractions = 3 if (vcp_contracting and vdu_flag) else (2 if vcp_contracting else 1)

        # 4. 樞紐突破
        today_change = ((c.iloc[-1] - c.iloc[-2]) / c.iloc[-2] * 100) if len(c) >= 2 else 0
        pivot_high = h.iloc[-20:-1].max()
        is_pivot_breakout = c.iloc[-1] >= pivot_high * 0.99

        # 5. RS
        rs_lookback = min(60, len(c))
        past_close = c.iloc[-rs_lookback]
        if past_close <= 0:
            return None
        rs_raw = 50 + (c.iloc[-1] - past_close) / past_close * 200
        rs = int(max(1, min(99, round(float(rs_raw)))))

        if rs < 85 or vol_ratio < 1.5:
            return None

        # 6. 品質評分
        qs = 0
        if 1.8 <= vol_ratio <= 3.0:
            qs += 2
        elif vol_ratio > 3.0:
            qs += 1
        if vdu_flag:
            qs += 1
        if vcp_contracting:
            qs += 1
        if rs >= 95:
            qs += 1

        quality = "A" if qs >= 4 else "B" if qs >= 2 else "C"

        # 7. 進場訊號
        ma50  = c.rolling(50).mean().iloc[-1]
        ma150 = c.rolling(150).mean().iloc[-1]
        ma200 = c.rolling(200).mean().iloc[-1]
        last  = c.iloc[-1]

        buy_signal = (
            is_pivot_breakout and
            vol_ratio >= 1.5 and
            last > ma50 > ma150 > ma200
        )

        return {
            "symbol": str(data["stock_id"].iloc[0]) if "stock_id" in data.columns else "",
            "price": round(float(last), 2),
            "change_pct": round(float(today_change), 2),
            "rs_score": rs,
            "contractions": contractions,
            "volume_ratio": round(float(vol_ratio), 2),
            "vdu": vdu_flag,
            "quality": quality,
            "buy_signal": bool(buy_signal),
            "consecutive_days": 0,
        }
    except Exception as e:
        print(f"  VCP error: {e}")
        return None

# ========== 第三層：大盤趨勢與風控交易過濾器 ==========
def apply_trade_filters(candidates):
    market_bull, market_price = get_market_status()
    filtered = []

    for c in candidates:
        symbol = c.get("symbol", "")
        if symbol in blacklist and blacklist[symbol] == -1:
            continue

        vol_ratio   = c.get("volume_ratio", 0)
        rs_score    = c.get("rs_score", 0)
        quality     = c.get("quality", "C")
        buy_signal  = c.get("buy_signal", False)
        consecutive = c.get("consecutive_days", 0)

        if not buy_signal or rs_score < 85:
            continue

        c["market_bull"] = market_bull
        c["market_price"] = market_price

        # 空頭防禦
        if not market_bull:
            if quality == "A" and vol_ratio >= 2.0 and rs_score >= 90:
                c["filter_note"] = "空頭防禦性試探部位"
                filtered.append(c)
            continue

        # 多頭情境
        if consecutive >= 2 and vol_ratio >= 1.3:
            c["filter_note"] = "動能持續加碼股"
            filtered.append(c)
            continue

        if 1.5 <= vol_ratio <= 3.0:
            if quality in ["A", "B"] and rs_score >= 85:
                c["filter_note"] = "黃金量比突破"
                filtered.append(c)
                continue

        if vol_ratio > 3.0:
            if rs_score >= 92 and quality in ["A", "B"]:
                c["filter_note"] = "天量動能突破"
                filtered.append(c)
                continue

    return filtered

# ========== 掃描執行器 ==========
def _run_scan(scanner_func):
    global any_scan_running
    with scan_lock:
        if any_scan_running:
            print("⚠️ 已有掃描在執行中，略過本次觸發")
            return
        any_scan_running = True
    try:
        scanner_func()
    except Exception as e:
        print(f"💥 掃描線程崩潰：{e}")
    finally:
        with scan_lock:
            any_scan_running = False

# ========== 手動掃描 ==========
_manual_scan_status = {"running": False, "total": 0, "done": 0, "results": []}

def manual_scanner():
    global _manual_scan_status, scan_results, trade_signals
    _manual_scan_status["running"] = True
    _manual_scan_status["done"] = 0
    _manual_scan_status["results"] = []
    stocks = get_filtered_stock_ids()
    if not stocks:
        _manual_scan_status["running"] = False
        print("❌ 無股票清單，掃描終止")
        return
    total = len(stocks)
    _manual_scan_status["total"] = total
    start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    layer1_pass = 0
    for idx, sid in enumerate(stocks, 1):
        df = fetch_daily(sid, start_date, end_date)
        if df is not None and minervini_check(df):
            layer1_pass += 1
            res = vcp_math_check(df)
            if res:
                res["symbol"] = sid
                _manual_scan_status["results"].append(res)
        _manual_scan_status["done"] = idx
        if idx % 100 == 0:
            print(f"📊 進度：{idx}/{total}，第一層通過：{layer1_pass}，候選：{len(_manual_scan_status['results'])}")
    _manual_scan_status["running"] = False
    with scan_lock:
        scan_results = _manual_scan_status["results"]
    trade_signals = apply_trade_filters(scan_results)
    send_email_report(scan_results, total)
    print(f"✅ 手動掃描完成，第一層通過：{layer1_pass} 檔，最終候選：{len(scan_results)} 檔，交易訊號：{len(trade_signals)} 檔")

# ========== 夜間背景掃描 ==========
def background_scanner():
    global scan_results, last_report_msg, _manual_scan_status, trade_signals
    stocks = get_filtered_stock_ids()
    if not stocks:
        print("❌ 無股票清單，夜間掃描終止")
        return
    total = len(stocks)
    start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    local_results = []
    layer1_pass = 0
    for idx, sid in enumerate(stocks, 1):
        df = fetch_daily(sid, start_date, end_date)
        if df is not None and minervini_check(df):
            layer1_pass += 1
            res = vcp_math_check(df)
            if res:
                local_results.append(res)
        if idx % 100 == 0:
            print(f"📊 背景掃描進度：{idx}/{total}，第一層通過：{layer1_pass}，候選：{len(local_results)}")
    with scan_lock:
        scan_results = local_results
    _manual_scan_status["running"] = False
    _manual_scan_status["total"] = total
    _manual_scan_status["done"] = total
    _manual_scan_status["results"] = local_results
    last_report_msg = build_report(total, scan_results)
    send_telegram_msg(last_report_msg)
    trade_signals = apply_trade_filters(scan_results)
    send_email_report(scan_results, total)
    print(f"✅ 背景掃描完成，第一層通過：{layer1_pass} 檔，最終候選：{len(scan_results)} 檔，交易訊號：{len(trade_signals)} 檔")

def build_report(total, results):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    if not results:
        return f"📉 <b>每日 VCP 報告 ({now_str})</b>\n掃描 {total} 檔，無符合條件股票"
    sorted_results = sorted(results, key=lambda x: -x["rs_score"])
    msg = f"📈 <b>每日 VCP 報告 ({now_str})</b>\n掃描 {total} 檔，符合 {len(results)} 檔\n\n"
    for i, c in enumerate(sorted_results[:15], 1):
        symbol = c['symbol']
        yahoo_link = f"https://tw.stock.yahoo.com/quote/{symbol}"
        msg += f"🔹 <b>{symbol}</b> | 價:{c['price']} | RS:{c['rs_score']} | 品質:{c['quality']} <a href='{yahoo_link}'>📈 Yahoo</a>\n"
    return msg

# ========== API 端點 ==========
@app.get("/start_scan_async")
def start_scan_async():
    global any_scan_running
    if any_scan_running:
        return {"status": "already running"}
    thread = threading.Thread(target=_run_scan, args=(manual_scanner,))
    thread.start()
    return {"status": "started"}

@app.get("/start_scan")
def start_scan():
    global any_scan_running
    if any_scan_running:
        return {"status": "already running"}
    thread = threading.Thread(target=_run_scan, args=(background_scanner,))
    thread.start()
    return {"status": "started"}

@app.get("/scan_status")
def scan_status():
    try:
        if _manual_scan_status["running"]:
            return convert_numpy({
                "running": True,
                "total": _manual_scan_status["total"],
                "done": _manual_scan_status["done"],
                "candidates": []
            })
        if _manual_scan_status["results"]:
            return convert_numpy({
                "running": False,
                "total": _manual_scan_status["total"],
                "done": _manual_scan_status["done"],
                "candidates": _manual_scan_status["results"]
            })
        with scan_lock:
            if scan_results:
                return convert_numpy({
                    "running": False,
                    "total": len(get_filtered_stock_ids()),
                    "done": len(scan_results),
                    "candidates": scan_results
                })
        return {"running": False, "total": 0, "done": 0, "candidates": []}
    except Exception as e:
        return {"error": str(e), "running": False, "total": 0, "done": 0, "candidates": []}

@app.get("/send_report")
def send_report():
    global scan_results, last_report_msg
    total = len(get_filtered_stock_ids())
    if _manual_scan_status["results"]:
        msg = build_report(total, _manual_scan_status["results"])
        current = _manual_scan_status["results"]
    else:
        msg = build_report(total, scan_results)
        current = scan_results
    last_report_msg = msg
    send_telegram_msg(msg)
    send_email_report(current, total)
    return {"status": "report sent"}

@app.get("/latest_report")
def latest_report():
    global last_report_msg
    return {"report": last_report_msg}

@app.get("/health")
def health():
    with _request_lock:
        pending = len(_request_times)
    return {"status": "ok", "scanning": any_scan_running, "requests_last_hour": pending}

@app.get("/debug_scan")
def debug_scan(symbol: str = "3008"):
    return {"status": "ok"}

@app.get("/full_report", response_class=HTMLResponse)
def full_report():
    results = _manual_scan_status["results"] if _manual_scan_status["results"] else scan_results
    total = _manual_scan_status["total"] if _manual_scan_status["total"] else len(get_filtered_stock_ids())
    if not results:
        return HTMLResponse(content="<html><body><h2>尚無篩選結果</h2></body></html>")
    sorted_results = sorted(results, key=lambda x: -x["rs_score"])
    html = f"""<html><head><meta charset='utf-8'><title>VCP 完整報告</title>
    <style>
        body {{ background: #060d16; color: #e2f0ff; font-family: sans-serif; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
        th {{ background: #1a2d40; padding: 6px; text-align: left; }}
        td {{ padding: 6px; border-bottom: 1px solid #1a2d40; }}
        a {{ color: #38bdf8; }}
    </style></head><body>
    <h2>📈 VCP 完整篩選報告</h2>
    <p>掃描 {total} 檔，符合 {len(sorted_results)} 檔</p>
    <table><tr><th>代號</th><th>股價</th><th>漲跌%</th><th>RS</th><th>收縮次數</th><th>量比</th><th>品質</th><th>進場訊號</th><th>Yahoo</th></tr>"""
    for c in sorted_results:
        buy_icon = "✅" if c.get("buy_signal") else "❌"
        html += f"<tr><td>{c['symbol']}</td><td>{c['price']}</td><td>{c['change_pct']:+.2f}%</td><td>{c['rs_score']}</td><td>{c['contractions']}</td><td>{c['volume_ratio']}</td><td>{c['quality']}</td><td>{buy_icon}</td><td><a href='https://tw.stock.yahoo.com/quote/{c['symbol']}' target='_blank'>Yahoo</a></td></tr>"
    html += "</table></body></html>"
    return HTMLResponse(content=html)

@app.get("/final_candidates")
def final_candidates():
    results = _manual_scan_status["results"] if _manual_scan_status["results"] else scan_results
    if not results:
        return []
    return convert_numpy([c for c in results if c.get("buy_signal")])

@app.get("/trade_signals")
def get_trade_signals():
    market_bull, market_price = get_market_status()
    return convert_numpy({
        "market_bull": market_bull,
        "market_price": market_price,
        "suggestion": "暫停進場，或只投 1/3 資金" if not market_bull else "可正常進場",
        "candidates": trade_signals,
        "entry_tips": [
            "⚠️ 開盤觀察 (9:00-10:00)：若開盤即跌破篩選日最低價 → 刪除",
            "⚠️ 若開盤跳空大漲 (>5%) → 觀望不追（可能已過高點）",
            "✅ 若開盤平穩或小幅上漲 → 10:00 後進場",
            "✅ 進場價：不超過篩選日收盤價 +2%",
            "✅ 倉位：單檔不超過 1/5 資金"
        ],
        "exit_tips": [
            "停利：+5% 或持有 3 天後評估",
            "停損：-3% 或跌破篩選日最低價",
            "時間停損：5 天未達 +3% 即出場"
        ]
    })

@app.get("/blacklist")
def get_blacklist():
    return blacklist

@app.post("/blacklist/add")
def add_blacklist(symbol: str, status: int = -1):
    blacklist[symbol] = status
    save_blacklist()
    return {"message": "ok"}

@app.get("/market_status")
def market_status():
    bull, price = get_market_status()
    return {"bull": bull, "price": price}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
