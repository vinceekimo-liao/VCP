import os
import time
import threading
from datetime import datetime, timedelta
from collections import deque

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

# ========== 環境變數 ==========
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ========== 全域變數 ==========
scan_results = []
any_scan_running = False
scan_lock = threading.Lock()
last_report_msg = "尚無報告"

# 請求頻率控制 ── 滑動窗口 (每小時 500 次) + 最小間隔 7 秒
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

def send_telegram_msg(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ 缺少 Telegram 設定")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"Telegram 發送失敗：{e}")

def convert_numpy(obj):
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

# 股票清單快取 (含重試)
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

        rs_lookback = min(60, len(close))
        past_close = close.iloc[-rs_lookback]
        if past_close <= 0:
            return None
        rs_raw = 50 + (close.iloc[-1] - past_close) / past_close * 200
        rs = int(max(1, min(99, round(float(rs_raw)))))

        # ── 進一步收緊的過濾條件 ──
        if rs < 70:
            return None

        cond1 = (contractions >= 2) and (vol_ratio >= 1.1)
        cond2 = (contractions >= 1) and (vol_ratio >= 1.4)
        cond3 = (today_change > 2.5) and (vol_ratio > 1.5)
        cond4 = (contractions >= 6) and (vol_ratio >= 0.8) and (rs >= 95)
        cond5 = (contractions >= 4) and (vol_ratio >= 0.9) and (rs >= 97)

        if not (cond1 or cond2 or cond3 or cond4 or cond5):
            return None

        qs = 0
        if contractions >= 3: qs += 1
        if vol_ratio >= 1.3: qs += 1
        if rs >= 85: qs += 1
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

# ========== 除錯版函數（完整恢復） ==========
def minervini_check_with_debug(data):
    debug = {"passed": False, "reason": ""}
    if data is None or len(data) < 200:
        debug["reason"] = f"資料筆數不足：{len(data) if data is not None else 'None'}"
        return debug
    close = _get_col(data, "close", "Close")
    high  = _get_col(data, "max", "high", "High")
    if close is None or high is None:
        debug["reason"] = f"缺少欄位：close={close is not None}, high/max={high is not None}"
        return debug
    close_clean = pd.to_numeric(close, errors='coerce').dropna()
    high_clean  = pd.to_numeric(high,  errors='coerce').dropna()
    if len(close_clean) < 200 or len(high_clean) < 200:
        debug["reason"] = f"去除 NaN 後資料不足：close {len(close_clean)}, high/max {len(high_clean)}"
        return debug
    try:
        ma50  = close_clean.rolling(50).mean()
        ma150 = close_clean.rolling(150).mean()
        ma200 = close_clean.rolling(200).mean()
        last  = close_clean.iloc[-1]
        debug["last"] = round(last, 2)
        debug["ma150"] = round(ma150.iloc[-1], 2) if not pd.isna(ma150.iloc[-1]) else "NaN"
        debug["ma200"] = round(ma200.iloc[-1], 2) if not pd.isna(ma200.iloc[-1]) else "NaN"
        cond_ma = (last > ma150.iloc[-1]) or (last > ma200.iloc[-1])
        debug["cond_ma"] = cond_ma
        if not cond_ma:
            debug["reason"] = "收盤價未大於 MA150 或 MA200"
            return debug
        if len(high_clean) >= 200:
            high_52w = high_clean.rolling(250, min_periods=1).max().iloc[-1]
            debug["high_52w"] = round(high_52w, 2) if pd.notna(high_52w) else "NaN"
            if pd.notna(high_52w):
                debug["high_52w_65pct"] = round(high_52w * 0.65, 2)
                if last < high_52w * 0.65:
                    debug["reason"] = f"距 52 週高點太遠：現價 {last} < {round(high_52w*0.65,2)}"
                    return debug
        debug["passed"] = True
        return debug
    except Exception as e:
        debug["reason"] = f"計算錯誤：{str(e)}"
        return debug

def vcp_math_check_with_debug(data):
    debug = {"passed": False, "reason": ""}
    if data is None or len(data) < 60:
        debug["reason"] = "資料筆數不足 60"
        return debug
    close  = _get_col(data, "close", "Close")
    volume = _get_col(data, "Trading_Volume", "volume", "Volume")
    high   = _get_col(data, "max", "high", "High")
    low    = _get_col(data, "min", "low", "Low")
    if close is None or volume is None or high is None or low is None:
        debug["reason"] = "缺少必要欄位"
        return debug
    close  = pd.to_numeric(close, errors='coerce').dropna()
    high   = pd.to_numeric(high,  errors='coerce').dropna()
    low    = pd.to_numeric(low,   errors='coerce').dropna()
    volume = pd.to_numeric(volume, errors='coerce').dropna()
    if len(close) < 60 or len(volume) < 60:
        debug["reason"] = f"有效資料不足：close {len(close)}, volume {len(volume)}"
        return debug
    try:
        vol_ma_20 = volume.rolling(20).mean()
        recent_vol = volume.iloc[-3:].mean()
        if pd.isna(vol_ma_20.iloc[-1]) or vol_ma_20.iloc[-1] == 0:
            debug["reason"] = "vol_ma_20 為 NaN 或 0"
            return debug
        vol_ratio = recent_vol / vol_ma_20.iloc[-1]
        debug["vol_ratio"] = round(float(vol_ratio), 2)

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
        debug["contractions"] = contractions

        today_change = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) >= 2 else 0
        debug["today_change"] = round(today_change, 2)

        rs_lookback = min(60, len(close))
        past_close = close.iloc[-rs_lookback]
        if past_close <= 0:
            debug["reason"] = "歷史收盤價無效"
            return debug
        rs_raw = 50 + (close.iloc[-1] - past_close) / past_close * 200
        rs = int(max(1, min(99, round(float(rs_raw)))))
        debug["rs"] = rs

        if rs < 70:
            debug["reason"] = f"RS < 70 (實際 {rs})"
            return debug

        cond1 = (contractions >= 2) and (vol_ratio >= 1.1)
        cond2 = (contractions >= 1) and (vol_ratio >= 1.4)
        cond3 = (today_change > 2.5) and (vol_ratio > 1.5)
        cond4 = (contractions >= 6) and (vol_ratio >= 0.8) and (rs >= 95)
        cond5 = (contractions >= 4) and (vol_ratio >= 0.9) and (rs >= 97)
        passed = cond1 or cond2 or cond3 or cond4 or cond5
        debug["passed_vcp"] = passed
        if not passed:
            debug["reason"] = f"未滿足任一條件 (c1:{cond1}, c2:{cond2}, c3:{cond3}, c4:{cond4}, c5:{cond5})"
            return debug
        debug["passed"] = True
        return debug
    except Exception as e:
        debug["reason"] = f"計算錯誤：{str(e)}"
        return debug

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
    global _manual_scan_status, scan_results
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
    print(f"✅ 手動掃描完成，第一層通過：{layer1_pass} 檔，最終候選：{len(scan_results)} 檔")

# ========== 夜間背景掃描 ==========
def background_scanner():
    global scan_results, last_report_msg, _manual_scan_status
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
    print(f"✅ 背景掃描完成，第一層通過：{layer1_pass} 檔，最終候選：{len(scan_results)} 檔")

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
    if _manual_scan_status["running"]:
        return {
            "running": True,
            "total": _manual_scan_status["total"],
            "done": _manual_scan_status["done"],
            "candidates": []
        }
    if _manual_scan_status["results"]:
        return {
            "running": False,
            "total": _manual_scan_status["total"],
            "done": _manual_scan_status["done"],
            "candidates": _manual_scan_status["results"]
        }
    with scan_lock:
        if scan_results:
            return {
                "running": False,
                "total": len(get_filtered_stock_ids()),
                "done": len(scan_results),
                "candidates": scan_results
            }
    return {"running": False, "total": 0, "done": 0, "candidates": []}

@app.get("/send_report")
def send_report():
    global scan_results, last_report_msg
    total = len(get_filtered_stock_ids())
    if _manual_scan_status["results"]:
        msg = build_report(total, _manual_scan_status["results"])
    else:
        msg = build_report(total, scan_results)
    last_report_msg = msg
    send_telegram_msg(msg)
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
    result = {"symbol": symbol, "step1_fetch": None, "step2_minervini": None, "step3_vcp": None}
    start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    df = fetch_daily(symbol, start_date, end_date)
    if df is None:
        result["step1_fetch"] = "下載失敗"
        return convert_numpy(result)
    result["step1_fetch"] = {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "tail_close": df["close"].tail(5).tolist() if "close" in df.columns else "無 close",
        "tail_max": df["max"].tail(5).tolist() if "max" in df.columns else "無 max",
    }
    result["step2_minervini"] = minervini_check_with_debug(df)
    if result["step2_minervini"].get("passed"):
        result["step3_vcp"] = vcp_math_check_with_debug(df)
    else:
        result["step3_vcp"] = "未執行（Minervini 未通過）"
    return convert_numpy(result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)