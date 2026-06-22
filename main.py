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

# ========== 環境變數 ==========
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ========== 全域變數 ==========
scan_results = []          # 夜間掃描結果（給 /send_report 使用）
is_scanning = False        # 背景掃描是否進行中（避免重複觸發）
scan_lock = threading.Lock()
last_report_msg = "尚無報告"

# 全域 DataLoader（重用，節省連線）
_api_instance = None
def get_api():
    global _api_instance
    if _api_instance is None:
        _api_instance = DataLoader()
        _api_instance.login_by_token(FINMIND_TOKEN)
    return _api_instance

# ========== Telegram 通知 ==========
def send_telegram_msg(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ 缺少 Telegram 設定")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"Telegram 發送失敗：{e}")

# ========== Numpy 轉換工具（避免 JSON 序列化錯誤） ==========
def convert_numpy(obj):
    """將 numpy 型別遞迴轉換為 Python 原生型別，確保可以 JSON 序列化"""
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

# ========== 股票清單（快取 24 小時） ==========
_stock_ids_cache = {"ids": [], "ts": 0}

def get_filtered_stock_ids():
    now = time.time()
    if _stock_ids_cache["ids"] and (now - _stock_ids_cache["ts"]) < 86400:
        return _stock_ids_cache["ids"]

    api = get_api()
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

# ========== 輔助：動態取得欄位 ==========
def _get_col(data, *names):
    """從 DataFrame 中依序嘗試取得欄位，回傳第一個存在的 Series，若皆無則回傳 None"""
    for n in names:
        if n in data.columns:
            return data[n]
    return None

# ========== 第一層：Minervini 趨勢模板（放寬版，欄位修正） ==========
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
        ma50  = close.rolling(50).mean()
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()
        last  = close.iloc[-1]

        # 放寬：收盤 > MA150 或 > MA200（滿足其一即可）
        cond_ma = (last > ma150.iloc[-1]) or (last > ma200.iloc[-1])
        if not cond_ma:
            return False

        # 距 52 週高點放寬至 65%
        if len(high) >= 200:
            high_52w = high.rolling(250, min_periods=1).max().iloc[-1]
            if pd.notna(high_52w) and last < high_52w * 0.65:
                return False

        return True
    except:
        return False

# ========== 第二層：VCP 波動收縮（放寬版，欄位修正） ==========
def vcp_math_check(data):
    if data is None or len(data) < 60:
        return None

    close  = _get_col(data, "close", "Close")
    volume = _get_col(data, "Trading_Volume", "volume", "Volume")
    high   = _get_col(data, "max", "high", "High")
    low    = _get_col(data, "min", "low", "Low")

    if close is None or volume is None or high is None or low is None:
        return None

    close  = pd.to_numeric(close, errors='coerce').dropna()
    high   = pd.to_numeric(high,  errors='coerce').dropna()
    low    = pd.to_numeric(low,   errors='coerce').dropna()
    volume = pd.to_numeric(volume, errors='coerce').dropna()

    if len(close) < 60 or len(volume) < 60:
        return None

    try:
        vol_ma_20 = volume.rolling(20).mean()
        recent_vol = volume.iloc[-3:].mean()
        if pd.isna(vol_ma_20.iloc[-1]) or vol_ma_20.iloc[-1] == 0:
            return None
        vol_ratio = recent_vol / vol_ma_20.iloc[-1]
        # 防止無窮大或 NaN
        if not np.isfinite(vol_ratio):
            return None

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

        # 放寬：只要有一次收縮，或是量比 > 1.2 即可通過
        if contractions == 0 and vol_ratio <= 1.2:
            return None

        rs_lookback = min(60, len(close))
        rs = min(99, max(1, int(50 + (close.iloc[-1] - close.iloc[-rs_lookback]) / close.iloc[-rs_lookback] * 200)))

        qs = (1 if contractions >= 2 else 0) + (1 if vol_ratio >= 1.5 else 0) + (1 if rs >= 70 else 0) + (1 if rs >= 85 else 0)
        quality = "A" if qs >= 3 else "B" if qs >= 1 else "C"

        return {
            "symbol": data["stock_id"].iloc[0] if "stock_id" in data.columns else "",
            "price": round(float(close.iloc[-1]), 2),
            "change_pct": round(float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100), 2),
            "rs_score": rs,
            "contractions": contractions,
            "volume_ratio": round(float(vol_ratio), 2),
            "quality": quality,
        }
    except Exception as e:
        print(f"  VCP error: {e}")
        return None


# ========== 除錯版函數（用於 /debug_scan） ==========
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
        debug["reason"] = f"去除 NaN 後資料不足：close {len(close)}, volume {len(volume)}"
        return debug

    try:
        vol_ma_20 = volume.rolling(20).mean()
        recent_vol = volume.iloc[-3:].mean()
        if pd.isna(vol_ma_20.iloc[-1]) or vol_ma_20.iloc[-1] == 0:
            debug["reason"] = "vol_ma_20 為 NaN 或 0"
            return debug
        vol_ratio = recent_vol / vol_ma_20.iloc[-1]
        if not np.isfinite(vol_ratio):
            debug["reason"] = "vol_ratio 為無限大或 NaN"
            return debug
        debug["vol_ratio"] = round(vol_ratio, 2)

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

        debug["contractions"] = contractions
        if contractions == 0 and vol_ratio <= 1.2:
            debug["reason"] = f"無收縮且量比 {round(vol_ratio,2)} <= 1.2"
            return debug

        debug["passed"] = True
        return debug
    except Exception as e:
        debug["reason"] = f"計算錯誤：{str(e)}"
        return debug

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

# ========== 非同步手動掃描狀態 ==========
_manual_scan_status = {
    "running": False,
    "total": 0,
    "done": 0,
    "results": []
}

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
    for idx, sid in enumerate(stocks, 1):
        loop_start = time.time()
        df = fetch_daily(sid, start_date, end_date)
        if df is not None and minervini_check(df):
            res = vcp_math_check(df)
            if res:
                res["symbol"] = sid
                _manual_scan_status["results"].append(res)
        _manual_scan_status["done"] = idx  # 即時更新已處理筆數
        elapsed = time.time() - loop_start
        time.sleep(max(0, 7.5 - elapsed))
    _manual_scan_status["running"] = False

# ========== 夜間背景掃描（供排程觸發） ==========
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
            if idx % 100 == 0:
                print(f"📊 背景掃描進度：{idx}/{total}")
            elapsed = time.time() - loop_start
            time.sleep(max(0, 7.5 - elapsed))
        with scan_lock:
            scan_results = local_results
        last_report_msg = build_report(total, scan_results)
        send_telegram_msg(last_report_msg)
    except Exception as e:
        print(f"背景掃描失敗：{e}")
    finally:
        with scan_lock:
            is_scanning = False

# ========== API 端點 ==========
@app.get("/start_scan_async")
def start_scan_async():
    """非同步手動掃描觸發"""
    global _manual_scan_status
    if _manual_scan_status["running"]:
        return {"status": "already running"}
    thread = threading.Thread(target=manual_scanner)
    thread.start()
    return {"status": "started"}

@app.get("/start_scan")
def start_scan():
    """保留舊端點，指向非同步掃描"""
    return start_scan_async()

@app.get("/scan_status")
def scan_status():
    """前端輪詢用：回傳即時進度"""
    return {
        "running": _manual_scan_status["running"],
        "total": _manual_scan_status["total"],
        "done": _manual_scan_status["done"],
        "candidates": _manual_scan_status["results"] if not _manual_scan_status["running"] else []
    }

@app.get("/send_report")
def send_report():
    """由排程呼叫，發送夜間掃描 Telegram 報告並清空結果"""
    global scan_results, last_report_msg
    total = len(get_filtered_stock_ids())
    msg = build_report(total, scan_results)
    last_report_msg = msg
    send_telegram_msg(msg)
    scan_results.clear()
    return {"status": "report sent"}

@app.get("/latest_report")
def latest_report():
    """前端取得最新報告內容"""
    global last_report_msg
    return {"report": last_report_msg}

@app.get("/health")
def health():
    return {"status": "ok", "scanning": is_scanning or _manual_scan_status["running"]}

@app.get("/debug_scan")
def debug_scan(symbol: str = "3008"):
    """單股詳細診斷：回傳每一步的中間計算值"""
    result = {"symbol": symbol, "step1_fetch": None, "step2_minervini": None, "step3_vcp": None}
    start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    df = fetch_daily(symbol, start_date, end_date)
    if df is None:
        result["step1_fetch"] = "下載失敗（None）"
        return convert_numpy(result)
    result["step1_fetch"] = {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "tail_close": df["close"].tail(5).tolist() if "close" in df.columns else "無 close",
        "tail_max": df["max"].tail(5).tolist() if "max" in df.columns else "無 max",
    }
    mv = minervini_check_with_debug(df)
    result["step2_minervini"] = mv
    if mv.get("passed"):
        result["step3_vcp"] = vcp_math_check_with_debug(df)
    else:
        result["step3_vcp"] = "未執行（Minervini 未通過）"
    return convert_numpy(result)

# ========== 若需本地測試 ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)