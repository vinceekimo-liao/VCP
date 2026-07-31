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

# ========== 純 NumPy 替代 SciPy 極值算子 ==========
def argrelextrema(data, comparator, order=1, mode='clip'):
    """純 NumPy 實作之相對極值尋找函式（替代 scipy.signal.argrelextrema）"""
    data = np.asarray(data)
    datalen = len(data)
    locs = np.arange(0, datalen)
    results = np.ones(datalen, dtype=bool)
    main = np.take(data, locs, mode=mode)
    for shift in range(1, order + 1):
        plus = np.take(data, locs + shift, mode=mode)
        minus = np.take(data, locs - shift, mode=mode)
        results &= comparator(main, plus)
        results &= comparator(main, minus)
        if not results.any():
            return (np.array([], dtype=int),)
    return (np.nonzero(results)[0],)

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
sepa_stage2_candidates = []  # 儲存符合 SEPA Stage 2 範本個股
industry_map = {}            # 股票代號 -> 產業名稱
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

# ========== 股票清單與產業 Mapping ==========
_stock_ids_cache = {"ids": [], "ts": 0}

def get_filtered_stock_ids():
    global industry_map
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
            
            # 建立代號對應產業 Mapping
            if "industry_category" in info.columns:
                industry_map = dict(zip(info["stock_id"], info["industry_category"]))
            
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

# ========== 大盤健康度與加權指數檢查 ==========
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
        return False, None
        
    close = _get_col(data, "close", "Close")
    high  = _get_col(data, "max", "high", "High")
    low   = _get_col(data, "min", "low", "Low")
    
    if close is None or high is None or low is None:
        return False, None
        
    close = pd.to_numeric(close, errors='coerce').dropna()
    high  = pd.to_numeric(high,  errors='coerce').dropna()
    low   = pd.to_numeric(low,   errors='coerce').dropna()
    
    if len(close) < 250 or len(high) < 250 or len(low) < 250:
        return False, None

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
            return False, None
            
        # 2. 200MA 向上
        if m200_last <= ma200.iloc[-20]:
            return False, None
            
        # 3. 52 週高低點
        high_52w = high.tail(250).max()
        low_52w  = low.tail(250).min()
        
        if c_last < low_52w * 1.30:
            return False, None
            
        if c_last < high_52w * 0.75:
            return False, None
            
        pct_from_52w_high = round(float((c_last - high_52w) / high_52w * 100), 2)
        pct_from_52w_low  = round(float((c_last - low_52w) / low_52w * 100), 2)
        
        stage2_info = {
            "symbol": str(data["stock_id"].iloc[0]) if "stock_id" in data.columns else "",
            "price": round(float(c_last), 2),
            "ma50": round(float(m50_last), 2),
            "ma150": round(float(m150_last), 2),
            "ma200": round(float(m200_last), 2),
            "high_52w": round(float(high_52w), 2),
            "low_52w": round(float(low_52w), 2),
            "pct_from_52w_high": pct_from_52w_high,
            "pct_from_52w_low": pct_from_52w_low,
            "near_52w_high": bool(c_last >= high_52w * 0.85)
        }
            
        return True, stage2_info
    except Exception as e:
        return False, None

# ========== 第二層：修訂版 VCP 收縮演算法與動能突破判斷 ==========
def vcp_math_check(data):
    # VCP 型態建構通常需要 3~6 個月，資料量建議至少 120 根 K 線
    if data is None or len(data) < 120:
        return None

    # 1. 資料欄位清洗與轉換
    close  = pd.to_numeric(_get_col(data, "close", "Close"), errors='coerce')
    high   = pd.to_numeric(_get_col(data, "max", "high", "High"), errors='coerce')
    low    = pd.to_numeric(_get_col(data, "min", "low", "Low"), errors='coerce')
    volume = pd.to_numeric(_get_col(data, "Trading_Volume", "volume", "Volume"), errors='coerce')

    df_clean = pd.DataFrame({"close": close, "high": high, "low": low, "volume": volume}).dropna()
    if len(df_clean) < 120:
        return None

    c = df_clean["close"].reset_index(drop=True)
    h = df_clean["high"].reset_index(drop=True)
    l = df_clean["low"].reset_index(drop=True)
    v = df_clean["volume"].reset_index(drop=True)

    try:
        # 2. 技術指標計算 (量比與 VDU)
        vol_ma20 = v.rolling(20).mean()
        if pd.isna(vol_ma20.iloc[-1]) or vol_ma20.iloc[-1] == 0:
            return None
        vol_ratio = v.iloc[-1] / vol_ma20.iloc[-1]

        # VDU (Volume Dry-Up)：突破前 3 天量能嚴重窒息，要求低於 20MA 的 65%
        vdu_flag = v.iloc[-4:-1].mean() < (vol_ma20.iloc[-1] * 0.65)

        # 3. 動態尋找波段轉折點 (VCP核心邏輯)
        order_days = 5  # 前後 5 天最高/最低視為一個區域轉折點
        peaks_idx = argrelextrema(h.values, np.greater, order=order_days)[0]
        valleys_idx = argrelextrema(l.values, np.less, order=order_days)[0]

        # 僅觀察最近 80 天內形成的型態
        recent_limit = len(df_clean) - 80
        peaks_idx = [p for p in peaks_idx if p > recent_limit]
        valleys_idx = [v_idx for v_idx in valleys_idx if v_idx > recent_limit]

        # 配對轉折點，計算每次回檔（從高點到隨後最低點的跌幅）
        pullbacks = []
        for p_idx in peaks_idx:
            sub_valleys = [v_idx for v_idx in valleys_idx if v_idx > p_idx]
            if sub_valleys:
                v_idx = sub_valleys[0]
                peak_p = h.iloc[p_idx]
                valley_p = l.iloc[v_idx]
                
                # 正確的 VCP 回檔幅度公式：(高 - 低) / 高
                drop_pct = (peak_p - valley_p) / peak_p * 100
                pullbacks.append(round(drop_pct, 2))

        # 驗證收縮邏輯：後次的波動幅度必須比前次小
        vcp_contracting = False
        contractions = len(pullbacks)

        if contractions >= 2:
            if pullbacks[-2] > pullbacks[-1]:
                if contractions >= 3:
                    vcp_contracting = pullbacks[-3] > pullbacks[-2]
                else:
                    vcp_contracting = True

        # 如果轉折點抓取雜訊導致次數異常（通常 VCP 不會超過 5 次收縮）
        if contractions > 5:
            vcp_contracting = False

        # 4. 樞紐突破 (Pivot Breakout)
        today_change = ((c.iloc[-1] - c.iloc[-2]) / c.iloc[-2] * 100) if len(c) >= 2 else 0
        # 樞紐高點定義為最近一次收縮的高點
        pivot_high = h.iloc[peaks_idx[-1]] if len(peaks_idx) > 0 else h.iloc[-20:-1].max()
        
        # 突破訊號：今日收盤價必須站上樞紐高點的 99% 以上，且當天是收紅K
        is_pivot_breakout = (c.iloc[-1] >= pivot_high * 0.99) and (c.iloc[-1] >= c.iloc[-2])

        # 5. 相對強度 (RS) 門檻
        rs_lookback = min(60, len(c))
        past_close = c.iloc[-rs_lookback]
        if past_close <= 0:
            return None
        rs_raw = 50 + (c.iloc[-1] - past_close) / past_close * 200
        rs = int(max(1, min(99, round(float(rs_raw)))))

        # 篩選核心強勢股門檻
        if rs < 85 or vol_ratio < 1.3:
            return None

        # 6. 品質評分 (Quality Score)
        qs = 0
        if 1.5 <= vol_ratio <= 3.0:  # 突破當天量能溫和放大
            qs += 2
        elif vol_ratio > 3.0:        # 爆量突破
            qs += 1
        if vdu_flag:                 # 伴隨量能窒息
            qs += 1
        if vcp_contracting:          # 具備完美的遞減收縮
            qs += 2
        if rs >= 95:                 # 強勢股加分
            qs += 1

        quality = "A" if qs >= 5 else "B" if qs >= 3 else "C"

        # 7. 趨勢樣板與進場訊號 (Trend Template)
        ma50  = c.rolling(50).mean().iloc[-1]
        ma150 = c.rolling(150).mean().iloc[-1]
        ma200 = c.rolling(200).mean().iloc[-1]
        last  = c.iloc[-1]

        # 必須處於多頭排列，且具備收縮型態、完成樞紐突破與量能放大
        buy_signal = (
            vcp_contracting and
            is_pivot_breakout and
            vol_ratio >= 1.5 and
            last > ma50 > ma150 > ma200
        )

        sid = str(data["stock_id"].iloc[0]) if "stock_id" in data.columns else ""

        return {
            "symbol": sid,
            "industry": industry_map.get(sid, "其他"),
            "price": round(float(last), 2),
            "change_pct": round(float(today_change), 2),
            "rs_score": rs,
            "contractions": contractions if vcp_contracting else 0, # 不符合收縮則次數歸零
            "volume_ratio": round(float(vol_ratio), 2),
            "vdu": vdu_flag,
            "quality": quality,
            "buy_signal": bool(buy_signal),
            "consecutive_days": 0,
            "pullback_history": str(pullbacks) # 額外輸出回檔紀錄字串，方便 Excel 檢視
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
                c["filter_note"] = "空頭防禦性試探部位 (5%~10%倉位)"
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

# ========== SEPA + VCP 數據分析與 Excel / Email 產出 ==========
def generate_and_send_sepa_vcp_report(results, total_scanned, stage2_list):
    if not RESEND_API_KEY or not EMAIL_SENDER or not EMAIL_RECEIVER:
        print("⚠️ 缺少 Resend 郵件設定，跳過 Email 寄送")
        return

    now_dt = datetime.now()
    timestamp_str = now_dt.strftime("%Y%m%d%H%M%S")
    date_str = now_dt.strftime("%Y-%m-%d %H:%M")

    # 1. 整理各分頁資料集
    df_all_vcp = pd.DataFrame(results) if results else pd.DataFrame()
    df_stage2 = pd.DataFrame(stage2_list) if stage2_list else pd.DataFrame()

    if not df_all_vcp.empty:
        df_all_vcp = df_all_vcp.sort_values("rs_score", ascending=False)
        df_buy_signals = df_all_vcp[df_all_vcp["buy_signal"] == True].copy()
    else:
        df_buy_signals = pd.DataFrame()

    # 產業族群統計
    group_summary_df = pd.DataFrame()
    if not df_stage2.empty:
        df_stage2["industry"] = df_stage2["symbol"].map(lambda x: industry_map.get(x, "其他"))
        group_counts = df_stage2.groupby("industry").size().reset_index(name="強勢股數量")
        group_counts = group_counts.sort_values("強勢股數量", ascending=False)
        
        # 標註領頭族群 (>=2 檔)
        group_counts["族群狀態"] = group_counts["強勢股數量"].apply(lambda x: "🔥 領頭族群共振" if x >= 2 else "單兵強勢")
        group_summary_df = group_counts

    # 大盤與市場廣度簡報
    market_bull, market_price = get_market_status()
    near_52w_high_count = len([s for s in stage2_list if s.get("near_52w_high")]) if stage2_list else 0

    breadth_summary = pd.DataFrame([{
        "分析日期": date_str,
        "總掃描股票數": total_scanned,
        "Stage2趨勢範本合格數": len(stage2_list) if stage2_list else 0,
        "接近52週高點數量(15%內)": near_52w_high_count,
        "VCP選股符合數": len(results) if results else 0,
        "最終黃金買點訊號數": len(df_buy_signals) if not df_buy_signals.empty else 0,
        "加權指數價格": market_price,
        "加權指數>20MA": "✅ 多頭" if market_bull else "⚠️ 震盪/空頭",
        "建議部位風控": "正常進場 (10%~15%)" if market_bull else "嚴格防守/試倉 (5%~10%)"
    }])

    # 2. 寫入多頁籤 Excel：SEPA+VCP_yyyymmddhhmmss.xlsx
    sepa_excel_filename = f"SEPA+VCP_{timestamp_str}.xlsx"
    vcp_excel_filename = f"VCP_{now_dt.strftime('%Y%m%d_%H%M%S')}.xlsx"

    try:
        # 建立專屬 SEPA + VCP 報表
        with pd.ExcelWriter(sepa_excel_filename, engine='openpyxl') as writer:
            breadth_summary.to_excel(writer, sheet_name="大盤廣度與健康度簡報", index=False)
            
            if not df_buy_signals.empty:
                df_buy_signals.to_excel(writer, sheet_name="SEPA+VCP 最終進場訊號", index=False)
            else:
                pd.DataFrame([{"訊息": "今日無最終進場訊號"}]).to_excel(writer, sheet_name="SEPA+VCP 最終進場訊號", index=False)

            if not group_summary_df.empty:
                group_summary_df.to_excel(writer, sheet_name="產業族群共振榜", index=False)

            if not df_stage2.empty:
                df_stage2.to_excel(writer, sheet_name="SEPA 趨勢範本合格池", index=False)

            if not df_all_vcp.empty:
                df_all_vcp.to_excel(writer, sheet_name="VCP 全漏斗候選", index=False)

        # 保留原有的 VCP_yyyymmdd_hhmmss.xlsx 檔案格式以防外部引用
        with pd.ExcelWriter(vcp_excel_filename, engine='openpyxl') as writer:
            if not df_all_vcp.empty:
                df_all_vcp.to_excel(writer, sheet_name="全部漏斗候選", index=False)
                df_buy_signals.to_excel(writer, sheet_name="最終進場訊號", index=False)
            else:
                pd.DataFrame().to_excel(writer, sheet_name="全部漏斗候選")

        print(f"📁 專業 SEPA+VCP 報表已建立：{sepa_excel_filename}")

        # 3. 透過 Resend 發送 Email 附件
        resend.api_key = RESEND_API_KEY
        attachments = []

        for fn in [sepa_excel_filename, vcp_excel_filename]:
            if os.path.exists(fn):
                with open(fn, "rb") as f:
                    attachments.append({
                        "filename": fn,
                        "content": list(f.read())
                    })

        buy_count = len(df_buy_signals) if not df_buy_signals.empty else 0
        html_content = f"""
        <h2>📈 SEPA + VCP 每日量化交易系統分析報告</h2>
        <p><b>分析時間：</b>{date_str}</p>
        <ul>
            <li><b>總掃描標的：</b>{total_scanned} 檔</li>
            <li><b>Stage 2 趨勢範本符合：</b>{len(stage2_list) if stage2_list else 0} 檔</li>
            <li><b>接近 52 週高點 (15%內)：</b>{near_52w_high_count} 檔</li>
            <li><b>VCP 形態與籌碼合格：</b>{len(results) if results else 0} 檔</li>
            <li><b>🔥 最終 SEPA+VCP 進場訊號：</b><b style='color:red;'>{buy_count} 檔</b></li>
            <li><b>加權指數狀態：</b>{'多頭格局 (Risk-On)' if market_bull else '震盪/空頭防禦 (Risk-Off)'}</li>
        </ul>
        <p>詳細 SEPA 產業共振、大盤廣度與買點訊號請參閱附件 Excel 檔案：<b>{sepa_excel_filename}</b></p>
        """

        params = {
            "from": f"SEPA+VCP 量化掃描器 <{EMAIL_SENDER}>",
            "to": [EMAIL_RECEIVER],
            "subject": f"🎯 SEPA+VCP 每日量化策略報告 {now_dt.strftime('%Y-%m-%d')} (進場訊號: {buy_count} 檔)",
            "html": html_content,
            "attachments": attachments
        }
        response = resend.Emails.send(params)
        print(f"✅ 每日 SEPA+VCP Email 已成功寄送至 {EMAIL_RECEIVER}，Resend ID: {response['id']}")

    except Exception as e:
        print(f"❌ SEPA+VCP Email 寄送失敗：{type(e).__name__} - {str(e)}")

def send_email_report(results, total_scanned):
    """向下相容之原始呼叫入口"""
    generate_and_send_sepa_vcp_report(results, total_scanned, sepa_stage2_candidates)

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
    global _manual_scan_status, scan_results, trade_signals, sepa_stage2_candidates
    _manual_scan_status["running"] = True
    _manual_scan_status["done"] = 0
    _manual_scan_status["results"] = []
    sepa_stage2_candidates = []
    
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
        if df is not None:
            is_stage2, stage2_info = minervini_check(df)
            if is_stage2:
                layer1_pass += 1
                if stage2_info:
                    sepa_stage2_candidates.append(stage2_info)
                res = vcp_math_check(df)
                if res:
                    res["symbol"] = sid
                    _manual_scan_status["results"].append(res)
                    
        _manual_scan_status["done"] = idx
        if idx % 100 == 0:
            print(f"📊 進度：{idx}/{total}，第一層 Stage2 通過：{layer1_pass}，VCP 候選：{len(_manual_scan_status['results'])}")
            
    _manual_scan_status["running"] = False
    with scan_lock:
        scan_results = _manual_scan_status["results"]
    trade_signals = apply_trade_filters(scan_results)
    
    # 執行包含 SEPA+VCP 分析的大盤與 Excel 寄送
    generate_and_send_sepa_vcp_report(scan_results, total, sepa_stage2_candidates)
    print(f"✅ 手動掃描完成，Stage2 通過：{layer1_pass} 檔，VCP 候選：{len(scan_results)} 檔，交易訊號：{len(trade_signals)} 檔")

# ========== 夜間背景掃描 ==========
def background_scanner():
    global scan_results, last_report_msg, _manual_scan_status, trade_signals, sepa_stage2_candidates
    stocks = get_filtered_stock_ids()
    if not stocks:
        print("❌ 無股票清單，夜間掃描終止")
        return
        
    total = len(stocks)
    start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    local_results = []
    local_stage2 = []
    layer1_pass = 0
    
    for idx, sid in enumerate(stocks, 1):
        df = fetch_daily(sid, start_date, end_date)
        if df is not None:
            is_stage2, stage2_info = minervini_check(df)
            if is_stage2:
                layer1_pass += 1
                if stage2_info:
                    local_stage2.append(stage2_info)
                res = vcp_math_check(df)
                if res:
                    local_results.append(res)
                    
        if idx % 100 == 0:
            print(f"📊 背景掃描進度：{idx}/{total}，Stage2 通過：{layer1_pass}，VCP 候選：{len(local_results)}")
            
    with scan_lock:
        scan_results = local_results
        sepa_stage2_candidates = local_stage2
        
    _manual_scan_status["running"] = False
    _manual_scan_status["total"] = total
    _manual_scan_status["done"] = total
    _manual_scan_status["results"] = local_results
    
    last_report_msg = build_report(total, scan_results)
    send_telegram_msg(last_report_msg)
    trade_signals = apply_trade_filters(scan_results)
    
    # 執行包含 SEPA+VCP 分析的大盤與 Excel 寄送
    generate_and_send_sepa_vcp_report(scan_results, total, sepa_stage2_candidates)
    print(f"✅ 背景掃描完成，Stage2 通過：{layer1_pass} 檔，VCP 候選：{len(scan_results)} 檔，交易訊號：{len(trade_signals)} 檔")

def build_report(total, results):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    if not results:
        return f"📉 <b>每日 VCP 報告 ({now_str})</b>\n掃描 {total} 檔，無符合條件股票"
    sorted_results = sorted(results, key=lambda x: -x["rs_score"])
    msg = f"📈 <b>每日 SEPA+VCP 報告 ({now_str})</b>\n掃描 {total} 檔，符合 {len(results)} 檔\n\n"
    for i, c in enumerate(sorted_results[:15], 1):
        symbol = c['symbol']
        yahoo_link = f"https://tw.stock.yahoo.com/quote/{symbol}"
        msg += f"🔹 <b>{symbol}</b> ({c.get('industry','其他')}) | 價:{c['price']} | RS:{c['rs_score']} | 品質:{c['quality']} <a href='{yahoo_link}'>📈 Yahoo</a>\n"
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
    generate_and_send_sepa_vcp_report(current, total, sepa_stage2_candidates)
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
    html = f"""<html><head><meta charset='utf-8'><title>SEPA+VCP 完整報告</title>
    <style>
        body {{ background: #060d16; color: #e2f0ff; font-family: sans-serif; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
        th {{ background: #1a2d40; padding: 6px; text-align: left; }}
        td {{ padding: 6px; border-bottom: 1px solid #1a2d40; }}
        a {{ color: #38bdf8; }}
    </style></head><body>
    <h2>📈 SEPA + VCP 完整篩選報告</h2>
    <p>掃描 {total} 檔，符合 VCP 形態 {len(sorted_results)} 檔</p>
    <table><tr><th>代號</th><th>產業</th><th>股價</th><th>漲跌%</th><th>RS</th><th>收縮次數</th><th>量比</th><th>品質</th><th>進場訊號</th><th>Yahoo</th></tr>"""
    for c in sorted_results:
        buy_icon = "✅" if c.get("buy_signal") else "❌"
        ind = c.get("industry", "其他")
        html += f"<tr><td>{c['symbol']}</td><td>{ind}</td><td>{c['price']}</td><td>{c['change_pct']:+.2f}%</td><td>{c['rs_score']}</td><td>{c['contractions']}</td><td>{c['volume_ratio']}</td><td>{c['quality']}</td><td>{buy_icon}</td><td><a href='https://tw.stock.yahoo.com/quote/{c['symbol']}' target='_blank'>Yahoo</a></td></tr>"
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
        "suggestion": "暫停進場，或只投 5%~10% 防禦性試倉" if not market_bull else "可正常進場 (10%~15%)",
        "candidates": trade_signals,
        "entry_tips": [
            "⚠️ 開盤觀察 (9:00-10:00)：若開盤即跌破篩選日最低價 → 刪除",
            "⚠️ 若開盤跳空大漲 (>5%) → 觀望不追（可能已過高點）",
            "✅ 若開盤平穩或小幅上漲 → 10:00 後進場",
            "✅ 進場價：不超過篩選日收盤價 +2%",
            "✅ 倉位：試倉 5%~10%，獲利後才加碼"
        ],
        "exit_tips": [
            "停利：+5% 或持有 3 天後評估移動停利",
            "停損：硬性 7%~8% 或跌破關鍵樞紐低點",
            "時間停損：5 天未達 +3% 即出場離場"
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
