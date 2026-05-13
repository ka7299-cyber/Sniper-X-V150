import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
import requests
import time
import csv
from io import StringIO
import urllib3

# 關閉警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 頁面配置
st.set_page_config(page_title="Sniper X V151 (Pro Chart)", layout="wide")

# UI 魔法：防止文字切斷 & 縮小 Metric 字體
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem !important; }
    .stMetric { border: 1px solid #f0f2f6; padding: 5px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================
# 1. 籌碼探針引擎 (比照 V75 智慧補位邏輯)
# ==============================================
class ChipCrawlerV151:
    def __init__(self, stock_id, is_otc=False):
        self.stock_id = str(stock_id).strip()
        self.is_otc = is_otc 
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    def get_trend_summary(self, dates, lookback_days=5):
        history = []
        for d in dates:
            if len(history) >= lookback_days: break
            margin = self._get_margin(d)
            inst = self._get_inst(d)
            sbl = self._get_sbl(d)
            if margin or inst or sbl:
                history.append({'date': d, 'margin': margin, 'inst': inst, 'sbl': sbl})
        return history

    def _get_margin(self, date_obj):
        date_str = date_obj.strftime('%Y%m%d')
        if self.is_otc:
            roc_date = f"{date_obj.year - 1911}/{date_obj.month:02d}/{date_obj.day:02d}"
            url = f"https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php?l=zh-tw&o=csv&d={roc_date}&s=0,asc,0"
        else:
            url = f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={date_str}&selectType=ALL&response=csv"
        try:
            res = requests.get(url, headers=self.headers, verify=False, timeout=5)
            content = res.content.decode('big5', errors='ignore')
            reader = csv.reader(StringIO(content))
            for row in reader:
                if row and self.stock_id in row[0]:
                    def clean(v): return int(str(v).replace(',', '').strip())
                    if self.is_otc: return clean(row[6]), (clean(row[6]) - clean(row[2])), clean(row[14]), (clean(row[14]) - clean(row[10]))
                    else: return clean(row[6]), (clean(row[6]) - clean(row[5])), clean(row[12]), (clean(row[12]) - clean(row[11]))
            return None
        except: return None

    def _get_inst(self, date_obj):
        date_str = date_obj.strftime('%Y%m%d')
        if self.is_otc:
            roc_date = f"{date_obj.year - 1911}/{date_obj.month:02d}/{date_obj.day:02d}"
            url = f"https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php?l=zh-tw&o=csv&se=EW&t=D&d={roc_date}"
        else:
            url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={date_str}&selectType=ALL&response=json"
        try:
            res = requests.get(url, headers=self.headers, verify=False, timeout=5)
            if self.is_otc:
                content = res.content.decode('big5', errors='ignore')
                reader = csv.reader(StringIO(content))
                for row in reader:
                    if row and self.stock_id in row[0]:
                        def clean(v): return int(str(v).replace(',', '').strip()) // 1000
                        return clean(row[10]), clean(row[13]), (clean(row[14]) + clean(row[15]))
            else:
                data = res.json()
                if data['stat'] == 'OK':
                    df = pd.DataFrame(data['data'], columns=data['fields'])
                    row = df[df['證券代號'] == self.stock_id]
                    if not row.empty:
                        rec = row.iloc[0]
                        def clean(v): return int(v.replace(',', '').strip()) // 1000
                        d_val = clean(rec.get('自營商買賣超股數', '0')) if '自營商買賣超股數' in rec else clean(rec.get('自營商買賣超股數(自行買賣)', '0')) + clean(rec.get('自營商買賣超股數(避險)', '0'))
                        return clean(rec['外陸資買賣超股數(不含外資自營商)']), clean(rec['投信買賣超股數']), d_val
            return None
        except: return None

    def _get_sbl(self, date_obj):
        date_str = date_obj.strftime('%Y%m%d')
        if self.is_otc:
            roc_date = f"{date_obj.year - 1911}/{date_obj.month:02d}/{date_obj.day:02d}"
            url = f"https://www.tpex.org.tw/web/stock/margin_trading/margin_sbl/margin_sbl_result.php?l=zh-tw&o=csv&d={roc_date}"
        else:
            url = f"https://www.twse.com.tw/rwd/zh/marginTrading/TWT93U?date={date_str}&response=csv"
        try:
            res = requests.get(url, headers=self.headers, verify=False, timeout=5)
            content = res.content.decode('big5', errors='ignore')
            reader = csv.reader(StringIO(content))
            for row in reader:
                if row and self.stock_id in row[0]:
                    def clean(v): return int(str(v).replace(',', '').strip()) // 1000 
                    return clean(row[12]), (clean(row[12]) - clean(row[8]))
            return None
        except: return None

# ==============================================
# 2. 資料庫與 AI 引擎 (承襲 V145)
# ==============================================
TW_STRATEGIES = {
    '1210': (26, 48), '1216': (26, None), '1477': (25, None), '1514': (None, 51),
    '2006': (21, 48), '2301': (18, 53), '2303': (21, 48), '2308': (27, 97),
    '2313': (20, 61), '2317': (18, 57), '2324': (19, 57), '2327': (20, None),
    '2330': (17, 57), '2337': (28, None), '2344': (31, None), '2345': (28, 60),
    '2352': (19, 59), '2353': (17, 63), '2354': (34, None), '2356': (23, 58),
    '2357': (21, None), '2360': (21, None), '2362': (23, None), '2368': (22, 60),
    '2376': (None, 29), '2377': (18, None), '2379': (26, None), '2382': (23, 57),
    '2383': (18, 50), '2385': (20, 55), '2395': (22, 49), '2404': (29, None),
    '2408': (23, 43), '2409': (18, 52), '2428': (24, 59), '2439': (18, 74),
    '2454': (29, 60), '2472': (24, 48), '2496': (29, None), '2603': (35, None),
    '2727': (28, 46), '2753': (29, 52), '2755': (22, 53), '2891': (18, 47),
    '3005': (21, 62), '3017': (21, 55), '3029': (20, 43), '3036': (18, 53),
    '3037': (35, 70), '3081': (20, 60), '3130': (22, 36), '3231': (26, 76),
    '3443': (18, 67), '3583': (21, 50), '3706': (23, None), '4987': (25, None),
    '5904': (21, 57), '6138': (21, None), '6146': (25, 67), '6176': (22, None),
    '6191': (25, None), '6192': (29, None), '6197': (23, 48), '6201': (34, None),
    '6239': (23, 48), '6279': (19, 53), '6284': (24, 56), '6285': (21, 59),
    '6409': (23, 50), '6667': (26, 44), '6669': (28, 58), '6721': (17, 41),
    '6728': (19, 48), '6805': (18, None), '8210': (25, None), '8367': (22, 55),
    '9939': (17, 57)
}

TW_NAMES = {k: "大師鎖定" for k in TW_STRATEGIES.keys()}

@st.cache_data(ttl=600)
def fetch_data_robust(ticker_symbol):
    try:
        df = yf.Ticker(ticker_symbol).history(period="2y")
        if not df.empty: return df
    except: pass
    return pd.DataFrame()

def find_best_ma_v2(df, start_day, end_day):
    closes = df['Close'].values; lows = df['Low'].values
    best_ma = start_day; best_score = -np.inf
    for ma_len in range(start_day, end_day + 1):
        ma = df['Close'].rolling(window=ma_len).mean().values
        valid = slice(ma_len, len(df))
        min_idxs = argrelextrema(lows[valid], np.less, order=3)[0]
        if len(min_idxs) == 0: continue
        err = (np.abs(lows[valid][min_idxs] - ma[valid][min_idxs]) / ma[valid][min_idxs]).mean()
        score = 100 - (err * 3000) + (ma_len - start_day) * 0.8
        if score > best_score: best_score = score; best_ma = ma_len
    return best_ma

# ==============================================
# 3. 介面與顯示 (修正圖表瑕疵)
# ==============================================
st.sidebar.header("🕹️ Sniper X V151")
market = st.sidebar.radio("市場", ["🇹🇼 台股", "🇺🇸 美股"], horizontal=True)

if "🇹🇼" in market:
    stock_id = st.sidebar.text_input("輸入代號 (例如 2330)", "2330")
    t_symbol = f"{stock_id}.TW"
    df = fetch_data_robust(t_symbol)
    if df.empty: 
        t_symbol = f"{stock_id}.TWO"
        df = fetch_data_robust(t_symbol)
    
    if not df.empty:
        is_otc = ".TWO" in t_symbol
        p_short, p_long = TW_STRATEGIES.get(stock_id, (None, None))
        
        with st.spinner('🎯 籌碼探針偵測中...'):
            final_s = p_short if p_short else find_best_ma_v2(df, 16, 25)
            final_l = p_long if p_long else find_best_ma_v2(df, 45, 70)
            crawler = ChipCrawlerV151(stock_id, is_otc)
            recent_dates = df.index[-10:][::-1]
            chip_history = crawler.get_trend_summary(recent_dates, lookback_days=5)

        df['MS'] = df['Close'].rolling(window=final_s).mean()
        df['ML'] = df['Close'].rolling(window=final_l).mean()
        last = df.iloc[-1]; ms_v = last['MS']; ml_v = last['ML']; price = last['Close']
        
        # 頂部儀表板
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現價", f"{price:.2f}")
        c2.metric(f"短({final_s})", f"{ms_v:.2f}")
        c3.metric(f"長({final_l})", f"{ml_v:.2f}")
        
        # 趨勢判定
        if price > ms_v > ml_v: trend = "🔥 強勢多頭"
        elif ms_v >= price >= ml_v: trend = "⚠️ 多頭回檔"
        elif ml_v >= ms_v >= price: trend = "❄️ 絕對空頭"
        else: trend = "🧩 震盪整理"
        
        chip_msg = "🟢 籌碼中性"
        if chip_history:
            latest = chip_history[0]
            f, t, d = latest['inst'] if latest['inst'] else (0,0,0)
            if t > 500: chip_msg = "🚀 投信大買"
            elif f > 1000: chip_msg = "💰 外資敲進"
            if "多頭" in trend and (t > 0 or f > 0): trend = "🏆 雙刀流確認"

        c4.metric("戰情/籌碼", trend, chip_msg)

        # 籌碼趨勢區
        if chip_history:
            st.markdown("### 🔍 籌碼探針偵測結果 (近 5 日趨勢)")
            cols = st.columns(5)
            for i, data in enumerate(chip_history):
                with cols[i]:
                    d_str = data['date'].strftime('%m/%d')
                    f, t, d = data['inst'] if data['inst'] else (0,0,0)
                    m_c = data['margin'][1] if data['margin'] else 0
                    st.markdown(f"**{d_str}**")
                    st.write(f"外資: {f:+}")
                    st.write(f"投信: {t:+}")
                    st.write(f"資增: {m_c:+}")
        
        # --- ★ 修正後的無縫雙層圖表 ★ ---
        p_df = df.tail(60).copy()
        # 關鍵：轉為字串並移除年份，避免 Plotly 自動生成時間軸
        p_df.index = p_df.index.strftime('%m-%d')
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.3, 0.7])
        
        # K 線與均線
        fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K棒', increasing_line_color='#ef5350', decreasing_line_color='#26a69a'), row=1, col=1)
        fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MS'], name='短線', line=dict(color='orange', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=p_df.index, y=p_df['ML'], name='長線', line=dict(color='purple', width=2)), row=1, col=1)
        
        # 成交量
        v_colors = ['#ef5350' if c >= o else '#26a69a' for c, o in zip(p_df['Close'], p_df['Open'])]
        fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], marker_color=v_colors, name='成交量'), row=2, col=1)
        
        # ★ 關鍵設定：強制使用 category 模式移除間隙 ★
        fig.update_layout(height=600, template="plotly_white", xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=0,r=20,t=0,b=0), hovermode="x unified")
        fig.update_xaxes(type='category', nticks=10, row=1, col=1)
        fig.update_xaxes(type='category', nticks=10, row=2, col=1)
        fig.update_yaxes(side="right")
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.error("代號錯誤或無資料")
else:
    st.info("美股模式暫不支援台股籌碼探針")
