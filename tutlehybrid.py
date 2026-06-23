import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

try:
    from pykrx import stock as krx
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False

# --- 페이지 설정 ---
st.set_page_config(page_title="CAN SLIM x 터틀 실전 매니저", layout="wide")
st.title("🦅 CAN SLIM x 🐢 터틀 트레이딩 실전 자산 매니저")
st.markdown("""
본 프로그램은 **CAN SLIM** 기준으로 유망 종목을 발굴하고, **터틀 트레이딩** 원칙에 기반하여 
사용자가 **실제 매수한 포지션의 손절, 피라미딩(추가매수), 청산**을 실시간으로 가이드합니다.
""")

if not PYKRX_AVAILABLE:
    st.warning("⚠️ `pykrx` 라이브러리가 설치되어 있지 않아 국내 종목명이 코드로만 표시됩니다.")

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ 시스템 및 자금 관리 설정")

# 1. 시스템 선택 (터틀)
system_type = st.sidebar.radio(
    "터틀 트레이딩 시스템 선택",
    ("시스템 1 (20일 고점/10일 저점)", "시스템 2 (55일 고점/20일 저점)")
)
entry_window, exit_window = (20, 10) if system_type == "시스템 1 (20일 고점/10일 저점)" else (55, 20)

st.sidebar.divider()

# 2. 자금 관리 설정
account_size = st.sidebar.number_input("총 투자 자본금", value=100000, step=10000)
risk_per_trade = st.sidebar.slider("1회 거래당 리스크 비율 (%, 1유닛 기준)", min_value=0.5, max_value=5.0, value=1.0, step=0.1) / 100
max_units = st.sidebar.slider("최대 피라미딩 유닛 수", min_value=1, max_value=4, value=4)

st.sidebar.divider()
apply_market_filter = st.sidebar.checkbox("시장이 하락추세면 매수등급 자동 하향", value=True)

# --- 기본 함수들 (종목명, 가격정보, 시장방향성 등) ---
@st.cache_data(ttl=86400)
def get_stock_name(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if PYKRX_AVAILABLE and (ticker.endswith(".KS") or ticker.endswith(".KQ")):
        code = ticker.split(".")[0]
        try:
            name = krx.get_market_ticker_name(code)
            if name: return name
        except Exception: pass
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker
    except Exception: return ticker

def get_benchmark_ticker(ticker: str) -> str:
    if ticker.endswith(".KS"): return "^KS11"
    elif ticker.endswith(".KQ"): return "^KQ11"
    return "^GSPC"

@st.cache_data(ttl=3600)
def load_price_history(ticker: str, days: int = 500) -> pd.DataFrame | None:
    start = datetime.today() - timedelta(days=days)
    try:
        df = yf.download(ticker, start=start, end=datetime.today(), progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        return df
    except Exception: return None

@st.cache_data(ttl=3600)
def get_market_trend(benchmark_ticker: str) -> dict:
    df = load_price_history(benchmark_ticker, days=400)
    if df is None or len(df) < 200: return {"status": "확인불가", "detail": "데이터 부족"}
    close = df["Close"]
    sma50, sma200 = close.rolling(50).mean().iloc[-1], close.rolling(200).mean().iloc[-1]
    sma200_prev = close.rolling(200).mean().iloc[-20] if len(df) > 220 else sma200
    if (close.iloc[-1] > sma50) and (close.iloc[-1] > sma200) and (sma200 > sma200_prev):
        return {"status": "🟢 상승추세", "detail": "지수가 50/200일선 위, 200일선 상승 중"}
    elif (close.iloc[-1] < sma50) and (close.iloc[-1] < sma200) and (sma200 < sma200_prev):
        return {"status": "🔴 하락추세", "detail": "지수가 이평선 아래, 200일선 하락 중"}
    return {"status": "🟡 중립/전환구간", "detail": "이평선 신호 혼재"}

def compute_rs_raw(df: pd.DataFrame) -> float | None:
    close = df["Close"].dropna()
    if len(close) < 253: return None
    def r(e, s): return (close.iloc[e] / close.iloc[s]) - 1.0 if close.iloc[s] != 0 else 0
    return 2 * r(-1, -64) + r(-64, -127) + r(-127, -190) + r(-190, -253)

def rs_rating_from_raw(raw_scores: dict) -> dict:
    valid = {k: v for k, v in raw_scores.items() if v is not None}
    if len(valid) <= 1: return {k: 50 for k in raw_scores}
    values = sorted(valid.values())
    return {k: (max(1, min(99, round((sum(1 for x in values if x <= v) / len(values)) * 98) + 1)) if v is not None else None) for k, v in raw_scores.items()}

def compute_volume_signal(df: pd.DataFrame) -> dict:
    recent = df.tail(50).copy()
    recent["change"] = recent["Close"].diff()
    up_vol = recent.loc[recent["change"] > 0, "Volume"].sum()
    down_vol = recent.loc[recent["change"] < 0, "Volume"].sum()
    ratio = up_vol / down_vol if down_vol > 0 else 1.0
    return {"ratio": round(ratio, 2), "signal": "🟢 매집" if ratio >= 1.2 else "🔴 분산" if ratio <= 0.8 else "🟡 중립"}

@st.cache_data(ttl=3600)
def load_and_process_data(ticker, entry_w, exit_w):
    df = load_price_history(ticker, days=500)
    if df is None: return None
    try:
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_150'] = df['Close'].rolling(150).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df['52W_High'], df['52W_Low'] = df['High'].rolling(250).max(), df['Low'].rolling(250).min()
        df['SMA_200_Trend'] = df['SMA_200'] > df['SMA_200'].shift(20)
        df['Entry_High'] = df['High'].rolling(entry_w).max().shift(1)
        df['Exit_Low'] = df['Low'].rolling(exit_w).min().shift(1)
        ranges = pd.concat([df['High']-df['Low'], np.abs(df['High']-df['Close'].shift(1)), np.abs(df['Low']-df['Close'].shift(1))], axis=1)
        df['ATR'] = ranges.max(axis=1).rolling(20).mean()
        return df
    except Exception: return None

# --- [세션 상태] 실전 포지션 데이터 초기화 ---
if "active_positions" not in st.session_state:
    st.session_state.active_positions = pd.DataFrame([
        {"티커": "AAPL", "실제최초매수가": 175.0, "현재보유유닛": 2},
        {"티커": "005930.KS", "실제최초매수가": 72000.0, "현재보유유닛": 1}
    ])

# 데이터 미리 로드 및 캐싱
all_tickers = list(set(list(st.session_state.active_positions["티커"]) + ["AAPL", "MSFT", "NVDA", "TSLA", "005930.KS"]))
_df_cache = {t: load_and_process_data(t, entry_window, exit_window) for t in all_tickers}
_raw_cache = {t: compute_rs_raw(_df_cache[t]) if _df_cache[t] is not None else None for t in all_tickers}
rs_ratings = rs_rating_from_raw(_raw_cache)
market_trend_map = {b: get_market_trend(b) for b in set(get_benchmark_ticker(t) for t in all_tickers)}

# --- 탭 구성 (실전 관리 탭을 가장 앞으로 배치) ---
tab0, tab1, tab2 = st.tabs(["🔥 1. 실전 보유 포지션 관리", "📊 2. CAN SLIM 관심종목 스캐너", "📈 3. 개별 종목 융합 차트"])

# =========================================================
# 탭 0: 실전 보유 포지션 관리 (가장 중요하게 개선된 심장부)
# =========================================================
with tab0:
    st.subheader("🛠️ 현재 실제 보유 중인 포지션")
    st.markdown("💬 **매수한 종목의 정보(티커, 최초매수가, 유닛 수)를 아래 테이블에서 실시간으로 수정하거나 추가할 수 있습니다.**")
    
    # 사용자가 직접 데이터를 편집할 수 있는 인터페이스 제공
    edited_df = st.data_editor(
        st.session_state.active_positions, 
        num_rows="dynamic", 
        use_container_width=True,
        key="position_editor"
    )
    st.session_state.active_positions = edited_df

    st.divider()
    st.subheader("🚨 실시간 보유 포지션 대응 알림판")

    real_management_data = []
    
    for _, row in edited_df.iterrows():
        ticker = str(row["티커"]).strip().upper()
        init_price = float(row["실제최초매수가"])
        held_units = int(row["현재보유유닛"])
        
        if not ticker: continue
        
        df = load_and_process_data(ticker, entry_window, exit_window) if ticker not in _df_cache else _df_cache[ticker]
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            c_price = float(latest['Close'])
            atr = float(latest['ATR'])
            exit_channel = float(latest['Exit_Low'])
            
            # --- 실전 터틀 원칙 공식 계산 ---
            # 1. 터틀 규칙: 유닛이 하나씩 추가될 때마다 전량 손절선은 +0.5N 만큼 상향 조절됨
            # 최종 유닛의 매수가 = 최초매수가 + 0.5 * ATR * (보유유닛 - 1)
            # 실전 공통 손절선 = 최종 유닛 매수가 - 2 * ATR
            latest_unit_price = init_price + (0.5 * atr * (held_units - 1))
            actual_stop_loss = latest_unit_price - (2 * atr)
            
            # 2. 다음 피라미딩(증액) 가격 계산
            next_pyramid_price = init_price + (0.5 * atr * held_units)
            
            # 3. 실시간 현재가와 비교하여 '액션 가이드' 도출
            if c_price <= actual_stop_loss:
                action_guide = "🚨 즉시 매도 (2N 실전 손절선 이탈!)"
                status_color = "🔴"
            elif c_price <= exit_channel:
                action_guide = "🚨 즉시 매도 (채널 청산선 이탈!)"
                status_color = "🔴"
            elif held_units < max_units and c_price >= next_pyramid_price:
                action_guide = f"➕ 증액 추천 (+1유닛 추가 매수 완료 기준가: {round(next_pyramid_price, 2)})"
                status_color = "🔵"
            else:
                action_guide = "🟢 정상 보유 (추세 유지 중)"
                status_color = "🟢"
                
            pnl_pct = ((c_price - init_price) / init_price) * 100
            risk_amount = account_size * risk_per_trade
            unit_shares = int(risk_amount / atr) if atr > 0 else 0
            
            real_management_data.append({
                "상태": status_color,
                "종목명": get_stock_name(ticker),
                "티커": ticker,
                "현재가": round(c_price, 2),
                "최초 매수가": round(init_price, 2),
                "보유 유닛": f"{held_units} / {max_units}",
                "수익률": f"{pnl_pct:+.2f}%",
                "실전 손절가(2N)": round(actual_stop_loss, 2),
                "채널 청산선": round(exit_channel, 2),
                "다음 증액 목표가": round(next_pyramid_price, 2) if held_units < max_units else "최대 유닛 도달",
                "1유닛 적정 수량": f"{unit_shares} 주",
                "실시간 대응 가이드": action_guide
            })
            
    if real_management_data:
        real_df = pd.DataFrame(real_management_data)
        st.dataframe(real_df, use_container_width=True, hide_index=True)
    else:
        st.info("상단 테이블에 보유 중인 주식 정보를 입력하시면 실시간 자금 관리 계힉이 활성화됩니다.")

# =========================================================
# 탭 1: CAN SLIM 관심종목 스캐너 (기존 기능 유지 및 정돈)
# =========================================================
with tab1:
    st.subheader("🌐 시장 방향성(M) 현황")
    mcols = st.columns(len(market_trend_map) if market_trend_map else 1)
    bench_names = {"^KS11": "코스피(KOSPI)", "^KQ11": "코스닥(KOSDAQ)", "^GSPC": "S&P500"}
    for col, (bench, info) in zip(mcols, market_trend_map.items()):
        with col:
            st.metric(bench_names.get(bench, bench), info["status"])
            st.caption(info["detail"])

    st.divider()
    st.subheader("🔍 관심종목 발굴 스캐너")
    
    tickers_input = st.text_input("스캔할 관심 종목 리스트 (쉼표 구분)", "AAPL, MSFT, NVDA, TSLA, 005930.KS")
    scan_tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    portfolio_data = []
    for ticker in scan_tickers:
        df = load_and_process_data(ticker, entry_window, exit_window) if ticker not in _df_cache else _df_cache[ticker]
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            c_price = latest['Close']
            
            cond1 = (c_price > latest['SMA_150']) and (c_price > latest['SMA_200'])
            cond2 = latest['SMA_150'] > latest['SMA_200']
            cond3 = bool(latest['SMA_200_Trend'])
            cond4 = (latest['SMA_50'] > latest['SMA_150']) and (latest['SMA_50'] > latest['SMA_200'])
            cond5 = c_price > latest['SMA_50']
            cond6 = c_price >= (latest['52W_High'] * 0.75)
            cond7 = c_price >= (latest['52W_Low'] * 1.30)
            trend_score = sum([cond1, cond2, cond3, cond4, cond5, cond6, cond7])
            
            rs_val = rs_ratings.get(ticker, 50)
            cond_rs = rs_val >= 70 if rs_val else False
            vol_info = compute_volume_signal(df)
            cond_vol = vol_info["signal"] == "🟢 매집"
            
            canslim_score = trend_score + int(cond_rs) + int(cond_vol)
            
            bench = get_benchmark_ticker(ticker)
            market_bearish = "하락추세" in market_trend_map.get(bench, {}).get("status", "")
            
            if canslim_score >= 7: base_status, base_emoji = "강력 매수 고려", "🔥"
            elif canslim_score >= 5: base_status, base_emoji = "관망/대기", "🟡"
            else: base_status, base_emoji = "추세 약함", "❄️"
            
            status_text = f"🟠 시장약세로 보류 ({canslim_score}/9)" if apply_market_filter and market_bearish and base_status == "강력 매수 고려" else f"{base_emoji} {base_status} ({canslim_score}/9)"
            
            portfolio_data.append({
                "종목명": get_stock_name(ticker), "티커": ticker, "CAN SLIM 점수": status_text,
                "RS Rating": rs_val if rs_val else "-", "수급 신호": vol_info["signal"],
                "현재가": round(c_price, 2), "진입 기준선(채널)": round(latest['Entry_High'], 2)
            })
            
    st.dataframe(pd.DataFrame(portfolio_data), use_container_width=True, hide_index=True)

# =========================================================
# 탭 2: 개별 종목 융합 차트 (차트 기능 유지)
# =========================================================
with tab2:
    selected_ticker = st.selectbox("분석할 종목을 선택하세요", all_tickers, format_func=lambda x: f"{get_stock_name(x)} ({x})")
    df_chart = _df_cache.get(selected_ticker)
    
    if df_chart is not None and not df_chart.empty:
        df_plot = df_chart.tail(200).copy()
        df_plot['Buy_Signal'] = (df_plot['Close'] > df_plot['Entry_High']) & (df_plot['Close'].shift(1) <= df_plot['Entry_High'].shift(1))
        df_plot['Sell_Signal'] = (df_plot['Close'] < df_plot['Exit_Low']) & (df_plot['Close'].shift(1) >= df_plot['Exit_Low'].shift(1))
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='가격'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_50'], line=dict(color='orange', width=1.5), name='50일선'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_200'], line=dict(color='purple', width=2), name='200일선'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Entry_High'], line=dict(color='blue', dash='dot'), name='터틀 진입선'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Exit_Low'], line=dict(color='red', dash='dot'), name='터틀 청산선'))
        
        st.plotly_chart(fig, use_container_width=True)
