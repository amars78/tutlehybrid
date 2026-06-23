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
st.set_page_config(page_title="CAN SLIM x 터틀 매니저", layout="wide")
st.title("🦅 CAN SLIM x 🐢 터틀 트레이딩 융합 매니저")
st.markdown("""
**CAN SLIM**의 추세·모멘텀·수급 기준으로 매수할 만한 종목인지 다각도로 평가하고,
**터틀 트레이딩**의 ATR 기반 자금관리·피라미딩·돌파/이탈 시스템으로 실전 진입·증액·청산을 관리합니다.
""")

if not PYKRX_AVAILABLE:
    st.warning("⚠️ `pykrx` 라이브러리가 설치되어 있지 않아 국내 종목명이 코드로만 표시됩니다. "
               "터미널에서 `pip install pykrx` 실행 후 다시 시작해주세요.")

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ 포트폴리오 및 시스템 설정")

# 1. 시스템 선택 (터틀)
system_type = st.sidebar.radio(
    "터틀 트레이딩 시스템 선택",
    ("시스템 1 (20일 고점/10일 저점)", "시스템 2 (55일 고점/20일 저점)")
)

if system_type == "시스템 1 (20일 고점/10일 저점)":
    entry_window, exit_window = 20, 10
else:
    entry_window, exit_window = 55, 20

st.sidebar.divider()

# 2. 포트폴리오 설정
tickers_input = st.sidebar.text_area(
    "관심 종목 리스트 (쉼표로 구분)",
    "AAPL, MSFT, NVDA, TSLA, 005930.KS",
    help="국내 종목은 .KS(코스피) 또는 .KQ(코스닥) 접미사를 붙여주세요. 예: 005930.KS, 035720.KQ"
)
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

# 3. 자금 관리 (Position Sizing) 설정
account_size = st.sidebar.number_input("총 투자 자본금 (예: 달러 또는 원)", value=100000, step=10000)
risk_per_trade = st.sidebar.slider("1회 거래당 리스크 비율 (%, 1유닛 기준)", min_value=0.5, max_value=5.0, value=1.0, step=0.1) / 100
max_units = st.sidebar.slider("최대 피라미딩 유닛 수", min_value=1, max_value=4, value=4,
                               help="터틀 원칙: 최초 진입 후 0.5N(ATR)마다 추가 매수, 최대 4유닛")

st.sidebar.divider()
st.sidebar.subheader("🌐 시장 방향성(M) 필터")
apply_market_filter = st.sidebar.checkbox(
    "시장이 하락추세면 매수등급 자동 하향", value=True,
    help="오닐의 CAN SLIM에서 M(시장방향)은 개별 종목 점수보다 우선합니다. 시장이 약세면 좋은 종목도 보수적으로 평가합니다."
)

st.sidebar.info("""
💡 **전략 융합 가이드**
* **CAN SLIM 점수(0~9점):** 미너비니 추세템플릿 7개 + RS Rating + 거래량 수급 신호. 7점 이상이면서 시장방향이 우호적일 때 매수 우선 고려.
* **터틀 관리:** 1유닛 진입 후 0.5N마다 추가 매수(최대 4유닛), 손절은 진입가 -2N 또는 채널 이탈선 중 먼저 닿는 쪽.
""")


# --- 종목명 조회 함수 ---
@st.cache_data(ttl=86400)
def get_stock_name(ticker: str) -> str:
    """국내 종목(.KS/.KQ)은 pykrx로 한글명, 해외 종목은 yfinance info로 이름 조회. 실패 시 티커 반환."""
    ticker = ticker.strip().upper()
    if PYKRX_AVAILABLE and (ticker.endswith(".KS") or ticker.endswith(".KQ")):
        code = ticker.split(".")[0]
        try:
            name = krx.get_market_ticker_name(code)
            if name and isinstance(name, str) and name.strip():
                return name
        except Exception:
            pass
        return ticker
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName")
        if name:
            return name
    except Exception:
        pass
    return ticker


@st.cache_data(ttl=3600)
def load_all_names(_tickers: tuple) -> dict:
    return {t: get_stock_name(t) for t in _tickers}


def get_benchmark_ticker(ticker: str) -> str:
    """종목 시장에 맞는 벤치마크 지수 티커 반환 (M: 시장방향 판단용)."""
    if ticker.endswith(".KS"):
        return "^KS11"   # 코스피
    elif ticker.endswith(".KQ"):
        return "^KQ11"   # 코스닥
    else:
        return "^GSPC"   # S&P500 (해외 종목 기본값)


@st.cache_data(ttl=3600)
def load_price_history(ticker: str, days: int = 500) -> pd.DataFrame | None:
    """범용 가격 히스토리 로더 (종목/지수 공통 사용)."""
    start = datetime.today() - timedelta(days=days)
    try:
        df = yf.download(ticker, start=start, end=datetime.today(), progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_market_trend(benchmark_ticker: str) -> dict:
    """
    M(시장방향) 판단: 벤치마크 지수가 50일/200일선 위에 있고 200일선이 상승 중인지 확인.
    return: {"status": "상승추세"/"하락추세"/"중립", "detail": str}
    """
    df = load_price_history(benchmark_ticker, days=400)
    if df is None or len(df) < 200:
        return {"status": "확인불가", "detail": "지수 데이터 부족"}

    close = df["Close"]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    sma200_prev = close.rolling(200).mean().iloc[-20] if len(df) > 220 else sma200
    last_close = close.iloc[-1]

    above_both = (last_close > sma50) and (last_close > sma200)
    sma200_rising = sma200 > sma200_prev

    if above_both and sma200_rising:
        return {"status": "🟢 상승추세", "detail": f"지수가 50/200일선 위, 200일선 상승 중"}
    elif (not above_both) and (not sma200_rising):
        return {"status": "🔴 하락추세", "detail": f"지수가 주요 이평선 아래, 200일선 하락 중"}
    else:
        return {"status": "🟡 중립/전환구간", "detail": f"이평선 신호가 혼재된 상태"}


def compute_rs_raw(df: pd.DataFrame) -> float | None:
    """
    오닐식 RS(Relative Strength) 원점수 계산.
    최근 1분기(63일) 수익률에 2배 가중치, 이전 3개 분기(63일씩) 동일 가중치.
    RS_raw = 2*Q1 + Q2 + Q3 + Q4   (점수가 클수록 강한 모멘텀)
    """
    close = df["Close"].dropna()
    n = len(close)
    if n < 253:  # 최소 약 1년+1일 데이터 필요
        return None

    def period_return(end_idx, start_idx):
        try:
            end_p = close.iloc[end_idx]
            start_p = close.iloc[start_idx]
            if start_p == 0 or pd.isna(start_p) or pd.isna(end_p):
                return 0.0
            return (end_p / start_p) - 1.0
        except Exception:
            return 0.0

    q1 = period_return(-1, -64)     # 최근 ~1분기
    q2 = period_return(-64, -127)
    q3 = period_return(-127, -190)
    q4 = period_return(-190, -253)

    return 2 * q1 + q2 + q3 + q4


def rs_rating_from_raw(raw_scores: dict) -> dict:
    """포트폴리오 내 RS_raw 값을 1~99 백분위로 환산 (IBD RS Rating 흉내)."""
    valid = {k: v for k, v in raw_scores.items() if v is not None}
    if len(valid) <= 1:
        return {k: (50 if v is not None else None) for k, v in raw_scores.items()}

    values = sorted(valid.values())
    ratings = {}
    for k, v in raw_scores.items():
        if v is None:
            ratings[k] = None
            continue
        rank = sum(1 for x in values if x <= v) / len(values)  # 0~1
        ratings[k] = max(1, min(99, round(rank * 98) + 1))
    return ratings


def compute_volume_signal(df: pd.DataFrame, window: int = 50) -> dict:
    """
    최근 window일간 상승일 거래량 합 vs 하락일 거래량 합 비율 (수급 매집/분산 신호).
    ratio > 1.2 → 매집(수급 우호), < 0.8 → 분산(수급 비우호)
    """
    recent = df.tail(window).copy()
    recent["change"] = recent["Close"].diff()
    up_vol = recent.loc[recent["change"] > 0, "Volume"].sum()
    down_vol = recent.loc[recent["change"] < 0, "Volume"].sum()

    if down_vol == 0:
        ratio = float("inf") if up_vol > 0 else 1.0
    else:
        ratio = up_vol / down_vol

    if ratio == float("inf") or ratio >= 1.2:
        signal = "🟢 매집"
    elif ratio <= 0.8:
        signal = "🔴 분산"
    else:
        signal = "🟡 중립"

    ratio_display = "∞" if ratio == float("inf") else round(ratio, 2)
    return {"ratio": ratio_display, "signal": signal}


@st.cache_data(ttl=3600)
def get_eps_growth(ticker: str) -> str:
    """최근 분기 EPS YoY 성장률 (best-effort, 데이터 없으면 'N/A'). 점수에는 반영하지 않는 참고 지표."""
    try:
        tk = yf.Ticker(ticker)
        earnings = tk.quarterly_earnings  # index: 분기, columns: Revenue, Earnings
        if earnings is not None and len(earnings) >= 5 and "Earnings" in earnings.columns:
            latest = earnings["Earnings"].iloc[-1]
            year_ago = earnings["Earnings"].iloc[-5]
            if year_ago and year_ago != 0:
                growth = (latest - year_ago) / abs(year_ago) * 100
                return f"{growth:+.1f}%"
        return "N/A"
    except Exception:
        return "N/A"


# --- 데이터 처리 함수 (종목 자체 OHLCV + 기술지표) ---
@st.cache_data(ttl=3600)
def load_and_process_data(ticker, entry_w, exit_w):
    # RS Rating(최근 1년 분기수익률) 계산을 위해 500일 데이터 확보
    df = load_price_history(ticker, days=500)
    if df is None:
        return None
    try:
        # [CAN SLIM 기술적 템플릿 지표]
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_150'] = df['Close'].rolling(window=150).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['52W_High'] = df['High'].rolling(window=250).max()
        df['52W_Low'] = df['Low'].rolling(window=250).min()
        df['SMA_200_Trend'] = df['SMA_200'] > df['SMA_200'].shift(20)

        # [터틀 지표]
        df['Entry_High'] = df['High'].rolling(window=entry_w).max().shift(1)
        df['Exit_Low'] = df['Low'].rolling(window=exit_w).min().shift(1)

        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['TR'] = np.max(ranges, axis=1)
        df['ATR'] = df['TR'].rolling(window=20).mean()

        return df
    except Exception:
        return None


# 종목명 일괄 조회
name_map = load_all_names(tuple(tickers))

# 종목별 RS_raw 사전 계산 (포트폴리오 내 상대순위 산출용)
_raw_cache: dict[str, float | None] = {}
_df_cache: dict[str, pd.DataFrame | None] = {}
for t in tickers:
    _df = load_and_process_data(t, entry_window, exit_window)
    _df_cache[t] = _df
    _raw_cache[t] = compute_rs_raw(_df) if _df is not None else None
rs_ratings = rs_rating_from_raw(_raw_cache)

# 필요한 벤치마크 지수들의 시장방향 미리 계산 (중복 호출 방지)
benchmark_needed = sorted(set(get_benchmark_ticker(t) for t in tickers))
market_trend_map = {b: get_market_trend(b) for b in benchmark_needed}

# --- 탭 구성 ---
tab1, tab2 = st.tabs(["📊 CAN SLIM x 터틀 포트폴리오", "📈 개별 종목 융합 차트"])

# ==========================================
# 탭 1: 포트폴리오 대시보드
# ==========================================
with tab1:
    st.subheader("🌐 시장 방향성(M) 현황")
    mcols = st.columns(len(market_trend_map) if market_trend_map else 1)
    bench_names = {"^KS11": "코스피(KOSPI)", "^KQ11": "코스닥(KOSDAQ)", "^GSPC": "S&P500"}
    for col, (bench, info) in zip(mcols, market_trend_map.items()):
        with col:
            st.metric(bench_names.get(bench, bench), info["status"])
            st.caption(info["detail"])

    st.divider()
    st.subheader("종목별 캔슬림(추세·RS·수급) 평가 및 터틀 관리 상태")
    if st.button("포트폴리오 데이터 새로고침"):
        st.cache_data.clear()

    portfolio_data = []
    my_bar = st.progress(0, text="종목 데이터를 분석 중입니다...")

    for i, ticker in enumerate(tickers):
        stock_name = name_map.get(ticker, ticker)
        df = _df_cache.get(ticker)
        my_bar.progress((i + 1) / len(tickers), text=f"{stock_name}({ticker}) 분석 중...")

        if df is not None and not df.empty:
            latest = df.iloc[-1]
            c_price = latest['Close']

            # --- 1) 미너비니 추세템플릿 7개 조건 ---
            cond1 = (c_price > latest['SMA_150']) and (c_price > latest['SMA_200'])
            cond2 = latest['SMA_150'] > latest['SMA_200']
            cond3 = bool(latest['SMA_200_Trend'])
            cond4 = (latest['SMA_50'] > latest['SMA_150']) and (latest['SMA_50'] > latest['SMA_200'])
            cond5 = c_price > latest['SMA_50']
            cond6 = c_price >= (latest['52W_High'] * 0.75)
            cond7 = c_price >= (latest['52W_Low'] * 1.30)
            trend_score = sum([cond1, cond2, cond3, cond4, cond5, cond6, cond7])

            # --- 2) RS Rating 조건 (70 이상이면 가점) ---
            rs_val = rs_ratings.get(ticker)
            cond_rs = (rs_val is not None) and (rs_val >= 70)

            # --- 3) 거래량 수급 조건 (매집 신호면 가점) ---
            vol_info = compute_volume_signal(df)
            cond_vol = vol_info["signal"] == "🟢 매집"

            canslim_score = trend_score + int(cond_rs) + int(cond_vol)  # 0~9점

            # --- 4) 시장방향(M) 필터 적용 ---
            bench = get_benchmark_ticker(ticker)
            market_status = market_trend_map.get(bench, {}).get("status", "확인불가")
            market_bearish = "하락추세" in market_status

            if canslim_score >= 7:
                base_status, base_emoji = "강력 매수 고려", "🔥"
            elif canslim_score >= 5:
                base_status, base_emoji = "관망/대기", "🟡"
            else:
                base_status, base_emoji = "추세 약함", "❄️"

            if apply_market_filter and market_bearish and base_status == "강력 매수 고려":
                canslim_status = f"🟠 시장약세로 보류 ({canslim_score}/9)"
            else:
                canslim_status = f"{base_emoji} {base_status} ({canslim_score}/9)"

            # --- 터틀 상태 및 자금 관리 (2N 손절 + 채널 이탈 병행) ---
            entry_price = latest['Entry_High']
            exit_price = latest['Exit_Low']
            atr = latest['ATR']
            stop_2n = entry_price - 2 * atr if not pd.isna(entry_price) and not pd.isna(atr) else np.nan

            if not pd.isna(exit_price) and c_price < exit_price:
                turtle_status = "🔴 보유불가 (채널 이탈)"
            elif not pd.isna(stop_2n) and c_price < stop_2n:
                turtle_status = "🔴 보유불가 (2N 손절)"
            elif not pd.isna(entry_price) and c_price > entry_price:
                turtle_status = "🟢 신규 돌파"
            else:
                turtle_status = "🔵 보유/유지 구간"

            risk_amount = account_size * risk_per_trade
            unit_shares = int(risk_amount / atr) if not pd.isna(atr) and atr > 0 else 0

            eps_growth = get_eps_growth(ticker)

            portfolio_data.append({
                "종목명": stock_name,
                "티커": ticker,
                "CAN SLIM 점수(9점)": canslim_status,
                "RS Rating": rs_val if rs_val is not None else "N/A",
                "수급 신호": vol_info["signal"],
                "EPS성장(YoY,참고)": eps_growth,
                "터틀 관리 상태": turtle_status,
                "현재가": round(c_price, 2),
                "진입선(채널)": round(entry_price, 2) if not pd.isna(entry_price) else "-",
                "청산선(채널)": round(exit_price, 2) if not pd.isna(exit_price) else "-",
                "2N 손절가": round(stop_2n, 2) if not pd.isna(stop_2n) else "-",
                "ATR(20)": round(atr, 2) if not pd.isna(atr) else "-",
                "1유닛 수량": f"{unit_shares} 주"
            })
        else:
            portfolio_data.append({
                "종목명": stock_name, "티커": ticker,
                "CAN SLIM 점수(9점)": "⚠️ 데이터 조회 실패", "RS Rating": "-", "수급 신호": "-",
                "EPS성장(YoY,참고)": "-", "터틀 관리 상태": "-", "현재가": "-",
                "진입선(채널)": "-", "청산선(채널)": "-", "2N 손절가": "-", "ATR(20)": "-", "1유닛 수량": "-"
            })

    my_bar.empty()
    if portfolio_data:
        port_df = pd.DataFrame(portfolio_data)
        st.dataframe(port_df, use_container_width=True, hide_index=True)
        st.caption("RS Rating은 입력한 포트폴리오 내 상대 순위(1~99)이며, 전체 시장 기준 IBD RS Rating과는 다릅니다. "
                   "EPS 성장률은 데이터 소스 한계로 참고용이며 점수에 반영되지 않습니다.")
    else:
        st.warning("데이터를 불러올 수 없습니다. 종목 심볼을 확인해주세요.")

# ==========================================
# 탭 2: 개별 종목 융합 차트
# ==========================================
with tab2:
    st.subheader("추세(이동평균) 및 터틀(돌파/이탈/피라미딩) 정밀 차트")

    display_options = [f"{name_map.get(t, t)} ({t})" for t in tickers]
    label_to_ticker = {f"{name_map.get(t, t)} ({t})": t for t in tickers}

    selected_label = st.selectbox("분석할 종목을 선택하세요", display_options)
    selected_ticker = label_to_ticker[selected_label]
    selected_name = name_map.get(selected_ticker, selected_ticker)

    df_chart = _df_cache.get(selected_ticker)

    if df_chart is not None and not df_chart.empty:
        df_plot = df_chart.tail(200).copy()

        df_plot['Buy_Signal'] = (df_plot['Close'] > df_plot['Entry_High']) & (df_plot['Close'].shift(1) <= df_plot['Entry_High'].shift(1))
        df_plot['Sell_Signal'] = (df_plot['Close'] < df_plot['Exit_Low']) & (df_plot['Close'].shift(1) >= df_plot['Exit_Low'].shift(1))

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='가격'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_50'], line=dict(color='orange', width=1.5), name='50일 이평선'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_200'], line=dict(color='purple', width=2), name='200일 이평선'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Entry_High'], line=dict(color='rgba(0, 0, 255, 0.4)', dash='dot'), name='터틀 진입선'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Exit_Low'], line=dict(color='rgba(255, 0, 0, 0.4)', dash='dot'), name='터틀 청산선'))

        buy_signals = df_plot[df_plot['Buy_Signal']]
        sell_signals = df_plot[df_plot['Sell_Signal']]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=13), name='터틀 돌파(매수)'))
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=13), name='터틀 이탈(청산)'))

        fig.update_layout(title=f'{selected_name} ({selected_ticker}) — CAN SLIM x 터틀 분석 차트',
                           xaxis_rangeslider_visible=False, template='plotly_white', height=600)
        st.plotly_chart(fig, use_container_width=True)

        # --- RS / 수급 / 시장방향 요약 ---
        st.subheader("📌 CAN SLIM 보강 지표 요약")
        rs_val = rs_ratings.get(selected_ticker)
        vol_info = compute_volume_signal(df_chart)
        bench = get_benchmark_ticker(selected_ticker)
        m_info = market_trend_map.get(bench, {"status": "확인불가", "detail": ""})
        c1, c2, c3 = st.columns(3)
        c1.metric("RS Rating (포트폴리오 내 순위)", rs_val if rs_val is not None else "N/A")
        c2.metric("거래량 수급 신호 (50일)", f"{vol_info['signal']} (비율 {vol_info['ratio']})")
        c3.metric("시장 방향성 (M)", m_info["status"])

        # --- 터틀 피라미딩 / 2N 손절 테이블 ---
        st.subheader("🐢 터틀 피라미딩 & 손절 계획")
        latest = df_chart.iloc[-1]
        atr = latest['ATR']
        entry_price = latest['Entry_High']

        if pd.isna(atr) or pd.isna(entry_price) or atr <= 0:
            st.info("ATR 또는 진입선 데이터가 부족해 피라미딩 계획을 계산할 수 없습니다.")
        else:
            risk_amount = account_size * risk_per_trade
            unit_shares = int(risk_amount / atr)
            stop_2n = entry_price - 2 * atr

            plan_rows = []
            for unit_n in range(1, max_units + 1):
                add_price = entry_price + 0.5 * atr * (unit_n - 1)
                plan_rows.append({
                    "유닛": f"{unit_n}유닛",
                    "추가매수 기준가": round(add_price, 2),
                    "유닛당 수량": f"{unit_shares} 주",
                    "누적 수량": f"{unit_shares * unit_n} 주",
                })
            plan_df = pd.DataFrame(plan_rows)
            st.dataframe(plan_df, use_container_width=True, hide_index=True)
            st.caption(
                f"최초 진입가(채널 돌파) 기준 약 {round(entry_price, 2)}, "
                f"0.5N(ATR={round(atr, 2)})마다 1유닛씩 추가, 최대 {max_units}유닛. "
                f"전체 포지션 공통 손절가(2N) ≈ **{round(stop_2n, 2)}** "
                f"(또는 채널 이탈선 {round(latest['Exit_Low'], 2) if not pd.isna(latest['Exit_Low']) else '-'} 중 먼저 닿는 쪽)."
            )
    else:
        st.warning(f"{selected_name}({selected_ticker})의 데이터를 불러올 수 없습니다. 티커를 확인해주세요.")
