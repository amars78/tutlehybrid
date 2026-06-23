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
**CAN SLIM**의 강력한 추세 판단 기준(기술적 모멘텀)으로 매수할 만한 종목인지 필터링하고, 
**터틀 트레이딩**의 ATR 자금 관리와 돌파/이탈 시스템으로 실전 매수 수량과 청산 시기를 관리합니다.
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
risk_per_trade = st.sidebar.slider("1회 거래당 리스크 비율 (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1) / 100

st.sidebar.info("""
💡 **전략 융합 가이드**
* **CAN SLIM 점수 (0~7점):** 주가가 200일선 위에 있는지, 52주 신고가 근처인지 등 7가지 기술적 상승 추세 조건을 만족하는지 평가합니다. (6점 이상 매수 권장)
* **터틀 관리:** 점수가 높아 매수하기로 했다면, 터틀의 1 Unit 수량만큼 매수하고 '청산 기준선'을 이탈할 때까지 보유합니다.
""")


# --- 종목명 조회 함수 (신규 추가) ---
@st.cache_data(ttl=86400)  # 종목명은 자주 안 바뀌므로 하루 캐시
def get_stock_name(ticker: str) -> str:
    """
    티커를 받아 종목명을 반환한다.
    - 국내 종목(.KS/.KQ): pykrx로 KRX 공식 한글 종목명 조회
    - 해외 종목: yfinance info에서 shortName/longName 조회
    실패 시 티커 자체를 이름으로 반환한다.
    """
    ticker = ticker.strip().upper()

    # 1. 국내 종목 (코스피/코스닥)
    if PYKRX_AVAILABLE and (ticker.endswith(".KS") or ticker.endswith(".KQ")):
        code = ticker.split(".")[0]
        try:
            name = krx.get_market_ticker_name(code)
            if name and isinstance(name, str) and name.strip():
                return name
        except Exception:
            pass
        return ticker  # 조회 실패 시 티커 그대로

    # 2. 해외(또는 pykrx 미설치 시) 종목 → yfinance info 사용
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
    """여러 티커의 이름을 한 번에 조회해서 dict로 반환 (캐시 효율화)"""
    return {t: get_stock_name(t) for t in _tickers}


# --- 데이터 처리 함수 ---
@st.cache_data(ttl=3600)
def load_and_process_data(ticker, entry_w, exit_w):
    # 200일 이동평균선과 52주 신고가를 구하기 위해 넉넉히 400일 데이터 확보
    start = datetime.today() - timedelta(days=400)
    try:
        df = yf.download(ticker, start=start, end=datetime.today(), progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # [CAN SLIM 기술적 템플릿 지표]
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_150'] = df['Close'].rolling(window=150).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['52W_High'] = df['High'].rolling(window=250).max()
        df['52W_Low'] = df['Low'].rolling(window=250).min()

        # 200일선 추세(1개월 전 대비 상승)
        df['SMA_200_Trend'] = df['SMA_200'] > df['SMA_200'].shift(20)

        # [터틀 지표]
        df['Entry_High'] = df['High'].rolling(window=entry_w).max().shift(1)
        df['Exit_Low'] = df['Low'].rolling(window=exit_w).min().shift(1)

        # ATR 계산
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['TR'] = np.max(ranges, axis=1)
        df['ATR'] = df['TR'].rolling(window=20).mean()

        return df
    except Exception:
        return None


# 종목명 일괄 조회 (탭1, 탭2 공통 사용)
name_map = load_all_names(tuple(tickers))

# --- 탭 구성 ---
tab1, tab2 = st.tabs(["📊 CAN SLIM x 터틀 포트폴리오", "📈 개별 종목 융합 차트"])

# ==========================================
# 탭 1: 포트폴리오 대시보드
# ==========================================
with tab1:
    st.subheader("종목별 캔슬림(추세) 평가 및 터틀 관리 상태")
    if st.button("포트폴리오 데이터 새로고침"):
        st.cache_data.clear()

    portfolio_data = []
    my_bar = st.progress(0, text="종목 데이터를 분석 중입니다...")

    for i, ticker in enumerate(tickers):
        stock_name = name_map.get(ticker, ticker)
        df = load_and_process_data(ticker, entry_window, exit_window)
        my_bar.progress((i + 1) / len(tickers), text=f"{stock_name}({ticker}) 분석 중...")

        if df is not None and not df.empty:
            latest = df.iloc[-1]
            c_price = latest['Close']

            # --- CAN SLIM 기술적 조건 판별 (마크 미너비니 템플릿) ---
            # 1. 현재가가 150일, 200일 이평선 위에 있는가?
            cond1 = (c_price > latest['SMA_150']) and (c_price > latest['SMA_200'])
            # 2. 150일 이평선이 200일 이평선 위에 있는가?
            cond2 = latest['SMA_150'] > latest['SMA_200']
            # 3. 200일 이평선이 상승 추세인가?
            cond3 = latest['SMA_200_Trend']
            # 4. 50일 이평선이 150일, 200일 이평선 위에 있는가?
            cond4 = (latest['SMA_50'] > latest['SMA_150']) and (latest['SMA_50'] > latest['SMA_200'])
            # 5. 현재가가 50일 이평선 위에 있는가?
            cond5 = c_price > latest['SMA_50']
            # 6. 현재가가 52주 신고가 대비 25% 이내에 있는가?
            cond6 = c_price >= (latest['52W_High'] * 0.75)
            # 7. 현재가가 52주 신저가 대비 최소 30% 이상인가?
            cond7 = c_price >= (latest['52W_Low'] * 1.30)

            canslim_score = sum([cond1, cond2, cond3, cond4, cond5, cond6, cond7])

            if canslim_score >= 6:
                canslim_status = f"🔥 강력 매수 고려 ({canslim_score}/7)"
            elif canslim_score >= 4:
                canslim_status = f"🟡 관망/대기 ({canslim_score}/7)"
            else:
                canslim_status = f"❄️ 추세 약함 ({canslim_score}/7)"

            # --- 터틀 상태 및 자금 관리 ---
            entry_price = latest['Entry_High']
            exit_price = latest['Exit_Low']
            atr = latest['ATR']

            if c_price < exit_price:
                turtle_status = "🔴 보유불가 (청산 이탈)"
            elif c_price > entry_price:
                turtle_status = "🟢 신규 돌파"
            else:
                turtle_status = "🔵 보유/유지 구간"

            risk_amount = account_size * risk_per_trade
            unit_shares = int(risk_amount / atr) if not pd.isna(atr) and atr > 0 else 0

            portfolio_data.append({
                "티커": ticker,
                "종목명": stock_name,
                "CAN SLIM 추세 점수": canslim_status,
                "터틀 관리 상태": turtle_status,
                "현재가": round(c_price, 2),
                "터틀 진입선 (고점)": round(entry_price, 2) if not pd.isna(entry_price) else "-",
                "터틀 청산선 (저점)": round(exit_price, 2) if not pd.isna(exit_price) else "-",
                "현재 ATR(20)": round(atr, 2) if not pd.isna(atr) else "-",
                "적정 매수 수량(1 Unit)": f"{unit_shares} 주"
            })
        else:
            # 데이터를 못 가져온 경우에도 종목명/티커는 표시해 어떤 종목이 실패했는지 알 수 있게 함
            portfolio_data.append({
                "티커": ticker,
                "종목명": stock_name,
                "CAN SLIM 추세 점수": "⚠️ 데이터 조회 실패",
                "터틀 관리 상태": "-",
                "현재가": "-",
                "터틀 진입선 (고점)": "-",
                "터틀 청산선 (저점)": "-",
                "현재 ATR(20)": "-",
                "적정 매수 수량(1 Unit)": "-"
            })

    my_bar.empty()
    if portfolio_data:
        port_df = pd.DataFrame(portfolio_data)
        # 종목명이 맨 앞에 보이도록 컬럼 순서 정리
        cols = ["종목명", "티커"] + [c for c in port_df.columns if c not in ("종목명", "티커")]
        port_df = port_df[cols]
        st.dataframe(port_df, use_container_width=True, hide_index=True)
    else:
        st.warning("데이터를 불러올 수 없습니다. 종목 심볼을 확인해주세요.")

# ==========================================
# 탭 2: 개별 종목 융합 차트
# ==========================================
with tab2:
    st.subheader("추세(이동평균) 및 터틀(돌파/이탈) 정밀 차트")

    # 셀렉트박스에 "종목명 (티커)" 형태로 표시하되, 내부 값은 티커를 그대로 사용
    display_options = [f"{name_map.get(t, t)} ({t})" for t in tickers]
    label_to_ticker = {f"{name_map.get(t, t)} ({t})": t for t in tickers}

    selected_label = st.selectbox("분석할 종목을 선택하세요", display_options)
    selected_ticker = label_to_ticker[selected_label]
    selected_name = name_map.get(selected_ticker, selected_ticker)

    df_chart = load_and_process_data(selected_ticker, entry_window, exit_window)

    if df_chart is not None and not df_chart.empty:
        # 가독성을 위해 최근 200일 데이터만 차트에 표시
        df_plot = df_chart.tail(200).copy()

        df_plot['Buy_Signal'] = (df_plot['Close'] > df_plot['Entry_High']) & (df_plot['Close'].shift(1) <= df_plot['Entry_High'].shift(1))
        df_plot['Sell_Signal'] = (df_plot['Close'] < df_plot['Exit_Low']) & (df_plot['Close'].shift(1) >= df_plot['Exit_Low'].shift(1))

        fig = go.Figure()

        # 1. 캔들스틱
        fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='가격'))

        # 2. 캔슬림 핵심 이평선 (50일, 200일)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_50'], line=dict(color='orange', width=1.5), name='50일 이평선 (중기추세)'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_200'], line=dict(color='purple', width=2), name='200일 이평선 (장기추세)'))

        # 3. 터틀 기준선 (진입, 청산)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Entry_High'], line=dict(color='rgba(0, 0, 255, 0.4)', dash='dot'), name='터틀 진입선'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Exit_Low'], line=dict(color='rgba(255, 0, 0, 0.4)', dash='dot'), name='터틀 청산/손절선'))

        # 4. 시그널 마커
        buy_signals = df_plot[df_plot['Buy_Signal']]
        sell_signals = df_plot[df_plot['Sell_Signal']]

        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=13), name='터틀 돌파(매수)'))
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=13), name='터틀 이탈(청산)'))

        fig.update_layout(title=f'{selected_name} ({selected_ticker}) — CAN SLIM x 터틀 분석 차트', xaxis_rangeslider_visible=False, template='plotly_white', height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"{selected_name}({selected_ticker})의 데이터를 불러올 수 없습니다. 티커를 확인해주세요.")
