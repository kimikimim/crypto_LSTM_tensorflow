import streamlit as st
import pandas as pd
from datetime import date, timedelta
import plotly.graph_objects as go
import numpy as np


# 모듈 불러오기
from modules.crypto import get_crypto_history
from modules.analysis import add_sma, add_technical_indicators, get_sma_analysis, merge_sentiment_data
from modules.analysis import add_sma, add_technical_indicators, get_sma_analysis, merge_sentiment_data, get_signal_summary
from modules.view import get_candlestick_chart
from modules.prediction import get_future_price_prediction # 수정된 예측 함수
from modules.analysis import add_sma, add_technical_indicators, get_sma_analysis
from modules.analysis import normalize_columns

# 페이지 설정
st.set_page_config(page_title="Coin Detail", page_icon="📈", layout="wide")
st.title("🪙 코인 상세 분석 및 예측")

# --- 1. 사이드바: 사용자 입력 설정 ---
st.sidebar.header("설정")

# 코인 선택 (yfinance 심볼 기준)
coin_options = {"Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD"}
selected_name = st.sidebar.selectbox("코인 선택", list(coin_options.keys()))
selected_symbol = coin_options[selected_name]

# 과거 데이터 기간 설정
period_options = {"1개월": "1mo", "3개월": "3mo", "6개월": "6mo", "1년": "1y"}
selected_period_name = st.sidebar.selectbox("과거 데이터 기간", list(period_options.keys()))
selected_period = period_options[selected_period_name]

# 예측 기간 설정
days_to_predict = st.sidebar.slider("미래 예측 기간 (일)", 1, 7, 3)

# --- 2. 데이터 가져오기 및 분석 ---

data_load_state = st.info(f"{selected_name} ({selected_period}) 데이터를 불러오는 중...")
# crypto.py의 get_crypto_history는 이미 캐싱되어 있습니다.
price_data = get_crypto_history(selected_symbol, period=selected_period, interval="1d")
data_load_state.empty()



if price_data is None:
    st.error("데이터를 불러오는 데 실패했습니다. 심볼 또는 기간을 확인해주세요.")
else:
    # 🌟🌟🌟 최종 수정: MultiIndex 문제 해결 및 컬럼 정규화 🌟🌟🌟
    
    # 1. MultiIndex 컬럼을 평탄화
    if isinstance(price_data.columns, pd.MultiIndex):
        # MultiIndex인 경우: 튜플을 문자열로 변환
        price_data.columns = ['_'.join(str(c) for c in col).strip('_') if isinstance(col, tuple) else str(col) for col in price_data.columns]
    
    # 2. normalize_columns 함수를 사용하여 컬럼 이름 표준화
    price_data = normalize_columns(price_data)
    
    # 3. 필수 컬럼 확인
    required_cols = ['Close', 'Volume', 'Open', 'High', 'Low']
    missing_cols = [col for col in required_cols if col not in price_data.columns]
    
    if missing_cols:
        st.error(f"데이터에 필수 컬럼이 없습니다: {missing_cols}")
        st.write("현재 컬럼:", list(price_data.columns))
        st.stop()
    
    # 2-1. 기술적 지표 추가
    processed_data = price_data.copy()
    
    # SMA 추가
    processed_data = add_sma(processed_data) 
    
    # 3. RSI/MACD/BB 추가
    final_features_data = add_technical_indicators(processed_data, bb_period=20, bb_std=2)

    
    # 🌟🌟🌟 새로 추가할 부분: 감성 데이터 통합 🌟🌟🌟
    st.info("과거 뉴스 감성 데이터를 생성 및 병합 중...")
    
    # (주의: 실제 앱에서는 이 더미 데이터를 실제 과거 감성 데이터로 대체해야 합니다.)
    # 더미 감성 데이터 생성 (날짜만 맞추고 점수는 무작위)
    start_date = final_features_data.index.min().date()
    end_date = final_features_data.index.max().date()
    sentiment_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    dummy_sentiment = pd.DataFrame({
        'Date': sentiment_dates,
        'Sentiment_Score': np.random.uniform(-0.5, 0.5, size=len(sentiment_dates))
    })
    
    # 감성 데이터 병합
    final_features_data = merge_sentiment_data(final_features_data, dummy_sentiment)
    
    # 2-2. SMA 분석 결과 표시
    st.subheader("📊 기술적 분석 요약")
    st.markdown(f"**이동평균선 추세:** {get_sma_analysis(final_features_data)}")
    
    st.subheader("📢 종합 매매 신호")

    # 신호 함수 호출
    final_signal, detail_signals = get_signal_summary(final_features_data)

    # 최종 신호를 크게 표시
    st.markdown(f"### **종합 신호:** {final_signal}")

    # 상세 지표 신호를 표로 표시
    st.markdown("---")
    st.markdown("##### 지표별 상세 신호")

    signal_df = pd.DataFrame(list(detail_signals.items()), columns=['지표', '신호'])
    st.dataframe(signal_df, use_container_width=True, hide_index=True)
    
    
    
    # --- 3. 시세 예측 (캐싱 적용) ---
    
    @st.cache_data(ttl=3600) # 1시간(3600초) 동안 예측 결과를 캐시하여 속도를 개선합니다.
    def cached_prediction(data_hash, days):
        """
        Streamlit 캐시를 적용하기 위한 래퍼 함수입니다.
        데이터프레임 자체가 아닌, 데이터의 해시값을 인자로 사용합니다.
        """
        # 해시 대신 실제 데이터프레임을 get_future_price_prediction 함수에 전달
        return get_future_price_prediction(data_hash, days)

    st.subheader(f"🔮 {days_to_predict}일 미래 시세 예측")
    
    with st.spinner("LSTM 모델로 시세 예측 중... (첫 실행 시 모델 학습으로 인해 시간이 걸릴 수 있습니다.)"):
        # final_features_data를 직접 캐시에 전달
        prediction_status, predicted_prices = cached_prediction(final_features_data, days_to_predict)

    st.success(prediction_status)
    
    # --- 4. 시각화 및 예측 결과 표시 ---
    
    # 예측된 날짜 생성
    last_date = final_features_data.index[-1].date()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    
    # 4-1. 예측 결과 표
    if predicted_prices:
        col_p1, col_p2 = st.columns([1, 3])
        
        with col_p1:
            st.markdown("**예측 결과 (종가 기준)**")
            
            prediction_df = pd.DataFrame({
                '날짜': future_dates,
                '예측 종가 (USD)': [f"${p:,.2f}" for p in predicted_prices]
            })
            st.dataframe(prediction_df.set_index('날짜'), use_container_width=True)

        with col_p2:
            st.caption("예측 모델은 학습된 과거 패턴에 기반하며, 실제 시장 상황과 다를 수 있습니다.")

        # 4-2. 차트에 예측 결과 추가
        # view.py의 get_candlestick_chart 함수를 사용
        fig = get_candlestick_chart(final_features_data, selected_name)
        
        # 예측 선 추가
        # 마지막 종가와 예측 시작점을 연결
        historical_close = final_features_data['Close'].iloc[-1]
        
        prediction_trace = go.Scatter(
            x=[final_features_data.index[-1]] + future_dates,
            y=[historical_close] + predicted_prices,
            mode='lines+markers',
            name=f'{days_to_predict}일 예측',
            line=dict(color='yellow', width=2, dash='dot'),
            marker=dict(size=6)
        )
        fig.add_trace(prediction_trace)
        
        # 차트 출력
        st.plotly_chart(fig, use_container_width=True)

    # --- 5. 상세 데이터 (디버깅/참고용) ---
    st.subheader("📚 상세 데이터 (기술적 지표 포함)")
    st.dataframe(final_features_data.tail(30))