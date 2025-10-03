import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# 🌟 modules.crypto에서 모든 필수 항목과 COIN_LIST를 임포트합니다.
from modules.crypto import get_crypto_price, get_crypto_history, COIN_LIST 

st.set_page_config(page_title="Crypto Predictor", page_icon="📈", layout="wide")
st.title("📈 가상화폐 뉴스 & 시세 분석 대시보드")

# --- 뉴스 기능 ---
analyzer = SentimentIntensityAnalyzer()
@st.cache_data(ttl=300)
def get_news():
    # 실제 코인 관련 뉴스 소스로 변경 권장 (현재는 BBC 더미)
    url = "https://feeds.bbci.co.uk/news/world/rss.xml" 
    try:
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(res.content)
        items = root.findall(".//item")
        
        news_list = []
        for item in items[:5]:
            title = item.find("title").text
            link = item.find("link").text
            sentiment = analyzer.polarity_scores(title)["compound"]
            news_list.append({"title": title, "link": link, "sentiment": sentiment})
        return news_list
    except Exception as e:
        return []

# --- 미니 차트 생성 함수 ---
def create_mini_chart(df, coin_name):
    """지정된 기간의 종가 미니 차트를 생성합니다."""
    if df is None or df.empty:
        return go.Figure()
    
    # 🌟 컬럼 이름 표준화 (MultiIndex 및 대소문자 문제 해결)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(str(c) for c in col).strip('_') if isinstance(col, tuple) else str(col) for col in df.columns]
        
    col_mapping = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if 'close' in col_lower:
            col_mapping[col] = 'Close'
    
    df = df.rename(columns=col_mapping)

    # 종가 데이터만 사용
    if 'Close' not in df.columns:
        return go.Figure()
        
    fig = go.Figure(data=[
        go.Scatter(x=df.index, y=df['Close'], mode='lines', 
                   line=dict(color='lightgreen', width=2))
    ])
    
    fig.update_layout(
        title=f"7일 추이",
        xaxis_visible=False,
        yaxis_visible=False,
        showlegend=False,
        height=100,
        margin=dict(l=0, r=0, t=20, b=0),
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig

# --- 화면 구성 ---

st.subheader("💰 주요 가상화폐 시세 및 추이")
# COIN_LIST의 길이에 맞춰 컬럼 생성
coin_cols = st.columns(len(COIN_LIST)) 

with st.spinner("시세 및 추이 정보를 가져오는 중..."):
    # COIN_LIST의 항목들을 순회하며 가격과 차트를 표시
    for idx, (name, data) in enumerate(COIN_LIST.items()):
        price = get_crypto_price(data["coingecko"])
        history = get_crypto_history(data["yfinance"], period="7d") 
        
        with coin_cols[idx]:
            st.metric(name, 
                      f"${price:,.2f}" if price else "데이터 없음",
                      delta_color="normal")
            
            if history is not None and not history.empty:
                # 고유 키를 사용하여 Chart ID 중복 오류를 방지합니다.
                st.plotly_chart(create_mini_chart(history, name), 
                               use_container_width=True, 
                               key=f"mini_chart_{data['yfinance']}") 
            else:
                st.caption("차트 데이터 없음")


st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.info("💡 자세한 분석, 예측, 백테스팅은 왼쪽 메뉴에서 'Coin Detail' 페이지를 선택하세요.")

with col2:
    st.subheader("📰 최신 뉴스 & 감성 분석")
    with st.spinner("뉴스를 분석하는 중..."):
        news_list = get_news()

    if not news_list:
        st.warning("뉴스를 불러오는 데 실패했습니다.")
    else:
        for news in news_list:
            sentiment_label = "긍정 😊" if news["sentiment"] > 0.1 else "부정 😡" if news["sentiment"] < -0.1 else "중립 😐"
            st.write(f"[{news['title']}]({news['link']}) — **{sentiment_label}**")

st.caption("데이터는 주기적으로 자동 업데이트됩니다.")
