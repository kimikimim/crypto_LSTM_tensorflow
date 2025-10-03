import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def get_candlestick_chart(data, coin_name):
    """Plotly를 사용하여 캔들스틱, 이동평균선, 볼린저 밴드, MACD, RSI 차트를 생성합니다."""
    if data is None or data.empty:
        return go.Figure()

    # 지표를 계산한 후 데이터의 유효한 행만 남깁니다.
    data_clean = data.dropna(subset=['Close', 'SMA5', 'SMA20', 'RSI', 'MACD'])

    if data_clean.empty:
        return go.Figure().update_layout(title="데이터가 부족하거나 지표 계산 불가")

    # 3개의 행 (캔들스틱, MACD, RSI)을 갖는 서브플롯 생성
    # 행 높이 비율 지정 (캔들:MACD:RSI = 3:1:1)
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.08,
                        row_heights=[0.5, 0.25, 0.25])

    # --- 1. 캔들스틱 및 추세 지표 (Row 1) ---
    
    # 캔들스틱 차트 추가
    fig.add_trace(go.Candlestick(
        x=data_clean.index,
        open=data_clean["Open"],
        high=data_clean["High"],
        low=data_clean["Low"],
        close=data_clean["Close"],
        name=coin_name,
        increasing_line_color='red',
        decreasing_line_color='blue'
    ), row=1, col=1)

    # 이동평균선 추가
    if 'SMA5' in data_clean.columns:
        fig.add_trace(go.Scatter(x=data_clean.index, y=data_clean['SMA5'], mode='lines', name='SMA 5', line=dict(color='orange', width=1.5)), row=1, col=1)
    if 'SMA20' in data_clean.columns:
        fig.add_trace(go.Scatter(x=data_clean.index, y=data_clean['SMA20'], mode='lines', name='SMA 20', line=dict(color='purple', width=1.5)), row=1, col=1)
        
    # 볼린저 밴드 오버레이 추가
    if 'BB_Upper' in data_clean.columns:
        fig.add_trace(go.Scatter(x=data_clean.index, y=data_clean['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='cyan', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data_clean.index, y=data_clean['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='cyan', width=1)), row=1, col=1)


    # --- 2. MACD (Row 2) ---
    if 'MACD' in data_clean.columns:
        # MACD 히스토그램 (막대 그래프)
        fig.add_trace(go.Bar(x=data_clean.index, 
                             y=data_clean['MACD_Hist'], 
                             name='MACD Hist',
                             marker_color=data_clean['MACD_Hist'].apply(lambda x: 'red' if x >= 0 else 'blue')), row=2, col=1)
        
        # MACD 라인
        fig.add_trace(go.Scatter(x=data_clean.index, y=data_clean['MACD'], mode='lines', name='MACD', line=dict(color='white', width=1.5)), row=2, col=1)
        # Signal 라인
        fig.add_trace(go.Scatter(x=data_clean.index, y=data_clean['MACD_Signal'], mode='lines', name='MACD Signal', line=dict(color='yellow', width=1)), row=2, col=1)

    # --- 3. RSI (Row 3) ---
    if 'RSI' in data_clean.columns:
        fig.add_trace(go.Scatter(x=data_clean.index, y=data_clean['RSI'], mode='lines', name='RSI', line=dict(color='lightgreen', width=1.5)), row=3, col=1)
        
        # 과매수/과매도 기준선 추가 (70, 30)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="blue", row=3, col=1)


    # --- 전체 레이아웃 설정 ---
    fig.update_layout(
        title=f"{coin_name} 가격 및 지표 분석",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=900, # 차트 높이 확대
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # X축 설정 (공유 X축 설정)
    fig.update_xaxes(
        tickvals=[], # X축 레이블 제거
        row=1, col=1, 
        showticklabels=False # 캔들 차트의 X축 레이블 숨기기
    )
    fig.update_xaxes(
        showticklabels=False, 
        row=2, col=1
    )
    fig.update_xaxes(
        showticklabels=True, 
        row=3, col=1,
        title_text="날짜"
    )

    # Y축 설정
    fig.update_yaxes(title_text="가격/추세", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100]) # RSI는 0~100 고정

    return fig