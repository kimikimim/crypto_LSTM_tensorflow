import pandas as pd
import numpy as np

# --- 1. SMA (단순 이동평균선) 계산 ---

def add_sma(data, short_window=5, long_window=20):
    """데이터에 단기 및 장기 이동평균선을 추가합니다."""
    data_copy = data.copy()
    data_copy[f'SMA{short_window}'] = data_copy['Close'].rolling(window=short_window).mean()
    data_copy[f'SMA{long_window}'] = data_copy['Close'].rolling(window=long_window).mean()
    return data_copy

def get_sma_analysis(data):
    """이동평균선을 기반으로 간단한 분석 결과를 반환합니다."""
    
    # 1. 컬럼 존재 확인
    if 'SMA5' not in data.columns or 'SMA20' not in data.columns:
        return "➖ SMA 지표가 계산되지 않아 추세 분석이 어렵습니다. (데이터 처리 오류)"
    
    # 2. NaN 값 제거 및 유효 데이터 확보
    # SMA5, SMA20 컬럼만 복사한 후, NaN 값이 있는 행을 제거합니다.
    data_clean = data[['SMA5', 'SMA20']].dropna()

    if data_clean.empty or len(data_clean) < 2:
        return "➖ 데이터 부족 또는 지표 계산 불가로 추세 분석이 어렵습니다."

    last_row = data_clean.iloc[-1]
    prev_row = data_clean.iloc[-2]

    # 🌟🌟🌟 최종 수정: 모든 비교 값을 .item()으로 Scalar로 추출합니다 🌟🌟🌟
    last_sma5 = last_row['SMA5'].item()
    last_sma20 = last_row['SMA20'].item()
    prev_sma5 = prev_row['SMA5'].item()
    prev_sma20 = prev_row['SMA20'].item()
    # 🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟

    # 골든 크로스 & 데드 크로스 확인 (추출된 Scalar 값으로 비교)
    if last_sma5 > last_sma20 and prev_sma5 <= prev_sma20:
        return "📈 '골든 크로스' 발생! 단기적으로 상승 추세 전환 가능성이 있습니다."
    elif last_sma5 < last_sma20 and prev_sma5 >= prev_sma20:
        return "📉 '데드 크로스' 발생! 단기적으로 하락 추세 전환 가능성이 있습니다."

    # 현재 추세 확인
    if last_sma5 > last_sma20:
        return "🙂 단기 이동평균선이 장기 이동평균선 위에 있어 상승 추세를 유지하고 있습니다."
    else:
        return "😟 단기 이동평균선이 장기 이동평균선 아래에 있어 하락 추세를 유지하고 있습니다."


# --- 2. RSI, MACD, BB 등 기술적 지표 계산 ---

def add_technical_indicators(data, rsi_period=14, fast_period=12, slow_period=26, signal_period=9, bb_period=20, bb_std=2, atr_period=14, stoch_k=14, stoch_d=3):
    """데이터에 RSI, MACD, BB, ATR, Stochastics, OBV, CCI 지표를 추가합니다."""
    if data is None or data.empty:
        return None
    
    # 🌟🌟🌟 수정된 부분: 명시적으로 복사본을 만들어 SMA 컬럼을 보존하며 작업 🌟🌟🌟
    data_copy = data.copy()
    
    # RSI (Relative Strength Index) 계산
    delta = data_copy['Close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    avg_gain = up.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    avg_loss = down.abs().ewm(com=rsi_period - 1, min_periods=rsi_period).mean()

    rs = avg_gain / avg_loss
    data_copy['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD 계산
    data_copy['EMA_Fast'] = data_copy['Close'].ewm(span=fast_period, adjust=False).mean()
    data_copy['EMA_Slow'] = data_copy['Close'].ewm(span=slow_period, adjust=False).mean()
    data_copy['MACD'] = data_copy['EMA_Fast'] - data_copy['EMA_Slow']
    data_copy['MACD_Signal'] = data_copy['MACD'].ewm(span=signal_period, adjust=False).mean()
    data_copy['MACD_Hist'] = data_copy['MACD'] - data_copy['MACD_Signal']
    
    data_copy.drop(columns=['EMA_Fast', 'EMA_Slow'], inplace=True, errors='ignore')
    
     # 표준편차 계산 (Rolling Standard Deviation)
    data_copy['BB_Std'] = data_copy['Close'].rolling(window=bb_period).std()
    
    # Center Band: 20일 이동평균선 (add_sma에서 이미 계산되었을 수 있으나 안전하게 다시 계산)
    data_copy['BB_Middle'] = data_copy['Close'].rolling(window=bb_period).mean()
    
    # Upper Band: 중간선 + (표준편차 * N)
    data_copy['BB_Upper'] = data_copy['BB_Middle'] + (data_copy['BB_Std'] * bb_std)
    
    # Lower Band: 중간선 - (표준편차 * N)
    data_copy['BB_Lower'] = data_copy['BB_Middle'] - (data_copy['BB_Std'] * bb_std)
    
    # 계산에 사용된 임시 컬럼 제거 (MACD 계산의 임시 컬럼과 함께)
    data_copy.drop(columns=['EMA_Fast', 'EMA_Slow', 'BB_Std'], inplace=True, errors='ignore')
    
    # --- 3. 스토캐스틱 오실레이터 (Stochastic Oscillator) ---
    # %K = ((현재 종가 - K기간 중 최저가) / (K기간 중 최고가 - K기간 중 최저가)) * 100
    low_min = data_copy['Low'].rolling(window=stoch_k).min()
    high_max = data_copy['High'].rolling(window=stoch_k).max()
    
    data_copy['Stoch_%K'] = 100 * ((data_copy['Close'] - low_min) / (high_max - low_min))
    data_copy['Stoch_%D'] = data_copy['Stoch_%K'].rolling(window=stoch_d).mean()
    
    # --- 4. ATR (Average True Range) ---
    # True Range (TR) 계산: max[(H-L), abs(H-PC), abs(L-PC)]
    high_low = data_copy['High'] - data_copy['Low']
    high_close = abs(data_copy['High'] - data_copy['Close'].shift(1))
    low_close = abs(data_copy['Low'] - data_copy['Close'].shift(1))
    
    data_copy['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data_copy['ATR'] = data_copy['TR'].ewm(alpha=1/atr_period, adjust=False).mean()
    
    # --- 5. OBV (On-Balance Volume) ---
    # 가격이 오르면 거래량을 더하고, 가격이 내리면 거래량을 뺍니다.
    data_copy['OBV_Change'] = data_copy['Volume'].where(data_copy['Close'] > data_copy['Close'].shift(1), -data_copy['Volume'].where(data_copy['Close'] < data_copy['Close'].shift(1), 0))
    data_copy['OBV'] = data_copy['OBV_Change'].cumsum()
    
    # --- 6. CCI (Commodity Channel Index) ---
    # TP (Typical Price) 계산: (High + Low + Close) / 3
    data_copy['TP'] = (data_copy['High'] + data_copy['Low'] + data_copy['Close']) / 3
    # SMATP (Simple Moving Average of TP) 계산: TP의 20일 이동평균
    data_copy['SMATP'] = data_copy['TP'].rolling(window=bb_period).mean()
    # MD (Mean Deviation) 계산: |TP - SMATP|의 20일 이동평균
    data_copy['MD'] = abs(data_copy['TP'] - data_copy['SMATP']).rolling(window=bb_period).mean()
    
    data_copy['CCI'] = (data_copy['TP'] - data_copy['SMATP']) / (0.015 * data_copy['MD'])
    
    
    # 계산에 사용된 임시 컬럼 제거
    data_copy.drop(columns=['EMA_Fast', 'EMA_Slow', 'BB_Std', 'TR', 'OBV_Change', 'TP', 'SMATP', 'MD'], inplace=True, errors='ignore')
    
    return data_copy # 모든 지표와 SMA가 포함된 데이터 반환











def merge_sentiment_data(price_data, sentiment_data):
    if sentiment_data is None or sentiment_data.empty:
        price_data = price_data.copy()
        price_data['Sentiment_Score'] = 0.0
        return price_data

    # 1. 가격 데이터 처리
    price_data_reset = price_data.reset_index().copy()

    # MultiIndex 컬럼 평탄화
    if isinstance(price_data_reset.columns, pd.MultiIndex):
        price_data_reset.columns = [
            "_".join([str(c) for c in col if c]) for col in price_data_reset.columns
        ]

    # 첫 번째 컬럼을 날짜 컬럼으로 설정
    date_col = price_data_reset.columns[0]
    price_data_reset.rename(columns={date_col: 'ds'}, inplace=True)
    price_data_reset['ds'] = pd.to_datetime(price_data_reset['ds']).dt.normalize()

    # 2. 감성 데이터 처리
    sentiment_data = sentiment_data.copy()
    sentiment_data.rename(columns={'Date': 'ds'}, inplace=True)
    sentiment_data['ds'] = pd.to_datetime(sentiment_data['ds']).dt.normalize()

    # 3. 병합
    merged_data = pd.merge(
        price_data_reset,
        sentiment_data[['ds', 'Sentiment_Score']],
        on='ds',
        how='left'
    )

    # 4. NaN 값 보정
    merged_data['Sentiment_Score'] = merged_data['Sentiment_Score'].fillna(0.0)

    # 5. 인덱스 재설정
    merged_data = merged_data.set_index('ds')
    return merged_data.rename_axis('Date')


# modules/analysis.py 파일에 추가

def get_signal_summary(data):
    
    """ 주요 기술적 지표의 마지막 값을 기반으로 종합 매매 신호를 반환합니다. """

    data_clean = data.dropna() # NaN 값을 제거하고 데이터 프레임 확인
    
    # 유효한 행이 1개도 없으면 분석 불가 메시지 반환
    if data_clean.empty:
        return "➖ 데이터 부족 또는 지표 계산 불가", {}

    # NaN 값이 없는 유효한 마지막 행을 찾습니다.
    last_row = data.dropna().iloc[-1]
    
    if last_row.empty:
        return "➖ 데이터 부족 또는 지표 계산 불가", {}

    # 1. 각 지표별 신호 판단 (기준은 일반적인 설정입니다)
    signals = {}
    
    # 1. SMA (골든/데드 크로스 없이 단순 추세만)
    if last_row['SMA5'] > last_row['SMA20']:
        signals['SMA'] = '매수'  # 단기 > 장기
    elif last_row['SMA5'] < last_row['SMA20']:
        signals['SMA'] = '매도'  # 단기 < 장기
    else:
        signals['SMA'] = '중립'

    # 2. RSI (Relative Strength Index)
    # 70 이상: 과매수(매도 신호), 30 이하: 과매도(매수 신호)
    if last_row['RSI'] > 70:
        signals['RSI'] = '매도'
    elif last_row['RSI'] < 30:
        signals['RSI'] = '매수'
    else:
        signals['RSI'] = '중립'

    # 3. MACD (MACD 라인과 Signal 라인 비교)
    if last_row['MACD'] > last_row['MACD_Signal']:
        signals['MACD'] = '매수' # MACD > Signal (상승 모멘텀)
    elif last_row['MACD'] < last_row['MACD_Signal']:
        signals['MACD'] = '매도' # MACD < Signal (하락 모멘텀)
    else:
        signals['MACD'] = '중립'
        
    # 4. Stochastic Oscillator (%K와 %D 비교)
    # 80 이상: 과매수, 20 이하: 과매도
    if last_row['Stoch_%K'] > 80 and last_row['Stoch_%K'] < last_row['Stoch_%D']:
        signals['Stoch'] = '매도'
    elif last_row['Stoch_%K'] < 20 and last_row['Stoch_%K'] > last_row['Stoch_%D']:
        signals['Stoch'] = '매수'
    else:
        signals['Stoch'] = '중립'
        
    # 5. CCI (Commodity Channel Index)
    # +100 이상: 강한 매수, -100 이하: 강한 매도
    if last_row['CCI'] > 100:
        signals['CCI'] = '매수'
    elif last_row['CCI'] < -100:
        signals['CCI'] = '매도'
    else:
        signals['CCI'] = '중립'
    
    # --- 종합 신호 결정 (단순 다수결 방식) ---
    buy_count = list(signals.values()).count('매수')
    sell_count = list(signals.values()).count('매도')
    
    if buy_count > sell_count:
        final_signal = "✅ 강한 매수"
    elif sell_count > buy_count:
        final_signal = "❌ 강한 매도"
    else:
        final_signal = "➖ 중립 / 관망"
        
    return final_signal, signals

# modules/analysis.py

def normalize_columns(df):
    """
    데이터프레임의 컬럼 이름을 표준화합니다.
    MultiIndex나 ticker가 포함된 컬럼명도 처리합니다.
    """
    new_cols = []
    for col in df.columns:
        col_str = str(col).strip().lower()
        
        # '_' 로 split해서 첫 번째 부분만 사용 (예: 'close_btc-usd' -> 'close')
        col_base = col_str.split('_')[0]
        
        # 표준 컬럼명으로 매핑
        if col_base in ['adj close', 'adjclose', 'close', 'adj_close']:
            new_cols.append('Close')
        elif col_base == 'open':
            new_cols.append('Open')
        elif col_base == 'high':
            new_cols.append('High')
        elif col_base == 'low':
            new_cols.append('Low')
        elif col_base == 'volume':
            new_cols.append('Volume')
        else:
            # 표준 컬럼이 아닌 경우 첫 글자만 대문자로
            new_cols.append(col_str.capitalize())

    df.columns = new_cols
    return df
