import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# tensorflow 자체가 무거워서 주석 처리 시간부자들만 하는걸 추천 pytorch버전은 추후 업데이트 
#from tensorflow.keras.models import Sequential, load_model  
#from tensorflow.keras.layers import LSTM, Dense, Dropout  
import os

# 모델 저장 경로
MODEL_PATH = 'lstm_model.h5' 

# 예측에 사용할 데이터 컬럼 목록
FEATURES = ['Close', 'Volume', 
            'SMA5', 'SMA20', 'RSI', 'MACD', 'MACD_Signal', 
            'BB_Upper', 'BB_Middle', 'BB_Lower', 
            'Stoch_%K', 'Stoch_%D', 'ATR', 'OBV', 'CCI', 
            'Sentiment_Score'] 

LOOKBACK_DAYS = 60

def create_dataset(data, lookback):
    """LSTM 학습을 위해 시계열 데이터를 시퀀스 형태로 변환합니다."""
    X, Y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

def train_and_save_model(X_train, Y_train, units=50):
    """LSTM 모델을 정의하고 학습 후 저장합니다."""
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 학습
    model.fit(X_train, Y_train, epochs=1, batch_size=32, verbose=0)
    
    # 모델 저장
    try:
        model.save(MODEL_PATH)
        print(f"모델 저장 완료: {MODEL_PATH}")
    except Exception as e:
        print(f"모델 저장 실패: {e}")
        
    return model

def get_future_price_prediction(data: pd.DataFrame, days_to_predict=5):
    """
    주어진 과거 데이터를 기반으로 향후 N일의 시세를 예측하고 결과를 반환합니다.
    """
    if data is None or len(data) < LOOKBACK_DAYS + 1:
        return "데이터 부족", []
    
    feature_data = data[FEATURES].values
    
    # 1. 스케일러 학습 및 데이터 정규화
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(feature_data)
    
    # 2. 모델 로드 또는 학습
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
        except Exception as e:
            print(f"모델 로드 실패 ({e}). 재학습합니다.")
            X_train, Y_train = create_dataset(scaled_data[:-days_to_predict], LOOKBACK_DAYS)
            model = train_and_save_model(X_train, Y_train)
    else:
        X_train, Y_train = create_dataset(scaled_data[:-days_to_predict], LOOKBACK_DAYS)
        model = train_and_save_model(X_train, Y_train)
        
    # 3. 향후 N일 예측
    current_input = scaled_data[-LOOKBACK_DAYS:].copy()
    future_predictions = []

    for _ in range(days_to_predict):
        X_input = current_input[np.newaxis, -LOOKBACK_DAYS:, :]
        predicted_scaled_price = model.predict(X_input, verbose=0)[0]
        future_predictions.append(predicted_scaled_price[0])
        
        # 시퀀스 업데이트
        current_input = np.roll(current_input, -1, axis=0) 
        current_input[-1, 0] = predicted_scaled_price[0]
        
    # 4. 예측 값 역정규화
    prediction_array = np.zeros((days_to_predict, len(FEATURES)))
    prediction_array[:, 0] = future_predictions
    predicted_prices = scaler.inverse_transform(prediction_array)[:, 0]
    
    return "✅ 예측 완료", predicted_prices.tolist()