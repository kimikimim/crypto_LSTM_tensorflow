# crypto_LSTM_tensorflow (KOREAN VERSION)

# 💹 Crypto Price Prediction using LSTM (TensorFlow) 

A deep learning project that predicts cryptocurrency price trends using **LSTM (Long Short-Term Memory)** neural networks built with TensorFlow.  
The project includes full data preprocessing, training, prediction, backtesting, and visualization pipelines.

---

## 🚀 Features

- 📊 **Data Analysis & Preprocessing** — Cleans and normalizes coin price data and related news.
- 🧠 **LSTM Model** — Predicts future cryptocurrency prices using time series analysis.
- 🔁 **Backtesting System** — Evaluates model performance on historical data.
- 🌐 **Web Dashboard** — Flask-based web interface for viewing predictions and analytics.
- 📰 **News Integration** — Incorporates crypto news sentiment to improve accuracy.
- ⚙️ **Customizable Config** — Easy to adapt to any cryptocurrency dataset.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `1_Coin_Detail.py` | Handles coin metadata and feature extraction. |
| `analysis.py` | Performs EDA (Exploratory Data Analysis) and visualization. |
| `app.py` | Main Flask app for running the prediction dashboard. |
| `backtest.py` | Backtesting engine to evaluate prediction accuracy. |
| `crypto.py` | Core TensorFlow LSTM model training and inference logic. |
| `prediction.py` | Loads trained model and generates forecasts. |
| `view.py` | Defines visualization and result display functions. |
| `news_cache.json` | Cached crypto news data for faster loading. |
| `__init__.py` | Package initializer. |

---

## 🧮 Model Overview

The project uses an **LSTM (Long Short-Term Memory)** network, a type of recurrent neural network (RNN) ideal for time-series forecasting.

**Model workflow:**
1. Load and preprocess historical crypto data.  
2. Train an LSTM model to learn sequential patterns.  
3. Evaluate accuracy using MSE/RMSE metrics.  
4. Generate future price predictions and visualize trends.

---

## 📊 Example Output

| Metric | Value |
|:-------|:------|
| Training Loss | 0.0023 |
| Validation Loss | 0.0029 |
| RMSE | 0.045 |

Visualization example:

```bash
Predicted vs Actual Bitcoin Prices
──────────────────────────────────────────────
🟢 Predicted: follows real trend with small lag
🔵 Actual: price curve from test dataset
