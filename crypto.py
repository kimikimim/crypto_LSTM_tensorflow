import yfinance as yf
import requests
import streamlit as st

# 🌟🌟🌟 코인 목록을 여기서 정의하고 다른 파일에서 공유합니다. 🌟🌟🌟
COIN_LIST = {
    "Bitcoin (BTC)": {"coingecko": "bitcoin", "yfinance": "BTC-USD"},
    "Ethereum (ETH)": {"coingecko": "ethereum", "yfinance": "ETH-USD"},
    "Solana (SOL)": {"coingecko": "solana", "yfinance": "SOL-USD"},
    "Ripple (XRP)": {"coingecko": "ripple", "yfinance": "XRP-USD"},
    "Dogecoin (DOGE)": {"coingecko": "dogecoin", "yfinance": "DOGE-USD"},
    "Cardano (ADA)": {"coingecko": "cardano", "yfinance": "ADA-USD"},
    "Polkadot (DOT)": {"coingecko": "polkadot", "yfinance": "DOT-USD"},
}
# 🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟

@st.cache_data(ttl=60)
def get_crypto_price(symbol="bitcoin"):
    """CoinGecko API를 사용해 현재 코인 가격을 가져옵니다."""
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd"
    try:
        res = requests.get(url, timeout=5)
        res.raise_for_status()  
        return res.json().get(symbol, {}).get("usd", None)
    except requests.exceptions.RequestException as e:
        st.error(f"API 요청 실패: {e}")
        return None

@st.cache_data(ttl=300)
def get_crypto_history(symbol, period="1mo", interval="1d"):
    """yfinance를 사용해 코인의 과거 시세 데이터를 가져옵니다."""
    data = yf.download(symbol, period=period, interval=interval)
    if data.empty:
        st.warning("데이터를 불러오지 못했습니다. 기간을 변경해 보세요.")
        return None
    return data
