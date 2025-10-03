import pandas as pd
import numpy as np

def run_sma_backtest(data, short_window=5, long_window=20, initial_capital=10000):
    """
    SMA 크로스오버 전략을 기반으로 백테스팅을 실행하고 수익률을 계산합니다.
    (SMA5 > SMA20 이면 매수 신호)
    """
    
    # 분석에 필요한 컬럼 이름 확인
    required_cols = [f'SMA{short_window}', f'SMA{long_window}', 'Close']
    df = data.dropna(subset=required_cols).copy()
    
    if len(df) < long_window:
        return {'error': "데이터가 너무 짧아 백테스팅을 실행할 수 없습니다."}

    # 1. 신호 생성: 단기SMA가 장기SMA를 상향 돌파하면 1(매수), 하향 돌파하면 -1(매도)
    # 단기SMA > 장기SMA 일 때 포지션 1 (매수)
    df['Signal'] = np.where(df[f'SMA{short_window}'] > df[f'SMA{long_window}'], 1.0, 0.0)
    
    # 이전 날짜와 비교하여 포지션 변화를 포착 (실제 매매 시점을 위해 사용)
    df['Position'] = df['Signal'].diff()
    
    # 2. 투자 수익률 계산
    
    # 일일 수익률 계산 (가격 변동률)
    df['Returns'] = df['Close'].pct_change()
    
    # 포지션(1은 매수, 0은 중립)이 하루 전에 발생했다고 가정하고 일일 수익률에 적용
    # shift(1)은 포지션이 다음 날부터 적용됨을 의미
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift(1)
    
    # 3. 누적 수익률 계산
    
    # 초기 자본금에 기반한 전략 자산 가치
    df['Cumulative_Strategy_Value'] = (1 + df['Strategy_Returns']).cumprod() * initial_capital
    
    # 벤치마크 (코인 단순 보유) 수익률
    df['Cumulative_Benchmark_Value'] = (1 + df['Returns']).cumprod() * initial_capital
    
    # 결과 요약
    final_strategy_value = df['Cumulative_Strategy_Value'].iloc[-1]
    final_benchmark_value = df['Cumulative_Benchmark_Value'].iloc[-1]
    
    # 초기 자본금과 비교하여 최종 수익률 계산
    strategy_profit = (final_strategy_value / initial_capital - 1) * 100
    benchmark_profit = (final_benchmark_value / initial_capital - 1) * 100
    
    return {
        'strategy_profit': strategy_profit,
        'benchmark_profit': benchmark_profit,
        'cumulative_values': df[['Cumulative_Strategy_Value', 'Cumulative_Benchmark_Value']]
    }
