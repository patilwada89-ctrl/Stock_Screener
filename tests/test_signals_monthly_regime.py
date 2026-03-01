from src.signals import classify_monthly_regime


def test_monthly_regime_bull_bear_neutral():
    assert classify_monthly_regime(close_t=110, ema20_t=100, ema20_t_3=95) == "Bull"
    assert classify_monthly_regime(close_t=90, ema20_t=100, ema20_t_3=105) == "Bear"
    assert classify_monthly_regime(close_t=101, ema20_t=100, ema20_t_3=102) == "Neutral"
