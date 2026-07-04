import pytest

from src.signals import production_score_from_components

_ZERO = {
    "RSI14_State": 0,
    "RSI_Accel": 0,
    "MACD_Hist_Sign": 0,
    "MACD_Hist_Accel": 0,
    "Price_vs_EMA20": 0,
    "Volume_Confirm": 0,
    "Volatility_Expansion": 0,
}

_MOMENTUM = ("RSI14_State", "RSI_Accel", "MACD_Hist_Sign", "MACD_Hist_Accel")


def _components(**overrides):
    out = dict(_ZERO)
    out.update(overrides)
    return out


def test_all_positive_is_one():
    assert production_score_from_components(_components(**{k: 1 for k in _ZERO})) == 1.0


def test_all_negative_is_minus_one():
    assert production_score_from_components(_components(**{k: -1 for k in _ZERO})) == -1.0


def test_each_family_carries_one_quarter():
    # Momentum family (4 signals) collectively equals one 25% family.
    assert production_score_from_components(
        _components(**{k: 1 for k in _MOMENTUM})
    ) == pytest.approx(0.25)
    assert production_score_from_components(_components(Price_vs_EMA20=1)) == pytest.approx(0.25)
    assert production_score_from_components(_components(Volume_Confirm=1)) == pytest.approx(0.25)
    assert production_score_from_components(
        _components(Volatility_Expansion=1)
    ) == pytest.approx(0.25)


def test_single_momentum_signal_is_one_sixteenth():
    assert production_score_from_components(_components(RSI14_State=1)) == pytest.approx(1 / 16)


def test_momentum_no_longer_dominates_single_signal_families():
    # Full bullish momentum, but the three other families bearish -> net negative.
    score = production_score_from_components(
        _components(
            RSI14_State=1,
            RSI_Accel=1,
            MACD_Hist_Sign=1,
            MACD_Hist_Accel=1,
            Price_vs_EMA20=-1,
            Volume_Confirm=-1,
            Volatility_Expansion=-1,
        )
    )
    assert score == pytest.approx(-0.5)
