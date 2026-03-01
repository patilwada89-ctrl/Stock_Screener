from src.signals import decision_from_production_score


def test_production_score_decision_thresholds_buy_hold_sell():
    assert decision_from_production_score(0.5, buy_threshold=0.3, sell_threshold=-0.2) == "Buy"
    assert decision_from_production_score(-0.3, buy_threshold=0.3, sell_threshold=-0.2) == "Sell"
    assert decision_from_production_score(0.1, buy_threshold=0.3, sell_threshold=-0.2) == "Hold"
