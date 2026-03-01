from src.signals import normalize_score


def test_normalize_score_is_bounded_to_minus1_plus1():
    assert -1.0 <= normalize_score([1, 1, 1, 1, 1, 1, 1]) <= 1.0
    assert -1.0 <= normalize_score([-1, -1, -1, -1, -1, -1, -1]) <= 1.0
    assert -1.0 <= normalize_score([-1, 0, 1, 0, -1, 1, 0]) <= 1.0
