import numpy as np
import pandas as pd

from alpha_quat.backtest.position_sizing import STRATEGIES


def test_equal_weight():
    scores = np.array([0.1, 0.5, 0.9])
    result = STRATEGIES["equal"](scores)
    assert np.allclose(result, np.ones(3))


def test_kelly_adjust():
    scores = np.array([0.5, 0.5, 0.5])
    history = {
        "A": [0.4, 0.5, 0.6],
        "B": [0.1, 0.2],
        "C": [0.5, 0.5, 0.5, 0.5],
    }
    codes = ["A", "B", "C"]
    result = STRATEGIES["kelly"](scores, history, codes)
    assert len(result) == 3
    assert result[0] > 0  # positive mean/var ratio


def test_vol_parity():
    scores = np.array([1.0, 1.0])
    features = pd.DataFrame({"KLEN38_1": [0.5, 1.0]})
    result = STRATEGIES["vol_parity"](scores, features)
    assert len(result) == 2
    assert result[0] > result[1]  # lower vol -> higher weight


def test_score_momentum():
    scores = np.array([0.5, 0.5, 0.5])
    history = {"A": [0.4, 0.5, 0.6], "B": [], "C": [0.5]}
    codes = ["A", "B", "C"]
    result = STRATEGIES["score_momentum"](scores, history, codes)
    assert len(result) == 3
    assert result[0] > result[1]  # A has history, B doesn't
