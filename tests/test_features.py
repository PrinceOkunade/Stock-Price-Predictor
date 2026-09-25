"""Offline tests for feature engineering (synthetic data, no network)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from feature_engineer import FEATURE_COLS, engineer_features, latest_feature_row


def make_ohlcv(n=200, start_price=100.0, seed=0):
    rng = np.random.default_rng(seed)
    close = start_price * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    return pd.DataFrame({
        "Date": pd.bdate_range("2024-01-01", periods=n),
        "Close": close,
        "High": np.maximum(open_, close) * 1.01,
        "Low": np.minimum(open_, close) * 0.99,
        "Open": open_,
        "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    })


def test_target_is_next_day_direction():
    df = make_ohlcv()
    data, _ = engineer_features(df)
    close = df.set_index("Date")["Close"]
    expected = (close.shift(-1) > close).astype(int).loc[data["Date"]]
    assert (data["target"].to_numpy() == expected.to_numpy()).all()


def test_training_rows_exclude_latest_bar():
    df = make_ohlcv()
    data, _ = engineer_features(df)
    assert data["Date"].max() == df["Date"].iloc[-2]


def test_live_prediction_uses_latest_bar():
    df = make_ohlcv()
    latest = latest_feature_row(df)
    assert len(latest) == 1
    assert latest["Date"].iloc[0] == df["Date"].iloc[-1]
    assert latest[FEATURE_COLS].notna().all(axis=None)


def test_features_are_scale_free():
    df = make_ohlcv()
    scaled = df.copy()
    scaled[["Open", "High", "Low", "Close"]] *= 50  # same path, 50x the price
    a, cols = engineer_features(df)
    b, _ = engineer_features(scaled)
    pd.testing.assert_frame_equal(a[cols], b[cols], rtol=1e-6)


def test_no_raw_price_levels_in_features():
    for raw in ["sma_5", "sma_20", "sma_50", "ema_12", "ema_26", "bb_high",
                "bb_low", "atr_14", "obv", "volume_sma_20", "macd"]:
        assert raw not in FEATURE_COLS


@pytest.mark.parametrize("col", ["target", "Close", "Open", "High", "Low", "Volume"])
def test_no_leaky_columns_in_features(col):
    assert col not in FEATURE_COLS
