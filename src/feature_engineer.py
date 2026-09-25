"""Module 2: Engineers technical indicator features from OHLCV data."""
import numpy as np
import pandas as pd
import ta

# Model inputs. Every feature is scale-free (a return, ratio, percentage or
# bounded oscillator) so the model transfers across tickers and price regimes.
# Raw price-level indicators (sma_*, bb_high, macd, ...) are still computed
# below for charts, but are deliberately NOT fed to the model.
FEATURE_COLS = [
    "returns_1d", "returns_5d", "returns_10d", "log_return",
    "dist_sma_5", "dist_sma_20", "dist_sma_50", "sma_cross",
    "rsi_14", "stoch_k",
    "macd_pct", "macd_signal_pct", "macd_diff_pct",
    "bb_pband", "bb_width", "atr_pct",
    "volume_ratio", "obv_trend",
]


def compute_features(df):
    """Compute indicators and target for every row, including the latest one.

    The final row's target is NaN because tomorrow's close is not known yet.
    """
    data = df.copy()

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # --- Price features ---
    data["returns_1d"] = close.pct_change(1)
    data["returns_5d"] = close.pct_change(5)
    data["returns_10d"] = close.pct_change(10)
    data["log_return"] = np.log(close / close.shift(1))

    # --- Moving averages (raw levels kept for charts) ---
    data["sma_5"] = close.rolling(5).mean()
    data["sma_20"] = close.rolling(20).mean()
    data["sma_50"] = close.rolling(50).mean()
    data["ema_12"] = close.ewm(span=12).mean()
    data["ema_26"] = close.ewm(span=26).mean()
    data["dist_sma_5"] = close / data["sma_5"] - 1
    data["dist_sma_20"] = close / data["sma_20"] - 1
    data["dist_sma_50"] = close / data["sma_50"] - 1
    data["sma_cross"] = (data["sma_5"] > data["sma_20"]).astype(int)

    # --- Momentum indicators (ta library) ---
    data["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd_obj = ta.trend.MACD(close)
    data["macd"] = macd_obj.macd()
    data["macd_signal"] = macd_obj.macd_signal()
    data["macd_diff"] = macd_obj.macd_diff()
    data["macd_pct"] = data["macd"] / close
    data["macd_signal_pct"] = data["macd_signal"] / close
    data["macd_diff_pct"] = data["macd_diff"] / close
    data["stoch_k"] = ta.momentum.StochasticOscillator(high, low, close).stoch()

    # --- Volatility indicators ---
    bb = ta.volatility.BollingerBands(close)
    data["bb_high"] = bb.bollinger_hband()
    data["bb_low"] = bb.bollinger_lband()
    data["bb_pband"] = bb.bollinger_pband()
    data["bb_width"] = (data["bb_high"] - data["bb_low"]) / data["sma_20"]
    data["atr_14"] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
    data["atr_pct"] = data["atr_14"] / close

    # --- Volume indicators ---
    data["volume_sma_20"] = volume.rolling(20).mean()
    data["volume_ratio"] = volume / data["volume_sma_20"]
    data["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    # Net signed volume over 10 days as a share of total volume, in [-1, 1]
    data["obv_trend"] = data["obv"].diff(10) / volume.rolling(10).sum()

    # --- Target: did next day's close go UP? (NaN on the last row) ---
    next_close = close.shift(-1)
    data["target"] = (next_close > close).astype(float).where(next_close.notna())

    return data


def engineer_features(df):
    """Return labelled rows for training/evaluation, plus the feature columns."""
    data = compute_features(df)
    data = data.dropna(subset=FEATURE_COLS + ["target"]).copy()
    data["target"] = data["target"].astype(int)

    feature_cols = list(FEATURE_COLS)
    print(f"Shape: {data.shape}")
    print(f"Features engineered: {len(feature_cols)}")
    print(f"Target distribution:\n{data['target'].value_counts()}")
    return data, feature_cols


def latest_feature_row(df):
    """Return the most recent row (target still unknown) for a live prediction."""
    data = compute_features(df).dropna(subset=FEATURE_COLS)
    return data.iloc[[-1]]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_fetcher import fetch_stock_data

    df = fetch_stock_data("AAPL", period="2y")
    data, feat_cols = engineer_features(df)
    print(f"\nFeature columns ({len(feat_cols)}):")
    for f in feat_cols:
        print(f"  - {f}")
    print(f"\nSample:\n{data[feat_cols].head()}")
    print(f"\nLatest row for live prediction: {latest_feature_row(df)['Date'].iloc[0].date()}")
