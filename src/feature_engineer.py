"""Module 2: Engineers technical indicator features from OHLCV data."""
import numpy as np
import pandas as pd
import ta


def engineer_features(df):
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

    # --- Moving averages ---
    data["sma_5"] = close.rolling(5).mean()
    data["sma_20"] = close.rolling(20).mean()
    data["sma_50"] = close.rolling(50).mean()
    data["ema_12"] = close.ewm(span=12).mean()
    data["ema_26"] = close.ewm(span=26).mean()
    data["sma_cross"] = (data["sma_5"] > data["sma_20"]).astype(int)

    # --- Momentum indicators (ta library) ---
    data["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd_obj = ta.trend.MACD(close)
    data["macd"] = macd_obj.macd()
    data["macd_signal"] = macd_obj.macd_signal()
    data["macd_diff"] = macd_obj.macd_diff()
    data["stoch_k"] = ta.momentum.StochasticOscillator(high, low, close).stoch()

    # --- Volatility indicators ---
    bb = ta.volatility.BollingerBands(close)
    data["bb_high"] = bb.bollinger_hband()
    data["bb_low"] = bb.bollinger_lband()
    data["bb_width"] = (data["bb_high"] - data["bb_low"]) / data["sma_20"]
    data["atr_14"] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()

    # --- Volume indicators ---
    data["volume_sma_20"] = volume.rolling(20).mean()
    data["volume_ratio"] = volume / data["volume_sma_20"]
    data["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    # --- Target: did next day's close go UP? ---
    data["target"] = (close.shift(-1) > close).astype(int)

    data = data.dropna()
    data = data.iloc[:-1]  # drop last row (target is unknown)

    feature_cols = [c for c in data.columns if c not in ["Date", "Open", "High", "Low", "Close", "Volume", "target"]]
    print(f"Shape: {data.shape}")
    print(f"Features engineered: {len(feature_cols)}")
    print(f"Target distribution:\n{data['target'].value_counts()}")
    return data, feature_cols


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
