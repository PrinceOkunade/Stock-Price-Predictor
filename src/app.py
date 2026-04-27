"""Stock Price Direction Predictor — Streamlit App with Live Data."""
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent
MODELS_DIR = SRC_DIR.parent / "models"
sys.path.insert(0, str(SRC_DIR))

from data_fetcher import fetch_stock_data, get_latest_price
from feature_engineer import engineer_features

st.set_page_config(page_title="Stock Direction Predictor", page_icon="📈", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "stock_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    explainer = shap.TreeExplainer(model)
    return model, scaler, feature_names, explainer


def main():
    st.title("📈 Stock Price Direction Predictor")
    st.markdown(
        "Predicts whether tomorrow's closing price goes **UP** or **DOWN** using "
        "live market data from Yahoo Finance + **XGBoost** + **SHAP** explanations."
    )
    st.info("🟢 **LIVE DATA** — predictions update with real market data")

    # --- Sidebar ---
    st.sidebar.header("Stock Selection")
    popular = {"AAPL": "Apple", "GOOGL": "Google", "MSFT": "Microsoft",
               "AMZN": "Amazon", "TSLA": "Tesla", "NVDA": "NVIDIA"}
    st.sidebar.markdown("**Quick pick:**")
    cols = st.sidebar.columns(3)
    selected_quick = None
    for i, (sym, name) in enumerate(popular.items()):
        if cols[i % 3].button(sym, key=f"btn_{sym}", use_container_width=True):
            selected_quick = sym

    default_ticker = selected_quick or "AAPL"
    ticker = st.sidebar.text_input("Ticker symbol", value=default_ticker).upper().strip()
    period = st.sidebar.selectbox("Training period", ["1y", "2y", "5y"], index=1)

    st.sidebar.header("Model Options")
    retrain = st.sidebar.checkbox("Retrain model with latest data")
    if retrain:
        st.sidebar.warning("Retraining takes 2–5 minutes")

    predict = st.sidebar.button("Predict", type="primary", use_container_width=True)

    if not predict:
        st.markdown("Select a stock in the sidebar and click **Predict**.")
        return

    # --- Fetch live data ---
    try:
        with st.spinner(f"Fetching live data for {ticker}..."):
            df = fetch_stock_data(ticker, period=period)
        if len(df) < 60:
            st.error(f"Not enough data for {ticker}. Try a longer period or a different ticker.")
            return
    except Exception:
        st.error(f"Invalid ticker symbol: **{ticker}**. Please enter a valid stock ticker.")
        return

    # --- Section 1: Live market data ---
    st.header(f"📊 {ticker} — Live Market Data")
    latest_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    day_change = (latest_close - prev_close) / prev_close * 100
    year_high = float(df["High"].tail(252).max()) if len(df) >= 252 else float(df["High"].max())
    year_low = float(df["Low"].tail(252).min()) if len(df) >= 252 else float(df["Low"].min())

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Close", f"${latest_close:.2f}")
    c2.metric("Day Change", f"{day_change:+.2f}%")
    c3.metric("52-Week Range", f"${year_low:.2f} – ${year_high:.2f}")

    recent = df.tail(60).copy()
    fig_candle = go.Figure(data=[
        go.Candlestick(
            x=recent["Date"], open=recent["Open"], high=recent["High"],
            low=recent["Low"], close=recent["Close"], name="Price"),
        go.Bar(x=recent["Date"], y=recent["Volume"], name="Volume",
               marker_color="rgba(100,100,200,0.3)", yaxis="y2"),
    ])
    fig_candle.update_layout(
        yaxis=dict(title="Price ($)", side="left"),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        xaxis_rangeslider_visible=False, height=450,
        title=f"{ticker} — Last 60 Trading Days", legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_candle, use_container_width=True)
    st.caption(f"Data fetched live from Yahoo Finance at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Feature engineering ---
    with st.spinner("Engineering features..."):
        data, feature_cols = engineer_features(df)

    # --- Section 2: Technical indicators ---
    with st.expander("📉 Current Technical Indicators"):
        last_row = data.iloc[-1]
        indicators = pd.DataFrame({
            "Indicator": ["RSI (14)", "MACD", "MACD Signal", "Bollinger Width",
                          "ATR (14)", "SMA Cross (5/20)", "Volume Ratio"],
            "Value": [
                f"{last_row['rsi_14']:.1f}",
                f"{last_row['macd']:.4f}",
                f"{last_row['macd_signal']:.4f}",
                f"{last_row['bb_width']:.4f}",
                f"{last_row['atr_14']:.4f}",
                "Bullish" if last_row["sma_cross"] == 1 else "Bearish",
                f"{last_row['volume_ratio']:.2f}x",
            ],
            "Signal": [
                "Oversold (Bullish)" if last_row["rsi_14"] < 30 else ("Overbought (Bearish)" if last_row["rsi_14"] > 70 else "Neutral"),
                "Bullish" if last_row["macd"] > last_row["macd_signal"] else "Bearish",
                "",
                "High Volatility" if last_row["bb_width"] > 0.1 else "Low Volatility",
                "",
                "Bullish" if last_row["sma_cross"] == 1 else "Bearish",
                "High Volume" if last_row["volume_ratio"] > 1.5 else ("Low Volume" if last_row["volume_ratio"] < 0.5 else "Normal"),
            ],
        })
        st.dataframe(indicators, use_container_width=True, hide_index=True)

    # --- Retrain or load model ---
    if retrain:
        with st.spinner(f"Retraining model on {ticker} ({period})... this takes a few minutes."):
            from trainer import retrain_pipeline
            best_model, scaler, fnames, results, X_train, X_test, y_train, y_test = retrain_pipeline(
                ticker, period, models_path=str(MODELS_DIR) + "/"
            )
            explainer = shap.TreeExplainer(best_model)
            feature_names = fnames
    else:
        try:
            model, scaler, feature_names, explainer = load_artifacts()
            best_model = model
        except FileNotFoundError:
            st.error(
                "Model files not found. Check the **Retrain model** option in the sidebar "
                "or run `notebooks/stock_prediction.ipynb` first."
            )
            return

    # --- Build latest feature vector & predict ---
    latest_features = data[feature_cols].iloc[[-1]].copy()
    latest_features_scaled = pd.DataFrame(
        scaler.transform(latest_features),
        columns=feature_cols, index=latest_features.index,
    )
    latest_features_scaled = latest_features_scaled.reindex(columns=feature_names, fill_value=0)

    proba = float(best_model.predict_proba(latest_features_scaled)[0, 1])
    pred = int(proba >= 0.5)

    if proba > 0.70 or proba < 0.30:
        strength = "STRONG"
    elif proba > 0.55 or proba < 0.45:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    # --- Section 3: Prediction ---
    st.divider()
    st.header("🎯 Prediction")
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", f"{'🟢 UP' if pred == 1 else '🔴 DOWN'}")
    c2.metric("Confidence", f"{max(proba, 1 - proba) * 100:.1f}%")
    c3.metric("Signal Strength", strength)

    # --- Section 4: SHAP ---
    st.divider()
    st.subheader("Why this prediction?")
    shap_vals = explainer.shap_values(latest_features_scaled)
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.array(base).flatten()[0])

    explanation = shap.Explanation(
        values=shap_vals[0],
        base_values=base,
        data=latest_features_scaled.iloc[0].values,
        feature_names=list(latest_features_scaled.columns),
    )
    fig_shap = plt.figure()
    shap.plots.waterfall(explanation, max_display=10, show=False)
    st.pyplot(fig_shap, clear_figure=True)
    st.markdown(
        "**Red bars** push the prediction toward **UP**; "
        "**blue bars** push it toward **DOWN**."
    )

    # --- Section 5: Backtest ---
    st.divider()
    st.subheader("📈 Model Backtest")
    if not retrain:
        from trainer import prepare_data, train_and_evaluate
        X_train, X_test, y_train, y_test, _, _ = prepare_data(data, feature_cols)
        best_model.fit(X_train, y_train)

    y_test_pred = best_model.predict(
        pd.DataFrame(scaler.transform(data[feature_cols].iloc[-len(data[feature_cols])//5:]),
                     columns=feature_cols)
        .reindex(columns=feature_names, fill_value=0)
    ) if not retrain else best_model.predict(
        pd.DataFrame(scaler.transform(data[feature_cols].iloc[-len(y_test):]),
                     columns=feature_cols).reindex(columns=feature_names, fill_value=0)
    )

    test_slice = data.iloc[-len(y_test_pred):].copy()
    test_slice["daily_return"] = test_slice["Close"].pct_change()
    test_slice["strategy_return"] = test_slice["daily_return"] * pd.Series(y_test_pred, index=test_slice.index)
    test_slice["cumulative_market"] = (1 + test_slice["daily_return"]).cumprod()
    test_slice["cumulative_strategy"] = (1 + test_slice["strategy_return"]).cumprod()
    test_slice = test_slice.dropna()

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=test_slice["Date"], y=test_slice["cumulative_market"],
                                name="Buy & Hold", line=dict(color="gray")))
    fig_bt.add_trace(go.Scatter(x=test_slice["Date"], y=test_slice["cumulative_strategy"],
                                name="Model Strategy", line=dict(color="green")))
    fig_bt.update_layout(title="Cumulative Returns: Model vs Buy & Hold",
                         yaxis_title="Cumulative Return", height=400)
    st.plotly_chart(fig_bt, use_container_width=True)
    st.caption("⚠️ Past performance does not guarantee future results.")

    # --- Footer ---
    st.divider()
    st.caption("Built with yfinance · XGBoost · SHAP · Streamlit")
    st.caption("⚠️ This is an educational project, not financial advice.")


if __name__ == "__main__":
    main()
