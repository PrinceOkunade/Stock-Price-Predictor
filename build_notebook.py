"""Build stock_prediction.ipynb with all 13 cells per CLAUDE.md spec."""
import json, os

cells = []

def code_cell(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)})

def md_cell(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

md_cell("# Stock Price Direction Prediction\n\nEnd-to-end ML pipeline using **live data** from Yahoo Finance API.\n")

# Cell 1
md_cell("## Cell 1 — Imports")
code_cell("""import warnings
warnings.filterwarnings('ignore')

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import yfinance as yf
import ta
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, f1_score, accuracy_score)
from xgboost import XGBClassifier
import shap
import joblib

sys.path.insert(0, '../src')
from data_fetcher import fetch_stock_data, get_latest_price
from feature_engineer import engineer_features
from trainer import prepare_data, train_and_evaluate, tune_best_model, save_artifacts

os.makedirs('../images', exist_ok=True)
os.makedirs('../models', exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

print("All libraries loaded successfully.")
""")

# Cell 2
md_cell("## Cell 2 — Data Fetching (LIVE API CALL)")
code_cell("""df = fetch_stock_data("AAPL", period="2y")

print(f"\\nShape: {df.shape}")
print(f"\\nHead:")
print(df.head())
print(f"\\nTail:")
print(df.tail())
print(f"\\n*** Data fetched LIVE from Yahoo Finance API at {datetime.now()} ***")
""")

# Cell 3
md_cell("## Cell 3 — Stock Price Visualisation")
code_cell("""# Plot 1 — stock_price_history.png
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['Date'], df['Close'], label='Close Price', color='#2c3e50', linewidth=1.5)
ax.plot(df['Date'], df['Close'].rolling(20).mean(), label='20-day SMA', color='#e74c3c', linewidth=1)
ax.plot(df['Date'], df['Close'].rolling(50).mean(), label='50-day SMA', color='#3498db', linewidth=1)
ax.set_title('AAPL — Stock Price History with Moving Averages')
ax.set_xlabel('Date')
ax.set_ylabel('Price ($)')
ax.legend()
plt.tight_layout()
plt.savefig('../images/stock_price_history.png', bbox_inches='tight')
plt.show()
print("Insight: Moving average crossovers often signal trend changes — when the 20-day SMA crosses above the 50-day, it's a bullish signal.")
""")

code_cell("""# Plot 2 — volume_history.png
colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' for i in range(len(df))]
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(df['Date'], df['Volume'], color=colors, alpha=0.7, width=1)
ax.set_title('AAPL — Daily Trading Volume')
ax.set_xlabel('Date')
ax.set_ylabel('Volume')
plt.tight_layout()
plt.savefig('../images/volume_history.png', bbox_inches='tight')
plt.show()
print("Insight: Volume spikes often accompany major price moves — high volume confirms trend strength.")
""")

code_cell("""# Plot 3 — daily_returns_distribution.png
daily_ret = df['Close'].pct_change().dropna()
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(daily_ret, bins=50, alpha=0.6, color='#3498db', edgecolor='white', density=True)
sns.kdeplot(daily_ret, ax=ax, color='#e74c3c', linewidth=2)
ax.axvline(daily_ret.mean(), color='green', linestyle='--', label=f'Mean: {daily_ret.mean():.4f}')
ax.axvline(daily_ret.mean() + 2*daily_ret.std(), color='orange', linestyle='--', label=f'+2σ: {daily_ret.mean()+2*daily_ret.std():.4f}')
ax.axvline(daily_ret.mean() - 2*daily_ret.std(), color='orange', linestyle='--', label=f'-2σ: {daily_ret.mean()-2*daily_ret.std():.4f}')
ax.set_title('AAPL — Daily Returns Distribution')
ax.set_xlabel('Daily Return')
ax.legend()
plt.tight_layout()
plt.savefig('../images/daily_returns_distribution.png', bbox_inches='tight')
plt.show()
print(f"Insight: Returns are roughly normally distributed with mean {daily_ret.mean():.4f} and std {daily_ret.std():.4f}. Extreme moves beyond ±2σ are rare but impactful.")
""")

# Cell 4
md_cell("## Cell 4 — Feature Engineering")
code_cell("""data, feature_cols = engineer_features(df)

print(f"\\nFeature columns ({len(feature_cols)}):")
for f in feature_cols:
    print(f"  - {f}")
""")

code_cell("""# Plot 4 — feature_correlation.png
corr_with_target = data[feature_cols + ['target']].corr()['target'].drop('target').abs().sort_values(ascending=False)
top15 = corr_with_target.head(15)

fig, ax = plt.subplots(figsize=(10, 8))
top15_cols = list(top15.index) + ['target']
sns.heatmap(data[top15_cols].corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Top 15 Features — Correlation with Target')
plt.tight_layout()
plt.savefig('../images/feature_correlation.png', bbox_inches='tight')
plt.show()
print("Insight: The features most correlated with tomorrow's direction help us understand which technical signals the model will rely on.")
""")

# Cell 5
md_cell("## Cell 5 — Technical Indicators Visualisation")
code_cell("""# Plot 5 — technical_indicators.png (last 120 days)
recent = data.tail(120).copy()

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Row 1: Price + Bollinger Bands
axes[0].plot(recent['Date'], recent['Close'], label='Close', color='#2c3e50')
axes[0].fill_between(recent['Date'], recent['bb_high'], recent['bb_low'], alpha=0.2, color='#3498db', label='Bollinger Bands')
axes[0].set_title('Price with Bollinger Bands')
axes[0].set_ylabel('Price ($)')
axes[0].legend(loc='upper left')

# Row 2: RSI
axes[1].plot(recent['Date'], recent['rsi_14'], color='#8e44ad', linewidth=1.5)
axes[1].axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
axes[1].axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
axes[1].fill_between(recent['Date'], 30, 70, alpha=0.1, color='gray')
axes[1].set_title('RSI (14)')
axes[1].set_ylabel('RSI')
axes[1].legend(loc='upper left')

# Row 3: MACD
axes[2].plot(recent['Date'], recent['macd'], label='MACD', color='#2980b9')
axes[2].plot(recent['Date'], recent['macd_signal'], label='Signal', color='#e74c3c')
axes[2].bar(recent['Date'], recent['macd_diff'], alpha=0.4, color='gray', label='Histogram')
axes[2].set_title('MACD')
axes[2].set_ylabel('MACD')
axes[2].legend(loc='upper left')

plt.tight_layout()
plt.savefig('../images/technical_indicators.png', bbox_inches='tight')
plt.show()
print("Insight: RSI above 70 signals overbought (potential reversal down), MACD crossing above signal is bullish.")
""")

# Cell 6
md_cell("## Cell 6 — Data Preparation (Time-Based Split)")
code_cell("""X_train, X_test, y_train, y_test, scaler, fnames = prepare_data(data, feature_cols)
""")

# Cell 7
md_cell("## Cell 7 — Model Training & Comparison")
code_cell("""results = train_and_evaluate(X_train, y_train, X_test, y_test)
""")

code_cell("""# model_comparison.png
metrics_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [r['acc'] for r in results.values()],
    'F1 Score': [r['f1'] for r in results.values()],
    'ROC-AUC': [r['auc'] for r in results.values()],
})

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(metrics_df))
width = 0.25
for i, m in enumerate(['Accuracy', 'F1 Score', 'ROC-AUC']):
    ax.bar(x + i*width, metrics_df[m], width, label=m)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics_df['Model'])
ax.set_ylabel('Score')
ax.set_title('Model Comparison')
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('../images/model_comparison.png', bbox_inches='tight')
plt.show()
""")

code_cell("""# roc_curves.png
fig, ax = plt.subplots(figsize=(9, 7))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    ax.plot(fpr, tpr, label=f"{name} (AUC = {r['auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All Models')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('../images/roc_curves.png', bbox_inches='tight')
plt.show()
""")

# Cell 8
md_cell("## Cell 8 — Hyperparameter Tuning (XGBoost)")
code_cell("""best_model = tune_best_model(X_train, y_train, X_test, y_test)
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
""")

# Cell 9
md_cell("## Cell 9 — Detailed Evaluation")
code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'], ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

prec, rec, _ = precision_recall_curve(y_test, y_proba)
axes[1].plot(rec, prec, color='#3498db', linewidth=2)
axes[1].fill_between(rec, prec, alpha=0.3, color='#3498db')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')

plt.tight_layout()
plt.savefig('../images/evaluation_detailed.png', bbox_inches='tight')
plt.show()
""")

# Cell 10
md_cell("## Cell 10 — SHAP Explainability")
code_cell("""explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, plot_type='bar', max_display=15, show=False)
plt.tight_layout()
plt.savefig('../images/shap_feature_importance.png', bbox_inches='tight')
plt.show()

plt.figure()
shap.summary_plot(shap_values, X_test, max_display=15, show=False)
plt.tight_layout()
plt.savefig('../images/shap_beeswarm.png', bbox_inches='tight')
plt.show()

idx = int(np.argmax(y_proba))
expected_value = explainer.expected_value
if isinstance(expected_value, (list, np.ndarray)):
    expected_value = float(np.array(expected_value).flatten()[0])

explanation = shap.Explanation(
    values=shap_values[idx],
    base_values=expected_value,
    data=X_test.iloc[idx].values,
    feature_names=list(X_test.columns)
)
plt.figure()
shap.plots.waterfall(explanation, max_display=10, show=False)
plt.tight_layout()
plt.savefig('../images/shap_waterfall.png', bbox_inches='tight')
plt.show()

print(f"Highest UP-probability sample index: {idx}")
print(f"Predicted UP probability: {y_proba[idx]:.4f}")
print(f"Actual label: {'Up' if y_test.iloc[idx]==1 else 'Down'}")
""")

# Cell 11
md_cell("## Cell 11 — Save Artifacts")
code_cell("""save_artifacts(best_model, scaler, fnames, path='../models/')
""")

# Cell 12
md_cell("## Cell 12 — Live Prediction Demo")
code_cell("""latest_df = fetch_stock_data("AAPL", period="2y")
latest_data, latest_fcols = engineer_features(latest_df)

latest_row = latest_data[latest_fcols].iloc[[-1]]
latest_scaled = pd.DataFrame(
    scaler.transform(latest_row), columns=latest_fcols, index=latest_row.index
)
live_proba = best_model.predict_proba(latest_scaled)[0, 1]
live_pred = "UP" if live_proba >= 0.5 else "DOWN"
live_close = float(latest_df['Close'].iloc[-1])

print("=" * 50)
print(f"LIVE PREDICTION ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
print("=" * 50)
print(f"Stock       : AAPL")
print(f"Latest Close: ${live_close:.2f}")
print(f"Predicted   : {live_pred}")
print(f"Confidence  : {max(live_proba, 1-live_proba)*100:.1f}%")
print(f"\\nThis prediction was made using data fetched LIVE from Yahoo Finance.")
""")

# Cell 13
md_cell("## Cell 13 — Notebook Summary")
code_cell("""mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1][:5]
top_features = [(X_test.columns[i], float(mean_abs_shap[i])) for i in top_idx]

img_count = len([f for f in os.listdir('../images') if f.endswith('.png')])
model_count = len([f for f in os.listdir('../models') if f.endswith('.pkl')])

print("=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)
print(f"Data source      : Yahoo Finance API (LIVE)")
print(f"Ticker           : AAPL")
print(f"Date range       : {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Features         : {len(feature_cols)}")
print(f"Best model       : XGBoost (tuned)")
print(f"Test ROC-AUC     : {roc_auc_score(y_test, y_proba):.4f}")
print(f"Test F1 Score    : {f1_score(y_test, y_pred):.4f}")
print(f"Test Accuracy    : {accuracy_score(y_test, y_pred):.4f}")
print()
print("Top 5 features (by mean |SHAP|):")
for name, val in top_features:
    print(f"  - {name:<25} {val:.4f}")
print()
print(f"Images saved      : {img_count}")
print(f"Model artifacts   : {model_count}")
print(f"Timestamp         : {datetime.now()}")
print("=" * 60)
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"}
    },
    "nbformat": 4, "nbformat_minor": 5
}

out = os.path.join(os.path.dirname(__file__), "notebooks", "stock_prediction.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print(f"Wrote {out} with {len(cells)} cells.")
