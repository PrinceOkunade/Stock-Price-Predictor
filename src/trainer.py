"""Module 3: Trains, tunes, and evaluates models with time-series discipline."""
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def prepare_data(df, feature_cols, test_ratio=0.2):
    X = df[feature_cols].copy()
    y = df["target"].copy()

    split_idx = int(len(df) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"TRAIN: {df.iloc[:split_idx]['Date'].min().date()} to {df.iloc[:split_idx]['Date'].max().date()}")
    print(f"TEST:  {df.iloc[split_idx:]['Date'].min().date()} to {df.iloc[split_idx:]['Date'].max().date()}")
    print(f"No future data leaks into training — time-based split enforced.")
    print(f"Train target:\n{y_train.value_counts()}")
    print(f"Test target:\n{y_test.value_counts()}")
    return X_train, X_test, y_train, y_test, scaler, feature_cols


def train_and_evaluate(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42, eval_metric="logloss",
        ),
    }
    results = {}
    for name, model in models.items():
        print("=" * 60)
        print(f"Training: {name}")
        print("=" * 60)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        results[name] = {
            "model": model, "acc": acc, "f1": f1, "auc": auc,
            "y_pred": y_pred, "y_proba": y_proba,
        }
        print(f"Accuracy : {acc:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))
    return results


def tune_best_model(X_train, y_train, X_test, y_test):
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200, 300],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    xgb = XGBClassifier(random_state=42, eval_metric="logloss")
    grid = GridSearchCV(
        xgb, param_grid,
        scoring="roc_auc",
        cv=TimeSeriesSplit(n_splits=5),
        n_jobs=-1, verbose=1,
    )
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    print(f"\nTuned XGBoost on test set:")
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))
    return best_model


def save_artifacts(model, scaler, feature_names, path="../models/"):
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, "stock_model.pkl"))
    joblib.dump(scaler, os.path.join(path, "scaler.pkl"))
    joblib.dump(list(feature_names), os.path.join(path, "feature_names.pkl"))
    print(f"Saved: {path}stock_model.pkl")
    print(f"Saved: {path}scaler.pkl")
    print(f"Saved: {path}feature_names.pkl")


def retrain_pipeline(ticker="AAPL", period="2y", models_path="../models/"):
    from data_fetcher import fetch_stock_data
    from feature_engineer import engineer_features

    print(f"\n{'='*60}")
    print(f"RETRAINING PIPELINE — {ticker} — {datetime.now()}")
    print(f"{'='*60}\n")

    df = fetch_stock_data(ticker, period=period)
    data, feature_cols = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler, fnames = prepare_data(data, feature_cols)
    results = train_and_evaluate(X_train, y_train, X_test, y_test)
    best_model = tune_best_model(X_train, y_train, X_test, y_test)
    save_artifacts(best_model, scaler, fnames, path=models_path)
    print(f"\nRetraining complete at {datetime.now()}")
    return best_model, scaler, fnames, results, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    retrain_pipeline("AAPL")
