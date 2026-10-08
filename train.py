"""Phase 2 - Day 1: training script with MLflow tracking.

Reads tn_history.csv (columns: date, peak), builds features, trains
Linear Regression and Random Forest, logs every run to MLflow, picks the
better model by MAPE and saves it to models/.   Run:  python train.py
"""
import json
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_FILE = "tn_history.csv"
MODEL_DIR = "models"
FEATURES = ["day_of_week", "month", "festival", "prev_peak", "roll7"]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("tn-peak-demand")


def load_data(path=DATA_FILE):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated()]
    df = df.asfreq("D")
    df.loc[df["peak"] <= 5000, "peak"] = np.nan   # impossible values (e.g. 933)
    df["peak"] = df["peak"].interpolate(limit=3)
    return df.dropna()


def festival_flags(index):
    try:
        import holidays
        years = range(index.year.min(), index.year.max() + 2)
        try:
            tn = holidays.India(subdiv="TN", years=years)
        except Exception:
            tn = holidays.India(years=years)
        return [1 if d.date() in tn else 0 for d in index]
    except ImportError:
        print("WARNING: 'holidays' not installed -> festival flag = 0 (pip install holidays)")
        return [0] * len(index)


def build_features(data):
    feat = data.copy()
    feat["day_of_week"] = feat.index.dayofweek
    feat["month"] = feat.index.month
    feat["festival"] = festival_flags(feat.index)
    feat["prev_peak"] = feat["peak"].shift(1)
    feat["roll7"] = feat["peak"].shift(1).rolling(7).mean()
    return feat.dropna()


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "MAE": round(float(mean_absolute_error(y_test, pred)), 1),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 1),
        "MAPE": round(float(np.mean(np.abs((y_test - pred) / y_test)) * 100), 2),
    }


def main():
    data = load_data()
    feat = build_features(data)
    X, y = feat[FEATURES], feat["peak"]

    split = int(len(feat) * 0.8)                    # time-based split, no shuffle
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"Rows: {len(feat)} | train {len(X_train)} | test {len(X_test)}")
    print(f"Date range: {feat.index.min().date()} to {feat.index.max().date()}")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }
    common_params = {
        "features": ",".join(FEATURES),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "data_end": str(feat.index.max().date()),
    }

    results = {}
    for name, model in models.items():
        with mlflow.start_run(run_name=name):       # one MLflow run per model
            model.fit(X_train, y_train)
            results[name] = evaluate(model, X_test, y_test)
            mlflow.log_param("model", name)
            mlflow.log_params(common_params)
            if name == "Random Forest":
                mlflow.log_param("n_estimators", 200)
            mlflow.log_metrics(results[name])

    table = pd.DataFrame(results).T
    print(table)

    best_name = table["MAPE"].idxmin()
    print("Best model:", best_name)

    best_model = models[best_name].fit(X, y)         # refit on all data AFTER scoring
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, f"{MODEL_DIR}/model.pkl")
    with open(f"{MODEL_DIR}/features.json", "w") as f:
        json.dump(FEATURES, f)
    table.to_csv(f"{MODEL_DIR}/results.csv")
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump({"best_model": best_name, **results[best_name]}, f, indent=2)

    with mlflow.start_run(run_name="final-" + best_name):   # the model we actually save
        mlflow.log_param("model", best_name)
        mlflow.log_params(common_params)
        mlflow.log_metrics(results[best_name])
        mlflow.log_artifact(f"{MODEL_DIR}/model.pkl")
        mlflow.log_artifact(f"{MODEL_DIR}/features.json")

    print("Saved: models/model.pkl, features.json, results.csv, metrics.json")
    print("Logged to MLflow (experiment: tn-peak-demand)")


if __name__ == "__main__":
    main()