"""Phase 2 - Day 3: simple model drift monitoring.

Run:   python monitor.py
Demo:  python monitor.py --simulate-drift      (pretends demand jumped 15%)
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from train import FEATURES, build_features, load_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tn_history.csv")
    ap.add_argument("--window", type=int, default=30, help="recent days to check")
    ap.add_argument("--threshold", type=float, default=1.5,
                    help="alert if recent MAPE > threshold x baseline MAPE")
    ap.add_argument("--simulate-drift", action="store_true",
                    help="demo: make actual demand 15%% higher than the model expects")
    args = ap.parse_args()

    model = joblib.load("models/model.pkl")
    with open("models/metrics.json") as f:
        baseline = json.load(f)["MAPE"]            # test MAPE from training time

    feat = build_features(load_data(args.data))
    recent = feat.iloc[-args.window:]
    actual = recent["peak"].copy()
    pred = model.predict(recent[FEATURES])

    if args.simulate_drift:
        actual = actual * 1.15
        print("SIMULATION: actual demand multiplied by 1.15\n")

    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    bias = float(np.mean((actual - pred) / actual) * 100)   # + = model under-predicts
    limit = args.threshold * baseline
    drift = mape > limit

    print(f"Checked: last {len(recent)} days "
          f"({recent.index.min().date()} to {recent.index.max().date()})")
    print(f"Baseline MAPE (test set at training): {baseline:.2f}%")
    print(f"Recent MAPE: {mape:.2f}%   alert limit: {limit:.2f}%")
    print(f"Average bias: {bias:+.2f}%  (+ = model predicts too low)")

    report = {"recent_mape": round(mape, 2), "baseline_mape": baseline,
              "limit": round(limit, 2), "bias_pct": round(bias, 2),
              "drift_detected": bool(drift)}
    with open("monitoring_report.json", "w") as f:
        json.dump(report, f, indent=2)

    if drift:
        print("\nALERT: model drift detected - retraining recommended.")
        raise SystemExit(1)
    print("\nOK: model error is within the normal range.")


if __name__ == "__main__":
    main()
