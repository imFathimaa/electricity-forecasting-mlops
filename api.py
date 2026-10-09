"""Phase 2 - Day 2: FastAPI service for next-day peak demand (Tamil Nadu).

Run:  uvicorn api:app --reload --port 8000
Docs: http://127.0.0.1:8000/docs
"""
import datetime
import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    import holidays
    TN = holidays.India(subdiv="TN", years=range(2020, 2031))
except Exception:
    TN = {}

app = FastAPI(title="Tamil Nadu Peak Demand Forecast")

model = joblib.load("models/model.pkl")
with open("models/features.json") as f:
    FEATURES = json.load(f)


class PredictRequest(BaseModel):
    date: datetime.date = Field(description="Date to predict, e.g. 2025-04-01")
    prev_peak: float = Field(gt=0, description="Yesterday's peak demand (MW)")
    roll7: float = Field(gt=0, description="Average peak of the last 7 days (MW)")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    d = req.date
    festival = 1 if d in TN else 0
    row = pd.DataFrame([{
        "day_of_week": d.weekday(),
        "month": d.month,
        "festival": festival,
        "prev_peak": req.prev_peak,
        "roll7": req.roll7,
    }])[FEATURES]
    value = float(model.predict(row)[0])
    return {"date": str(d), "predicted_peak_mw": round(value), "festival": bool(festival)}