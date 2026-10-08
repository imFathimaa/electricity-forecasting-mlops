import json
import joblib
import pandas as pd
from flask import Flask, request, render_template_string

try:
    import holidays
    TN = holidays.India(subdiv="TN", years=range(2020, 2031))
except Exception:
    TN = {}

app = Flask(__name__)

model = joblib.load("models/model.pkl")
with open("models/features.json") as f:
    FEATURES = json.load(f)

PAGE = """
<!doctype html>
<title>TN Peak Demand Forecast</title>
<body style="font-family:Arial;max-width:480px;margin:40px auto">
<h2>Tamil Nadu next-day peak demand</h2>
<form method="post">
  <p>Date to predict<br><input type="date" name="date" required value="{{ f.date }}"></p>
  <p>Yesterday's peak (MW)<br><input type="number" step="any" name="prev_peak" required value="{{ f.prev_peak }}"></p>
  <p>Average of last 7 days (MW)<br><input type="number" step="any" name="roll7" required value="{{ f.roll7 }}"></p>
  <button type="submit">Predict</button>
</form>
{% if result %}
  <h3>Predicted peak: {{ result }} MW</h3>
  <p>Holiday / festival: {{ "Yes" if festival else "No" }}</p>
{% endif %}
{% if error %}<p style="color:red">{{ error }}</p>{% endif %}
</body>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    f = {"date": "", "prev_peak": "", "roll7": ""}
    result = error = None
    festival = 0
    if request.method == "POST":
        f = {k: request.form.get(k, "") for k in f}
        try:
            d = pd.to_datetime(f["date"])
            festival = 1 if d.date() in TN else 0
            row = pd.DataFrame([{
                "day_of_week": d.dayofweek,
                "month": d.month,
                "festival": festival,
                "prev_peak": float(f["prev_peak"]),
                "roll7": float(f["roll7"]),
            }])[FEATURES]
            result = round(float(model.predict(row)[0]))
        except Exception as e:
            error = "Check your inputs: " + str(e)
    return render_template_string(PAGE, f=f, result=result, error=error, festival=festival)

if __name__ == "__main__":
    app.run(debug=True)
